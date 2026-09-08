"""Qualidade de homografia sobre linhas anotadas, com procedencia explicita."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from jsonschema import Draft202012Validator, FormatChecker

from contracts import Clip, FieldGeometry
from edge.modules.field.geometry import FieldProjectionError, pitch_to_image

SCHEMA = Path(__file__).with_name("field-calibration-v1.schema.json")
JAC_TOLERANCE_PX = 5


class FieldLabelError(ValueError): ...


@dataclass(frozen=True)
class FieldPolyline:
    element: str
    image_xy: tuple[tuple[float, float], ...]
    pitch_xy: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class FieldLabel:
    clip_id: str
    match_id: str
    frame_index: int
    frame_width: int
    frame_height: int
    lines: tuple[FieldPolyline, ...]
    video_sha256: str
    annotator: str
    reviewed_by: str | None
    quality_flags: tuple[str, ...]

    @property
    def human_confirmed(self) -> bool:
        return (
            "human-annotated" in self.quality_flags
            and "requires-human-review" not in self.quality_flags
        )

    @property
    def assisted(self) -> bool:
        return bool(
            {"white-paint-alignment-reviewed", "temporal-propagation-reviewed"}
            & set(self.quality_flags)
        )


@dataclass(frozen=True)
class FieldReport:
    n_labels: int
    n_proposal_labels: int
    n_excluded_proposals: int
    n_assisted_frames: int
    includes_proposals: bool
    n_frames: int
    n_matches: int
    n_available: int
    n_unavailable: int
    completion: float
    completion_wilson_95: tuple[float, float]
    jac_at_5: float
    score: float
    line_error_median_px: float | None
    line_error_p95_px: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "n_labels": self.n_labels,
            "n_proposal_labels": self.n_proposal_labels,
            "n_excluded_proposals": self.n_excluded_proposals,
            "n_assisted_frames": self.n_assisted_frames,
            "evaluation_kind": (
                "exploratory-with-proposals" if self.includes_proposals else "human-confirmed"
            ),
            "n_frames": self.n_frames,
            "n_matches": self.n_matches,
            "n_available": self.n_available,
            "n_unavailable": self.n_unavailable,
            "completion": self.completion,
            "completion_wilson_95": list(self.completion_wilson_95),
            "jac_at_5": self.jac_at_5,
            "score": self.score,
            "line_error_median_px": self.line_error_median_px,
            "line_error_p95_px": self.line_error_p95_px,
        }


def read_field_labels(path: Path) -> tuple[FieldLabel, ...]:
    if not path.is_file():
        raise FieldLabelError(f"{path} nao existe")
    schema = cast("dict[str, Any]", json.loads(SCHEMA.read_text(encoding="utf-8")))
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    labels: list[FieldLabel] = []
    seen: set[tuple[str, int]] = set()
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            document = cast("dict[str, Any]", json.loads(raw))
        except json.JSONDecodeError as error:
            raise FieldLabelError(f"{path.name}:{number}: JSON invalido: {error.msg}") from error
        errors = sorted(
            cast("Any", validator).iter_errors(document),
            key=lambda error: list(error.path),
        )
        if errors:
            raise FieldLabelError(f"{path.name}:{number}: {errors[0].message}")
        label = _label(document)
        key = (label.clip_id, label.frame_index)
        if key in seen:
            raise FieldLabelError(
                f"{path.name}:{number}: quadro {label.frame_index} do clipe aparece duas vezes"
            )
        seen.add(key)
        labels.append(label)
    return tuple(labels)


def measure(
    labels: tuple[FieldLabel, ...],
    geometries: dict[tuple[str, int], FieldGeometry],
    *,
    include_proposals: bool = False,
) -> FieldReport:
    selected = evaluation_labels(labels, include_proposals=include_proposals)
    errors: list[float] = []
    jaccards: list[float] = []
    available = 0
    for label in selected:
        geometry = geometries.get((label.clip_id, label.frame_index))
        if geometry is None or geometry.homography is None:
            continue
        try:
            projected = tuple(
                tuple(pitch_to_image(geometry.homography, point) for point in line.pitch_xy)
                for line in label.lines
            )
        except FieldProjectionError:
            continue
        available += 1
        frame_errors = tuple(
            tuple(_distance_to_polyline(point, predicted) for point in line.image_xy)
            for line, predicted in zip(label.lines, projected, strict=True)
        )
        errors.extend(error for line_errors in frame_errors for error in line_errors)
        jaccards.append(_jaccard(frame_errors))
    total = len(selected)
    completion = available / total if total else 0.0
    jac = sum(jaccards) / len(jaccards) if jaccards else 0.0
    return FieldReport(
        n_labels=len(labels),
        n_proposal_labels=sum(not label.human_confirmed for label in labels),
        n_excluded_proposals=len(labels) - total,
        n_assisted_frames=sum(label.assisted for label in selected),
        includes_proposals=include_proposals,
        n_frames=total,
        n_matches=len({label.match_id for label in selected}),
        n_available=available,
        n_unavailable=total - available,
        completion=completion,
        completion_wilson_95=_wilson(available, total),
        jac_at_5=jac,
        score=completion * jac,
        line_error_median_px=None if not errors else float(np.median(errors)),
        line_error_p95_px=None if not errors else float(np.percentile(errors, 95)),
    )


def evaluation_labels(
    labels: tuple[FieldLabel, ...], *, include_proposals: bool = False
) -> tuple[FieldLabel, ...]:
    return (
        labels if include_proposals else tuple(label for label in labels if label.human_confirmed)
    )


def validate_label_video(label: FieldLabel, clip: Clip, video_sha256: str) -> None:
    if label.clip_id != clip.clip_id or label.match_id != clip.match_id:
        raise FieldLabelError("anotacao pertence a outro clipe ou partida")
    if label.video_sha256 != video_sha256:
        raise FieldLabelError("video_sha256 da anotacao difere do video medido")
    if (label.frame_width, label.frame_height) != (clip.width, clip.height):
        raise FieldLabelError("dimensoes da anotacao diferem do video medido")
    if label.frame_index >= clip.frame_count:
        raise FieldLabelError("frame_index da anotacao esta fora do video medido")


def _label(document: dict[str, Any]) -> FieldLabel:
    visible = set(cast("list[str]", document["visible_pitch_elements"]))
    lines: list[FieldPolyline] = []
    for raw in cast("list[dict[str, Any]]", document["line_points"]):
        image = _points(raw["image_xy"])
        pitch = _points(raw["pitch_xy"])
        if len(image) != len(pitch):
            raise FieldLabelError(
                f"elemento {raw['element']!r} exige a mesma quantidade de image_xy e pitch_xy"
            )
        if raw["element"] not in visible:
            raise FieldLabelError(
                f"elemento {raw['element']!r} nao aparece em visible_pitch_elements"
            )
        lines.append(FieldPolyline(str(raw["element"]), image, pitch))
    return FieldLabel(
        clip_id=str(document["clip_id"]),
        match_id=str(document["match_id"]),
        frame_index=int(document["frame_index"]),
        frame_width=int(document["frame_width"]),
        frame_height=int(document["frame_height"]),
        lines=tuple(lines),
        video_sha256=str(document["video_sha256"]),
        annotator=str(document["annotator"]),
        reviewed_by=cast("str | None", document["reviewed_by"]),
        quality_flags=tuple(cast("list[str]", document["quality_flags"])),
    )


def _points(raw: object) -> tuple[tuple[float, float], ...]:
    return tuple((float(point[0]), float(point[1])) for point in cast("list[list[float]]", raw))


def _jaccard(errors: tuple[tuple[float, ...], ...]) -> float:
    """JaC@5 do protocolo SoccerNet para os elementos visiveis anotados.

    Um elemento inteiro e verdadeiro positivo apenas quando todos os seus pontos anotados ficam a
    menos de cinco pixels da polilinha projetada. Elemento presente mas fora do limiar conta como
    falso positivo. A completude de frames sem homografia e medida separadamente.
    """
    true_positive = sum(
        bool(line) and all(value < JAC_TOLERANCE_PX for value in line) for line in errors
    )
    false_positive = len(errors) - true_positive
    denominator = true_positive + false_positive
    return true_positive / denominator if denominator else 0.0


def _distance_to_polyline(
    point: tuple[float, float], polyline: tuple[tuple[float, float], ...]
) -> float:
    target = np.asarray(point, dtype=np.float64)
    distances: list[float] = []
    for start, end in zip(polyline, polyline[1:], strict=False):
        origin = np.asarray(start, dtype=np.float64)
        vector = np.asarray(end, dtype=np.float64) - origin
        length_squared = float(vector @ vector)
        if length_squared <= 1e-12:
            distances.append(float(np.linalg.norm(target - origin)))
            continue
        position = min(1.0, max(0.0, float((target - origin) @ vector) / length_squared))
        distances.append(float(np.linalg.norm(target - (origin + position * vector))))
    return min(distances)


def _wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))
