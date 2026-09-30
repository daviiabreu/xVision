"""Leitura e avaliacao da anotacao privada de trajetoria da bola."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from jsonschema import Draft202012Validator, FormatChecker

from contracts import BallDetection
from mlops.evaluate import binomial_bilateral

SCHEMA = Path(__file__).with_name("ball-trajectory-v1.schema.json")
Visibility = Literal["visible", "occluded", "not-visible", "ambiguous"]
Outcome = Literal["hit", "miss", "absent"]
OUTCOMES: frozenset[str] = frozenset({"hit", "miss", "absent"})
FrameKey = tuple[str, int]


class BallTrajectoryLabelError(ValueError): ...


@dataclass(frozen=True)
class BallTrajectoryLabel:
    clip_id: str
    match_id: str
    video_sha256: str
    frame_index: int
    timestamp_s: float
    center_xy: tuple[float, float] | None
    apparent_diameter_px: float | None
    visibility: Visibility
    trajectory_segment: int | None
    is_keyframe: bool
    motion_blur: bool
    camera_cut: bool
    annotator: str
    reviewed_by: str | None
    quality_flags: tuple[str, ...]


@dataclass(frozen=True)
class BallTrajectoryReport:
    n_frames: int
    n_clips: int
    n_matches: int
    n_visible: int
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float
    pck_diameter: float
    pck_wilson_95: tuple[float, float]
    detected_points: int
    interpolated_points: int
    missing_points: int
    predicted_segments: int
    track_switches: int
    max_visible_gap_frames: int
    center_error_median_px: float | None
    center_error_p95_px: float | None
    detected_error_median_px: float | None
    interpolated_error_median_px: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "n_frames": self.n_frames,
            "n_clips": self.n_clips,
            "n_matches": self.n_matches,
            "n_visible": self.n_visible,
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "pck_diameter": self.pck_diameter,
            "pck_wilson_95": list(self.pck_wilson_95),
            "detected_points": self.detected_points,
            "interpolated_points": self.interpolated_points,
            "missing_points": self.missing_points,
            "predicted_segments": self.predicted_segments,
            "track_switches": self.track_switches,
            "max_visible_gap_frames": self.max_visible_gap_frames,
            "center_error_median_px": self.center_error_median_px,
            "center_error_p95_px": self.center_error_p95_px,
            "detected_error_median_px": self.detected_error_median_px,
            "interpolated_error_median_px": self.interpolated_error_median_px,
        }


@dataclass(frozen=True)
class PairedComparison:
    n_frames: int
    base_hits: int
    new_hits: int
    gains: int
    losses: int
    unchanged: int
    p_value: float

    def to_dict(self) -> dict[str, object]:
        return {
            "n_frames": self.n_frames,
            "base_hits": self.base_hits,
            "new_hits": self.new_hits,
            "gains": self.gains,
            "losses": self.losses,
            "unchanged": self.unchanged,
            "p_value": self.p_value,
            "test": "sign-exact-two-sided",
        }


def validate_match_folds(match_folds: Iterable[tuple[str, int]]) -> None:
    """Recusa um split que usa clipes da mesma partida em dobras diferentes."""
    folds_by_match: dict[str, set[int]] = {}
    for match_id, fold in match_folds:
        folds_by_match.setdefault(match_id, set()).add(fold)
    split_matches = sorted(match_id for match_id, folds in folds_by_match.items() if len(folds) > 1)
    if split_matches:
        raise BallTrajectoryLabelError(
            f"partida aparece em mais de uma dobra: {', '.join(split_matches)}"
        )


def aggregate_telemetry(
    reports: Sequence[Mapping[str, int | float]],
) -> dict[str, int | float]:
    aggregate: dict[str, int | float] = {}
    for report in reports:
        for name, value in report.items():
            if name == "max_gap_frames":
                aggregate[name] = max(aggregate.get(name, 0), value)
            else:
                aggregate[name] = aggregate.get(name, 0) + value
    return aggregate


def read_ball_trajectory_labels(path: Path) -> tuple[BallTrajectoryLabel, ...]:
    if not path.is_file():
        raise BallTrajectoryLabelError(f"{path} nao existe")
    schema = cast("dict[str, Any]", json.loads(SCHEMA.read_text(encoding="utf-8")))
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    labels: list[BallTrajectoryLabel] = []
    seen: set[tuple[str, int]] = set()
    clip_identity: dict[str, tuple[str, str]] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            document = cast("dict[str, Any]", json.loads(raw))
        except json.JSONDecodeError as error:
            raise BallTrajectoryLabelError(
                f"{path.name}:{number}: JSON invalido: {error.msg}"
            ) from error
        errors = sorted(
            cast("Any", validator).iter_errors(document), key=lambda error: list(error.path)
        )
        if errors:
            context = "visible: " if document.get("visibility") == "visible" else ""
            raise BallTrajectoryLabelError(f"{path.name}:{number}: {context}{errors[0].message}")
        label = _label(document)
        key = (label.clip_id, label.frame_index)
        if key in seen:
            raise BallTrajectoryLabelError(
                f"{path.name}:{number}: quadro {label.frame_index} do clipe aparece duas vezes"
            )
        identity = (label.match_id, label.video_sha256)
        previous = clip_identity.setdefault(label.clip_id, identity)
        if previous != identity:
            raise BallTrajectoryLabelError(
                f"{path.name}:{number}: clipe mudou match_id ou video_sha256"
            )
        seen.add(key)
        labels.append(label)
    _check_temporal_order(labels, path)
    return tuple(labels)


def measure(
    labels: Sequence[BallTrajectoryLabel],
    predictions: Mapping[tuple[str, int], BallDetection],
) -> BallTrajectoryReport:
    true_positive = false_positive = false_negative = 0
    detected = interpolated = missing = 0
    errors: list[float] = []
    detected_errors: list[float] = []
    interpolated_errors: list[float] = []
    predicted_segments: set[tuple[str, int]] = set()
    switches = 0
    max_gap = 0
    by_clip: dict[str, list[BallTrajectoryLabel]] = {}

    for label in labels:
        by_clip.setdefault(label.clip_id, []).append(label)
        prediction = predictions.get((label.clip_id, label.frame_index))
        if prediction is None:
            if label.visibility == "visible":
                false_negative += 1
                missing += 1
            continue
        if prediction.source == "interpolated":
            interpolated += 1
        else:
            detected += 1
        if prediction.trajectory_segment is not None:
            predicted_segments.add((label.clip_id, prediction.trajectory_segment))
        if label.visibility == "not-visible":
            false_positive += 1
            continue
        if label.visibility != "visible" or label.center_xy is None:
            continue
        error = _center_error(prediction, label.center_xy)
        errors.append(error)
        (interpolated_errors if prediction.source == "interpolated" else detected_errors).append(
            error
        )
        if _inside_tolerance(label, prediction):
            true_positive += 1
        else:
            false_positive += 1
            false_negative += 1

    for group in by_clip.values():
        ordered = sorted(group, key=lambda label: label.frame_index)
        max_gap = max(max_gap, _largest_visible_gap(ordered, predictions))
        switches += _track_switches(ordered, predictions)

    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    f1 = _ratio(2 * precision * recall, precision + recall)
    visible = sum(label.visibility == "visible" for label in labels)
    return BallTrajectoryReport(
        n_frames=len(labels),
        n_clips=len({label.clip_id for label in labels}),
        n_matches=len({label.match_id for label in labels}),
        n_visible=visible,
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        precision=precision,
        recall=recall,
        f1=f1,
        pck_diameter=_ratio(true_positive, visible),
        pck_wilson_95=_wilson(true_positive, visible),
        detected_points=detected,
        interpolated_points=interpolated,
        missing_points=missing,
        predicted_segments=len(predicted_segments),
        track_switches=switches,
        max_visible_gap_frames=max_gap,
        center_error_median_px=_percentile(errors, 50),
        center_error_p95_px=_percentile(errors, 95),
        detected_error_median_px=_percentile(detected_errors, 50),
        interpolated_error_median_px=_percentile(interpolated_errors, 50),
    )


def frame_outcomes(
    labels: Sequence[BallTrajectoryLabel],
    predictions: Mapping[FrameKey, BallDetection],
) -> dict[FrameKey, Outcome]:
    """O desfecho de cada quadro visivel, com a tolerancia do PCK e a chave de `measure`.

    So o visivel entra porque ele e o denominador do PCK: quadro sem bola visivel nao tem acerto a
    ganhar nem a perder. `absent` fica separado de `miss` porque a correcao de cada um e outra, e no
    pareamento os dois contam como nao acerto.
    """
    outcomes: dict[FrameKey, Outcome] = {}
    for label in labels:
        if label.visibility != "visible":
            continue
        key = (label.clip_id, label.frame_index)
        prediction = predictions.get(key)
        if prediction is None:
            outcomes[key] = "absent"
        else:
            outcomes[key] = "hit" if _inside_tolerance(label, prediction) else "miss"
    return outcomes


def label_set_sha256(labels: Iterable[BallTrajectoryLabel]) -> str:
    """Impressao dos rotulos medidos, para o pareamento recusar execucoes de gabaritos diferentes.

    Cobre o que decide o desfecho, e nao so a chave: com a mesma chave e um centro corrigido numa
    revisao, o ganho seria do rotulo e nao da configuracao.
    """
    rows = [
        [
            label.clip_id,
            label.frame_index,
            label.video_sha256,
            label.visibility,
            None if label.center_xy is None else list(label.center_xy),
            label.apparent_diameter_px,
        ]
        for label in sorted(labels, key=lambda label: (label.clip_id, label.frame_index))
    ]
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode("utf-8")).hexdigest()


def outcomes_document(outcomes: Mapping[FrameKey, str]) -> dict[str, dict[str, str]]:
    """O desfecho por clipe e por quadro, na forma que o JSON aceita como chave."""
    document: dict[str, dict[str, str]] = {}
    for (clip_id, frame_index), outcome in sorted(outcomes.items()):
        document.setdefault(clip_id, {})[str(frame_index)] = outcome
    return document


def read_frame_outcomes(path: Path) -> tuple[str, dict[FrameKey, Outcome]]:
    """A impressao dos rotulos e o desfecho por quadro de uma execucao gravada com `--saida`."""
    if not path.is_file():
        raise BallTrajectoryLabelError(f"{path} nao existe")
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise BallTrajectoryLabelError(f"{path.name}: JSON invalido: {error.msg}") from error
    if not isinstance(raw, dict):
        raise BallTrajectoryLabelError(f"{path.name}: documento de base invalido")
    document = cast("dict[str, Any]", raw)
    digest = document.get("label_set_sha256")
    by_clip = document.get("frame_outcomes")
    if not isinstance(digest, str) or not isinstance(by_clip, dict):
        raise BallTrajectoryLabelError(
            f"{path.name} nao tem o desfecho por quadro: grave a base de novo com --saida"
        )
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise BallTrajectoryLabelError(f"{path.name}: label_set_sha256 invalido")
    outcomes: dict[FrameKey, Outcome] = {}
    for clip_id, frames in cast("dict[str, Any]", by_clip).items():
        if not clip_id or not isinstance(frames, dict):
            raise BallTrajectoryLabelError(f"{path.name}: mapa de quadros invalido")
        for frame_index, outcome in cast("dict[str, Any]", frames).items():
            if (
                not isinstance(outcome, str)
                or outcome not in OUTCOMES
                or not frame_index.isascii()
                or not frame_index.isdecimal()
                or (frame_index != "0" and frame_index.startswith("0"))
            ):
                raise BallTrajectoryLabelError(
                    f"{path.name}: quadro {frame_index!r} com desfecho {outcome!r} invalido"
                )
            outcomes[(str(clip_id), int(frame_index))] = cast("Outcome", outcome)
    return digest, outcomes


def compare_outcomes(base: Mapping[FrameKey, str], new: Mapping[FrameKey, str]) -> PairedComparison:
    """Ganho e perda de acerto quadro a quadro, com o teste do sinal exato nos discordantes.

    O PCK agregado esconde troca: uma configuracao pode acertar 100 quadros novos e errar 80 que a
    base acertava, e a media mostra so o saldo de 20. A informacao esta nos discordantes, e com
    eles na casa das dezenas vale o exato binomial, pelo mesmo motivo do `mcnemar` da avaliacao.
    """
    if set(base) != set(new):
        raise BallTrajectoryLabelError("as duas execucoes nao mediram os mesmos quadros visiveis")
    gains = sum(1 for key, outcome in new.items() if outcome == "hit" and base[key] != "hit")
    losses = sum(1 for key, outcome in new.items() if outcome != "hit" and base[key] == "hit")
    return PairedComparison(
        n_frames=len(new),
        base_hits=sum(1 for outcome in base.values() if outcome == "hit"),
        new_hits=sum(1 for outcome in new.values() if outcome == "hit"),
        gains=gains,
        losses=losses,
        unchanged=len(new) - gains - losses,
        p_value=binomial_bilateral(gains, losses),
    )


def _label(document: dict[str, Any]) -> BallTrajectoryLabel:
    center = (
        None
        if document["center_x"] is None or document["center_y"] is None
        else (float(document["center_x"]), float(document["center_y"]))
    )
    return BallTrajectoryLabel(
        clip_id=str(document["clip_id"]),
        match_id=str(document["match_id"]),
        video_sha256=str(document["video_sha256"]),
        frame_index=int(document["frame_index"]),
        timestamp_s=float(document["timestamp_s"]),
        center_xy=center,
        apparent_diameter_px=(
            None
            if document["apparent_diameter_px"] is None
            else float(document["apparent_diameter_px"])
        ),
        visibility=cast("Visibility", document["visibility"]),
        trajectory_segment=(
            None if document["trajectory_segment"] is None else int(document["trajectory_segment"])
        ),
        is_keyframe=bool(document["is_keyframe"]),
        motion_blur=bool(document["motion_blur"]),
        camera_cut=bool(document["camera_cut"]),
        annotator=str(document["annotator"]),
        reviewed_by=(None if document["reviewed_by"] is None else str(document["reviewed_by"])),
        quality_flags=tuple(str(flag) for flag in document["quality_flags"]),
    )


def _check_temporal_order(labels: Sequence[BallTrajectoryLabel], path: Path) -> None:
    by_clip: dict[str, list[BallTrajectoryLabel]] = {}
    for label in labels:
        by_clip.setdefault(label.clip_id, []).append(label)
    for clip_id, group in by_clip.items():
        ordered = sorted(group, key=lambda label: label.frame_index)
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if current.timestamp_s <= previous.timestamp_s:
                raise BallTrajectoryLabelError(
                    f"{path.name}: timestamps do clipe {clip_id} nao sao crescentes"
                )


def _inside_tolerance(label: BallTrajectoryLabel, prediction: BallDetection) -> bool:
    return (
        label.center_xy is not None
        and label.apparent_diameter_px is not None
        and _center_error(prediction, label.center_xy) <= label.apparent_diameter_px
    )


def _center_error(prediction: BallDetection, target: tuple[float, float]) -> float:
    center = (
        (prediction.box.x1 + prediction.box.x2) / 2.0,
        (prediction.box.y1 + prediction.box.y2) / 2.0,
    )
    return math.hypot(center[0] - target[0], center[1] - target[1])


def _largest_visible_gap(
    labels: Sequence[BallTrajectoryLabel],
    predictions: Mapping[tuple[str, int], BallDetection],
) -> int:
    largest = current = 0
    for label in labels:
        missing = (
            label.visibility == "visible"
            and (
                label.clip_id,
                label.frame_index,
            )
            not in predictions
        )
        current = current + 1 if missing else 0
        largest = max(largest, current)
    return largest


def _track_switches(
    labels: Sequence[BallTrajectoryLabel],
    predictions: Mapping[tuple[str, int], BallDetection],
) -> int:
    switches = 0
    previous_label_segment: int | None = None
    previous_prediction_segment: int | None = None
    for label in labels:
        prediction = predictions.get((label.clip_id, label.frame_index))
        segment = None if prediction is None else prediction.trajectory_segment
        if (
            not label.camera_cut
            and label.trajectory_segment is not None
            and label.trajectory_segment == previous_label_segment
            and segment is not None
            and previous_prediction_segment is not None
            and segment != previous_prediction_segment
        ):
            switches += 1
        previous_label_segment = label.trajectory_segment
        if segment is not None:
            previous_prediction_segment = segment
    return switches


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _percentile(values: Sequence[float], percentile: int) -> float | None:
    return None if not values else float(np.percentile(values, percentile))


def _wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))
