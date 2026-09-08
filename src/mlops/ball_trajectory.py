"""Leitura e avaliacao da anotacao privada de trajetoria da bola."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from jsonschema import Draft202012Validator, FormatChecker

from contracts import BallDetection

SCHEMA = Path(__file__).with_name("ball-trajectory-v1.schema.json")
Visibility = Literal["visible", "occluded", "not-visible", "ambiguous"]


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
        if label.apparent_diameter_px is not None and error <= label.apparent_diameter_px:
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
