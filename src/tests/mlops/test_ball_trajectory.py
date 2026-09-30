"""Leitor e metricas da anotacao privada de trajetoria da bola."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from contracts import BallDetection, Box
from mlops.ball_trajectory import (
    BallTrajectoryLabelError,
    aggregate_telemetry,
    compare_outcomes,
    frame_outcomes,
    label_set_sha256,
    measure,
    outcomes_document,
    read_ball_trajectory_labels,
    read_frame_outcomes,
    validate_match_folds,
)


def _document(
    *,
    frame_index: int = 12,
    timestamp_s: float = 0.4,
    visibility: str = "visible",
    center: tuple[float, float] | None = (100.0, 200.0),
    segment: int | None = 0,
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "clip_id": "opaque-clip",
        "match_id": "opaque-match",
        "video_sha256": "a" * 64,
        "frame_index": frame_index,
        "timestamp_s": timestamp_s,
        "center_x": None if center is None else center[0],
        "center_y": None if center is None else center[1],
        "apparent_diameter_px": None if center is None else 12.0,
        "visibility": visibility,
        "trajectory_segment": segment,
        "is_keyframe": True,
        "motion_blur": False,
        "camera_cut": False,
        "annotator": "reviewer-a",
        "reviewed_by": None,
        "created_at": "2026-09-03T12:00:00Z",
        "quality_flags": [],
    }


def _write(path: Path, documents: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(document) + "\n" for document in documents), encoding="utf-8"
    )


def _prediction(
    center: tuple[float, float],
    *,
    source: str = "detected",
    segment: int = 0,
    gap_frames: int = 0,
) -> BallDetection:
    x, y = center
    return BallDetection(
        box=Box(x - 5.0, y - 5.0, x + 5.0, y + 5.0),
        confidence=0.0 if source == "interpolated" else 0.9,
        source=source,
        trajectory_segment=segment,
        gap_frames=gap_frames,
    )


def test_reader_accepts_versioned_jsonl_and_preserves_private_identity(tmp_path: Path) -> None:
    path = tmp_path / "ball.jsonl"
    _write(path, [_document()])

    (label,) = read_ball_trajectory_labels(path)

    assert label.clip_id == "opaque-clip"
    assert label.match_id == "opaque-match"
    assert label.center_xy == (100.0, 200.0)
    assert label.apparent_diameter_px == 12.0


def test_visible_label_requires_center_and_diameter(tmp_path: Path) -> None:
    path = tmp_path / "ball.jsonl"
    _write(path, [_document(center=None)])

    with pytest.raises(BallTrajectoryLabelError, match="visible"):
        read_ball_trajectory_labels(path)


def test_reader_rejects_duplicate_frame_in_same_clip(tmp_path: Path) -> None:
    path = tmp_path / "ball.jsonl"
    _write(path, [_document(), _document()])

    with pytest.raises(BallTrajectoryLabelError, match="duas vezes"):
        read_ball_trajectory_labels(path)


def test_metrics_keep_detected_interpolated_and_missing_separate(tmp_path: Path) -> None:
    path = tmp_path / "ball.jsonl"
    documents = [
        _document(frame_index=0, timestamp_s=0.0, center=(100.0, 200.0)),
        _document(frame_index=1, timestamp_s=1 / 30, center=(110.0, 200.0)),
        _document(frame_index=2, timestamp_s=2 / 30, center=(120.0, 200.0)),
    ]
    _write(path, documents)
    labels = read_ball_trajectory_labels(path)
    predictions = {
        ("opaque-clip", 0): _prediction((100.0, 200.0)),
        ("opaque-clip", 1): _prediction((111.0, 200.0), source="interpolated", gap_frames=1),
    }

    report = measure(labels, predictions)

    assert report.n_frames == 3
    assert report.n_matches == 1
    assert report.detected_points == 1
    assert report.interpolated_points == 1
    assert report.missing_points == 1
    assert report.true_positive == 2
    assert report.false_negative == 1
    assert report.precision == 1.0
    assert report.recall == pytest.approx(2 / 3)
    assert report.center_error_median_px == pytest.approx(0.5)
    assert report.max_visible_gap_frames == 1


def test_wrong_visible_position_counts_as_false_positive_and_false_negative(tmp_path: Path) -> None:
    path = tmp_path / "ball.jsonl"
    _write(path, [_document()])
    labels = read_ball_trajectory_labels(path)

    report = measure(labels, {("opaque-clip", 12): _prediction((200.0, 300.0))})

    assert report.true_positive == 0
    assert report.false_positive == 1
    assert report.false_negative == 1
    assert report.f1 == 0.0


def test_prediction_when_ball_is_not_visible_is_a_false_positive(tmp_path: Path) -> None:
    path = tmp_path / "ball.jsonl"
    _write(path, [_document(visibility="not-visible", center=None, segment=None)])
    labels = read_ball_trajectory_labels(path)

    report = measure(labels, {("opaque-clip", 12): _prediction((100.0, 200.0))})

    assert report.n_visible == 0
    assert report.false_positive == 1
    assert report.false_negative == 0


def test_segment_change_inside_same_annotated_track_counts_as_switch(tmp_path: Path) -> None:
    path = tmp_path / "ball.jsonl"
    _write(
        path,
        [
            _document(frame_index=0, timestamp_s=0.0),
            _document(frame_index=1, timestamp_s=1 / 30),
        ],
    )
    labels = read_ball_trajectory_labels(path)
    predictions = {
        ("opaque-clip", 0): _prediction((100.0, 200.0), segment=0),
        ("opaque-clip", 1): _prediction((100.0, 200.0), segment=1),
    }

    report = measure(labels, predictions)

    assert report.predicted_segments == 2
    assert report.track_switches == 1


def test_match_cannot_appear_in_more_than_one_fold() -> None:
    with pytest.raises(BallTrajectoryLabelError, match="partida"):
        validate_match_folds([("match-a", 0), ("match-a", 3)])

    validate_match_folds([("match-a", 0), ("match-a", 0), ("match-b", 3)])


def test_telemetry_sums_counters_but_keeps_largest_gap() -> None:
    report = aggregate_telemetry(
        [
            {"frames_requested": 10, "max_gap_frames": 2, "detector_ms": 30.0},
            {"frames_requested": 8, "max_gap_frames": 5, "detector_ms": 20.0},
        ]
    )

    assert report == {"frames_requested": 18, "max_gap_frames": 5, "detector_ms": 50.0}


def _labels_path(tmp_path: Path, second_center: tuple[float, float] = (110.0, 200.0)) -> Path:
    path = tmp_path / f"ball-{second_center[0]:g}.jsonl"
    _write(
        path,
        [
            _document(frame_index=0, timestamp_s=0.0, center=(100.0, 200.0)),
            _document(frame_index=1, timestamp_s=1 / 30, center=second_center),
            _document(frame_index=2, timestamp_s=2 / 30, center=(120.0, 200.0)),
            _document(
                frame_index=3,
                timestamp_s=3 / 30,
                visibility="not-visible",
                center=None,
                segment=None,
            ),
        ],
    )
    return path


def test_frame_outcomes_use_the_pck_tolerance_and_the_evaluator_key(tmp_path: Path) -> None:
    """A soma dos acertos por quadro tem de ser o `true_positive`, senao o par mede outra regua."""
    labels = read_ball_trajectory_labels(_labels_path(tmp_path))
    predictions = {
        ("opaque-clip", 0): _prediction((104.0, 200.0)),
        ("opaque-clip", 1): _prediction((140.0, 200.0)),
        ("opaque-clip", 3): _prediction((50.0, 50.0)),
    }

    outcomes = frame_outcomes(labels, predictions)
    report = measure(labels, predictions)

    assert outcomes == {
        ("opaque-clip", 0): "hit",
        ("opaque-clip", 1): "miss",
        ("opaque-clip", 2): "absent",
    }
    assert sum(outcome == "hit" for outcome in outcomes.values()) == report.true_positive
    assert len(outcomes) == report.n_visible


def test_paired_comparison_counts_gains_losses_and_unchanged() -> None:
    """Ausente e errado contam os dois como nao acerto: trocar um pelo outro nao e ganho."""
    base = {
        ("c", 0): "miss",
        ("c", 1): "absent",
        ("c", 2): "hit",
        ("c", 3): "hit",
        ("c", 4): "miss",
    }
    new = {
        ("c", 0): "hit",
        ("c", 1): "hit",
        ("c", 2): "absent",
        ("c", 3): "hit",
        ("c", 4): "absent",
    }

    paired = compare_outcomes(base, new)

    assert (paired.gains, paired.losses, paired.unchanged) == (2, 1, 2)
    assert (paired.base_hits, paired.new_hits, paired.n_frames) == (2, 3, 5)


def test_sign_test_p_value_matches_the_exact_binomial_on_known_cases() -> None:
    def paired(gains: int, losses: int) -> float:
        keys = [("c", i) for i in range(gains + losses)]
        base = {key: "miss" if i < gains else "hit" for i, key in enumerate(keys)}
        new = {key: "hit" if i < gains else "miss" for i, key in enumerate(keys)}
        return compare_outcomes(base, new).p_value

    assert paired(6, 0) == pytest.approx(2 / 64)
    assert paired(5, 0) == pytest.approx(2 / 32)
    assert paired(8, 2) == pytest.approx(2 * (1 + 10 + 45) / 1024)
    assert paired(0, 0) == 1.0


def test_paired_comparison_refuses_runs_over_different_frames() -> None:
    with pytest.raises(BallTrajectoryLabelError, match="mesmos quadros"):
        compare_outcomes({("c", 0): "hit"}, {("c", 1): "hit"})


def test_label_digest_ignores_order_and_catches_a_corrected_center(tmp_path: Path) -> None:
    labels = read_ball_trajectory_labels(_labels_path(tmp_path))
    corrected = read_ball_trajectory_labels(_labels_path(tmp_path, second_center=(111.0, 200.0)))

    assert label_set_sha256(labels) == label_set_sha256(tuple(reversed(labels)))
    assert label_set_sha256(labels) != label_set_sha256(corrected)


def test_frame_outcomes_survive_the_json_round_trip(tmp_path: Path) -> None:
    outcomes = {("c", 12): "hit", ("c", 3): "absent", ("d", 0): "miss"}
    path = tmp_path / "run.json"
    path.write_text(
        json.dumps({"label_set_sha256": "f" * 64, "frame_outcomes": outcomes_document(outcomes)}),
        encoding="utf-8",
    )

    assert read_frame_outcomes(path) == ("f" * 64, outcomes)


def test_a_run_saved_before_per_frame_outcomes_is_refused_as_base(tmp_path: Path) -> None:
    """Base so com o agregado nao tem o que parear, e comparar so o PCK seria o numero de antes."""
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"pck_diameter": 0.3}), encoding="utf-8")

    with pytest.raises(BallTrajectoryLabelError, match="desfecho por quadro"):
        read_frame_outcomes(path)
    with pytest.raises(BallTrajectoryLabelError, match="nao existe"):
        read_frame_outcomes(tmp_path / "missing.json")


def test_an_unknown_outcome_in_the_base_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    path.write_text(
        json.dumps({"label_set_sha256": "f" * 64, "frame_outcomes": {"c": {"1": "maybe"}}}),
        encoding="utf-8",
    )

    with pytest.raises(BallTrajectoryLabelError, match="invalido"):
        read_frame_outcomes(path)


@pytest.mark.parametrize(
    "document",
    [
        [],
        None,
        {"label_set_sha256": "f" * 64, "frame_outcomes": {"c": []}},
        {"label_set_sha256": "f" * 64, "frame_outcomes": {"c": {"1": []}}},
        {"label_set_sha256": "f" * 64, "frame_outcomes": {"c": {"²": "hit"}}},
        {"label_set_sha256": "f" * 64, "frame_outcomes": {"c": {"01": "hit"}}},
        {"label_set_sha256": "f" * 64, "frame_outcomes": {"": {"1": "hit"}}},
        {"label_set_sha256": "not-a-digest", "frame_outcomes": {"c": {"1": "hit"}}},
    ],
)
def test_malformed_base_is_a_label_error(tmp_path: Path, document: object) -> None:
    path = tmp_path / "base.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(BallTrajectoryLabelError):
        read_frame_outcomes(path)
