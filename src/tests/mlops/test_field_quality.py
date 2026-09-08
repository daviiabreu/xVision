"""Avaliacao independente dos keypoints usados para ajustar a homografia."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from contracts import Clip, FieldGeometry
from mlops.field_quality import FieldLabelError, measure, read_field_labels, validate_label_video


def _document() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "clip_id": "opaque-clip",
        "match_id": "opaque-match",
        "video_sha256": "a" * 64,
        "frame_index": 12,
        "frame_width": 1920,
        "frame_height": 1080,
        "annotator": "reviewer-a",
        "reviewed_by": None,
        "created_at": "2026-09-03T12:00:00Z",
        "quality_flags": ["human-annotated"],
        "visible_pitch_elements": ["touchline-top"],
        "line_points": [
            {
                "element": "touchline-top",
                "pitch_xy": [[-20.0, -34.0], [20.0, -34.0]],
                "image_xy": [[760.0, 880.0], [1160.0, 880.0]],
            }
        ],
        "field_dimensions_m": [105.0, 68.0],
    }


def _geometry() -> FieldGeometry:
    return FieldGeometry(
        homography=(10.0, 0.0, 960.0, 0.0, -10.0, 540.0, 0.0, 0.0, 1.0),
        reprojection_error_px=0.2,
        reprojection_error_p95_px=0.4,
        keypoints=(),
        observed_keypoint_count=8,
        inlier_count=8,
        source="detected",
        age_frames=0,
        template_id="pitch-105x68-center-v1",
        reason=None,
    )


def test_exact_geometry_has_complete_independent_line_score(tmp_path: Path) -> None:
    path = tmp_path / "field.jsonl"
    path.write_text(json.dumps(_document()) + "\n", encoding="utf-8")
    labels = read_field_labels(path)

    report = measure(labels, {("opaque-clip", 12): _geometry()})

    assert report.n_frames == 1
    assert report.n_matches == 1
    assert report.completion == 1.0
    assert report.jac_at_5 == 1.0
    assert report.line_error_p95_px == pytest.approx(0.0)
    assert report.score == 1.0


def test_unavailable_geometry_counts_against_completion(tmp_path: Path) -> None:
    path = tmp_path / "field.jsonl"
    path.write_text(json.dumps(_document()) + "\n", encoding="utf-8")

    report = measure(read_field_labels(path), {})

    assert report.n_unavailable == 1
    assert report.completion == 0.0
    assert report.jac_at_5 == 0.0
    assert report.score == 0.0


def test_schema_rejects_mismatched_image_and_pitch_polylines(tmp_path: Path) -> None:
    document = _document()
    line = document["line_points"][0]  # type: ignore[index]
    line["pitch_xy"] = [[-20.0, -34.0], [0.0, -34.0], [20.0, -34.0]]  # type: ignore[index]
    path = tmp_path / "field.jsonl"
    path.write_text(json.dumps(document) + "\n", encoding="utf-8")

    with pytest.raises(FieldLabelError, match="mesma quantidade"):
        read_field_labels(path)


@pytest.mark.parametrize("coordinates", [[], [1.0]])
@pytest.mark.parametrize("field", ["image_xy", "pitch_xy"])
def test_schema_rejects_incomplete_coordinate_pairs(
    tmp_path: Path, field: str, coordinates: list[float]
) -> None:
    document = _document()
    document["line_points"][0][field] = [coordinates, coordinates]  # type: ignore[index]
    path = tmp_path / "field.jsonl"
    path.write_text(json.dumps(document) + "\n", encoding="utf-8")

    with pytest.raises(FieldLabelError, match="too short"):
        read_field_labels(path)


def test_proposals_are_excluded_and_exploration_has_explicit_denominators(tmp_path: Path) -> None:
    human = _document()
    proposal = {
        **human,
        "frame_index": 13,
        "annotator": "model-generated",
        "quality_flags": ["requires-human-review"],
    }
    path = tmp_path / "field.jsonl"
    path.write_text("\n".join(map(json.dumps, (human, proposal))), encoding="utf-8")
    labels = read_field_labels(path)
    geometries = {("opaque-clip", 13): _geometry()}

    confirmed = measure(labels, geometries)
    exploratory = measure(labels, geometries, include_proposals=True)

    assert confirmed.n_labels == 2
    assert confirmed.n_frames == 1
    assert confirmed.n_proposal_labels == confirmed.n_excluded_proposals == 1
    assert confirmed.score == 0.0
    assert confirmed.to_dict()["evaluation_kind"] == "human-confirmed"
    assert exploratory.n_frames == 2
    assert exploratory.n_excluded_proposals == 0
    assert exploratory.score == 0.5
    assert exploratory.to_dict()["evaluation_kind"] == "exploratory-with-proposals"


def test_assisted_confirmation_remains_distinguishable_from_blind_review(tmp_path: Path) -> None:
    document = _document()
    document["quality_flags"] = ["human-annotated", "white-paint-alignment-reviewed"]
    path = tmp_path / "field.jsonl"
    path.write_text(json.dumps(document) + "\n", encoding="utf-8")
    labels = read_field_labels(path)

    report = measure(labels, {("opaque-clip", 12): _geometry()})

    assert report.n_frames == report.n_assisted_frames == 1
    assert labels[0].annotator == "reviewer-a"
    assert labels[0].reviewed_by is None
    assert labels[0].video_sha256 == "a" * 64


@pytest.mark.parametrize(
    ("changes", "digest", "message"),
    [
        ({}, "b" * 64, "video_sha256"),
        ({"width": 960}, "a" * 64, "dimensoes"),
        ({"frame_count": 12}, "a" * 64, "frame_index"),
        ({"match_id": "another-match"}, "a" * 64, "partida"),
    ],
)
def test_video_provenance_must_match_before_evaluation(
    tmp_path: Path, changes: dict[str, Any], digest: str, message: str
) -> None:
    path = tmp_path / "field.jsonl"
    path.write_text(json.dumps(_document()) + "\n", encoding="utf-8")
    label = read_field_labels(path)[0]
    clip = Clip("opaque-clip", "synthetic.mp4", 30.0, 30, 1920, 1080, "opaque-match")
    validate_label_video(label, clip, "a" * 64)

    with pytest.raises(FieldLabelError, match=message):
        validate_label_video(label, replace(clip, **changes), digest)
