"""O comando de campo valida o video e declara a procedencia da populacao medida."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import cli
from edge.artifacts import sha256_of
from edge.modules.stubs import StubField
from edge.testing.make_clip import make_synthetic_clip
from mlops.catalog import ShotRecord


@pytest.fixture
def field_population(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    video = make_synthetic_clip(tmp_path / "synthetic.mp4")
    record = ShotRecord("synthetic", "synthetic-match", video, True, 0, {})
    monkeypatch.setattr(cli, "load_catalog", lambda: (record,))

    def modules(_config: object) -> dict[str, StubField]:
        return {"field": StubField()}

    monkeypatch.setattr(cli, "load_modules", modules)
    common = {
        "schema_version": "1.0",
        "clip_id": "synthetic",
        "match_id": "synthetic-match",
        "video_sha256": sha256_of(video),
        "frame_width": 320,
        "frame_height": 240,
        "annotator": "reviewer",
        "reviewed_by": None,
        "created_at": "2026-09-03T12:00:00Z",
        "visible_pitch_elements": ["touchline-top"],
        "line_points": [
            {
                "element": "touchline-top",
                "pitch_xy": [[-20.0, -34.0], [20.0, -34.0]],
                "image_xy": [[60.0, 80.0], [160.0, 80.0]],
            }
        ],
        "field_dimensions_m": [105.0, 68.0],
    }
    return [
        {**common, "frame_index": 0, "quality_flags": ["human-annotated"]},
        {**common, "frame_index": 1, "quality_flags": ["requires-human-review"]},
    ]


@pytest.mark.parametrize("include_proposals", [False, True])
def test_eval_field_reports_the_measured_denominator(
    tmp_path: Path, field_population: list[dict[str, Any]], include_proposals: bool
) -> None:
    labels = tmp_path / "labels.jsonl"
    labels.write_text("\n".join(map(json.dumps, field_population)), encoding="utf-8")
    output = tmp_path / "report.json"
    args = ["eval-field", "--labels", str(labels), "--saida", str(output)]
    if include_proposals:
        args.append("--include-proposals")

    result = CliRunner().invoke(cli.app, args)

    assert result.exit_code == 0, result.output
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["n_labels"] == 2
    assert report["n_frames"] == (2 if include_proposals else 1)
    assert report["n_excluded_proposals"] == (0 if include_proposals else 1)
    assert report["evaluation_kind"] == (
        "exploratory-with-proposals" if include_proposals else "human-confirmed"
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [("video_sha256", "b" * 64, "video_sha256"), ("frame_width", 640, "dimensoes")],
)
def test_eval_field_rejects_video_mismatch_before_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field_population: list[dict[str, Any]],
    field: str,
    value: object,
    message: str,
) -> None:
    field_population[0][field] = value
    labels = tmp_path / "labels.jsonl"
    labels.write_text("\n".join(map(json.dumps, field_population)), encoding="utf-8")

    def refuse_inference(*_args: object) -> None:
        pytest.fail("infer nao pode receber um video diferente da anotacao")

    monkeypatch.setattr(StubField, "infer", refuse_inference)

    result = CliRunner().invoke(cli.app, ["eval-field", "--labels", str(labels)])

    assert result.exit_code == 1
    assert message in result.output
