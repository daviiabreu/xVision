from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cli import app


def _feature(value: bool) -> dict[str, object]:
    return {"name": "marked", "status": "ok", "value": value, "reason": None}


@pytest.mark.parametrize("existing_report", [False, True])
@pytest.mark.parametrize("duplicate", ["clip", "feature"])
def test_eval_refuses_ambiguous_predictions_without_replacing_the_report(
    tmp_path: Path, duplicate: str, existing_report: bool
) -> None:
    labels = tmp_path / "labels.toml"
    labels.write_text(
        '[[label]]\nname="marked"\ntable="t"\ncolumn="marked"\ntype="bool"\n'
        '[[metric]]\nfeature="marked"\nlabel="marked"\n'
    )
    truth = tmp_path / "truth.json"
    truth.write_text('{"c1":{"marked":"true"}}')
    if duplicate == "clip":
        events = [{"clip_id": "c1", "features": [_feature(v)]} for v in (False, True)]
        line = 2
    else:
        events = [{"clip_id": "c1", "features": [_feature(False), _feature(True)]}]
        line = 1
    source = tmp_path / "events.jsonl"
    source.write_text("".join(json.dumps(e) + "\n" for e in events))
    report = tmp_path / "report.json"
    previous = b'{"original":true}\n'
    if existing_report:
        report.write_bytes(previous)

    result = CliRunner().invoke(
        app, ["eval", str(tmp_path), "--labels", str(labels), "--truth", str(truth)]
    )

    assert result.exit_code == 1, result.output
    assert f"events.jsonl:{line}" in result.output
    assert "duplicad" in result.output
    assert "Traceback" not in result.output
    assert report.read_bytes() == previous if existing_report else not report.exists()


@pytest.mark.parametrize(
    "event",
    [
        [],
        {},
        {"clip_id": 1, "features": []},
        {"clip_id": "c1", "features": {}},
        {"clip_id": "c1", "features": [None]},
        {"clip_id": "c1", "features": [{"name": "", "reason": None}]},
        {"clip_id": "c1", "features": [{"name": "marked"}]},
        {"clip_id": "c1", "features": [{"name": "marked", "reason": 4}]},
    ],
)
def test_eval_reports_invalid_event_structure_and_preserves_previous_report(
    tmp_path: Path, event: object
) -> None:
    source = tmp_path / "events.jsonl"
    source.write_text("\n" + json.dumps(event) + "\n")
    truth = tmp_path / "truth.json"
    truth.write_text("{}")
    previous = b'{"original":true}\n'
    report = tmp_path / "report.json"
    report.write_bytes(previous)

    result = CliRunner().invoke(app, ["eval", str(tmp_path), "--truth", str(truth)])

    assert result.exit_code == 1, result.output
    assert "events.jsonl:2" in result.output
    assert "Traceback" not in result.output
    assert report.read_bytes() == previous


def test_eval_accepts_distinct_clips_blank_lines_and_unavailable_features(tmp_path: Path) -> None:
    source = tmp_path / "events.jsonl"
    unavailable = {"name": "marked", "status": "unavailable", "value": None, "reason": "ausente"}
    source.write_text(
        "\n"
        + json.dumps({"clip_id": "c1", "features": [_feature(True)]})
        + "\n\n"
        + json.dumps({"clip_id": "c2", "features": [unavailable]})
        + "\n"
    )
    truth = tmp_path / "truth.json"
    truth.write_text('{"c1":{"marked":"true"},"c2":{"marked":"false"}}')

    result = CliRunner().invoke(app, ["eval", str(tmp_path), "--truth", str(truth)])

    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path / "report.json").read_text())
    metric = next(m for m in report["metrics"] if m["feature"] == "marked")
    assert (metric["n_predicted"], metric["n_unavailable"], metric["coverage"]) == (1, 1, 0.5)
    assert metric["scores"]["accuracy"] == 1.0


@pytest.mark.parametrize("existing_report", [False, True])
def test_eval_refuses_unrepresentable_threshold_without_publishing_infinity(
    tmp_path: Path, existing_report: bool
) -> None:
    labels = tmp_path / "labels.toml"
    labels.write_text(
        '[[label]]\nname="marked"\ntable="t"\ncolumn="marked"\ntype="bool"\n'
        '[[metric]]\nfeature="marked"\nlabel="marked"\n'
    )
    truth = tmp_path / "truth.json"
    truth.write_text('{"c1":{"marked":"false"}}')
    feature = {**_feature(False), "evidence": {"score": sys.float_info.max}}
    (tmp_path / "events.jsonl").write_text(
        json.dumps({"clip_id": "c1", "features": [feature]}, allow_nan=False) + "\n"
    )
    report = tmp_path / "report.json"
    original = b'{"original":true}\n'
    if existing_report:
        report.write_bytes(original)

    result = CliRunner().invoke(
        app, ["eval", str(tmp_path), "--labels", str(labels), "--truth", str(truth)]
    )

    assert result.exit_code == 1, result.output
    assert "limiar finito" in result.output
    assert report.read_bytes() == original if existing_report else not report.exists()
