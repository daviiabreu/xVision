"""Retomada sobre decodificacao e pipeline reais, com o catalogo sintetico isolado."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import cli
from contracts.validation import validate_shot_event
from edge.pipeline import ClipResult
from edge.testing.make_clip import make_synthetic_clip
from mlops.catalog import ShotRecord
from mlops.checkpoint import CHECKPOINT_FILE, read_checkpoint
from mlops.instant import OBSERVATIONS_FILE


def test_resume_preserves_real_pipeline_events_and_stage_totals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = make_synthetic_clip(tmp_path / "fixture.mp4")
    labels = {
        "goalkeaper_position": "false",
        "occupied_area": "false",
        "marked": "false",
        "looked_before": "false",
        "assist": "false",
        "body_part": "PERNA_DIREITA",
        "type_of_play": "open play",
    }
    catalog = tuple(ShotRecord(str(i), "900", clip, True, 0, labels) for i in range(12))
    monkeypatch.setattr(cli, "load_catalog", lambda: catalog)
    process = cli._processa_registro  # pyright: ignore[reportPrivateUsage]
    seen: list[str] = []

    def interrupt(record: ShotRecord, config: Path, *, cache_enabled: bool = True) -> ClipResult:
        if len(seen) == 11:
            raise KeyboardInterrupt
        seen.append(record.shot_id)
        return process(record, config, cache_enabled=cache_enabled)

    monkeypatch.setattr(cli, "_processa_registro", interrupt)
    output = tmp_path / "runs"
    args = ["run", "--all", "--dobra", "ajuste", "--out", str(output)]
    first = CliRunner().invoke(cli.app, args)
    assert first.exit_code != 0
    (run,) = output.iterdir()
    assert read_checkpoint(run).ok == 10
    assert not (run / "events.jsonl").exists()

    resumed: list[str] = []

    def continue_processing(
        record: ShotRecord, config: Path, *, cache_enabled: bool = True
    ) -> ClipResult:
        resumed.append(record.shot_id)
        return process(record, config, cache_enabled=cache_enabled)

    monkeypatch.setattr(cli, "_processa_registro", continue_processing)
    result = CliRunner().invoke(cli.app, [*args, "--resume", str(run)])
    assert result.exit_code == 0, result.output
    assert resumed == ["10", "11"]
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    assert [event["clip_id"] for event in events] == [str(i) for i in range(12)]
    for event in events:
        validate_shot_event(event)
    manifest = json.loads((run / "run.json").read_text())
    assert manifest["dataset"]["clips"] == 12
    assert manifest["dataset"]["failed"] == 0
    assert all(stage["duration_ms"]["n"] == 12 for stage in manifest["modules"].values())
    assert len((run / OBSERVATIONS_FILE).read_text().splitlines()) == 12
    assert not (run / CHECKPOINT_FILE).exists()
