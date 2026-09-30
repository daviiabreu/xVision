"""The CLI can measure inference without touching the shared perception cache."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import pytest
from typer.testing import CliRunner

from cli import app
from edge.testing.make_clip import make_synthetic_clip
from mlops.catalog import SHOTS, XG

runner = CliRunner()


@pytest.fixture
def catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    root = tmp_path / "catalog"
    db = root / "db"
    clips = root / "cortes"
    db.mkdir(parents=True)
    clips.mkdir()
    fixture = make_synthetic_clip(tmp_path / "fixture.mp4")
    paths = [clips / f"synthetic_fixture_case_control_{index}.mp4" for index in range(12)]
    for path in paths:
        shutil.copyfile(fixture, path)
    with (db / SHOTS).open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["id", "id_match", "body_part"])
        writer.writerows([[str(index), "900", "PERNA_DIREITA"] for index in range(12)])
    with (db / XG).open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "id_match_shot",
                "goalkeaper_position",
                "occupied_area",
                "marked",
                "looked_before",
                "assist",
                "type_of_play",
            ]
        )
        writer.writerows([[str(index), *(["false"] * 5), "open play"] for index in range(12)])
    monkeypatch.setenv("XVISION_DATA_DIR", str(root))
    return paths


def _event(output: Path, clip: Path) -> dict[str, Any]:
    event = json.loads((output / clip.stem / "events.jsonl").read_text())
    del event["pipeline"]["timings_ms"]
    del event["pipeline"]["started_at"]
    return event


def test_uncached_clip_preserves_existing_cache_and_emits_the_same_event(tmp_path: Path) -> None:
    clip = make_synthetic_clip(tmp_path / "fixture.mp4")
    cold_output = tmp_path / "cold"
    first = runner.invoke(app, ["run", "--clip", str(clip), "--out", str(cold_output)])
    assert first.exit_code == 0, first.output
    entries = list((tmp_path / "var" / "cache" / "perception").glob("*.json"))
    assert entries
    for entry in entries:
        entry.write_text("{}\n")
    before = {entry.name: entry.read_bytes() for entry in entries}

    output = tmp_path / "uncached"
    result = runner.invoke(app, ["run", "--clip", str(clip), "--out", str(output), "--no-cache"])

    assert result.exit_code == 0, result.output
    assert _event(output, clip) == _event(cold_output, clip)
    assert {entry.name: entry.read_bytes() for entry in entries} == before
    assert set(entries[0].parent.glob("*.json")) == set(entries)


@pytest.mark.parametrize("cache_enabled", [True, False])
def test_catalog_records_cache_policy_and_only_writes_cache_when_enabled(
    tmp_path: Path, catalog: list[Path], cache_enabled: bool
) -> None:
    output = tmp_path / "runs"
    args = ["run", "--all", "--limit", "2", "--out", str(output)]
    if not cache_enabled:
        args.append("--no-cache")

    result = runner.invoke(app, args)

    assert result.exit_code == 0, result.output
    (manifest,) = output.rglob("run.json")
    document = json.loads(manifest.read_text())
    assert document["cache_enabled"] is cache_enabled
    assert document["dataset"]["clips"] == 2
    entries = list((tmp_path / "var" / "cache" / "perception").glob("*.json"))
    assert bool(entries) is cache_enabled


def _interrupt_catalog(
    tmp_path: Path, catalog: list[Path], cache_enabled: bool
) -> tuple[list[str], Path]:
    args = ["run", "--all", "--out", str(tmp_path / "runs")]
    if not cache_enabled:
        args.append("--no-cache")
    capture = cv2.VideoCapture

    def interrupted_capture(path: str, *args: Any, **kwargs: Any) -> Any:
        if Path(path) == catalog[-1]:
            raise KeyboardInterrupt
        return capture(path, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cv2, "VideoCapture", interrupted_capture)
        result = runner.invoke(app, args)
    assert result.exit_code != 0
    (run,) = (tmp_path / "runs").iterdir()
    return args, run


@pytest.mark.parametrize("cache_enabled", [True, False])
def test_resume_rejects_changed_policy_without_modifying_files_then_completes(
    tmp_path: Path, catalog: list[Path], cache_enabled: bool
) -> None:
    args, run = _interrupt_catalog(tmp_path, catalog, cache_enabled)
    progress = run / "progress.json"
    partial = run / "events.jsonl.parcial"
    document = json.loads(progress.read_text())
    assert document["ok"] == 10
    assert document["cache_enabled"] is cache_enabled
    before = progress.read_bytes(), partial.read_bytes()
    changed = [arg for arg in args if arg != "--no-cache"]
    if cache_enabled:
        changed.append("--no-cache")

    rejected = runner.invoke(app, [*changed, "--resume", str(run)])

    assert rejected.exit_code == 1, rejected.output
    assert "politica de cache" in rejected.output
    assert (progress.read_bytes(), partial.read_bytes()) == before
    assert not (run / "run.json").exists()

    resumed = runner.invoke(app, [*args, "--resume", str(run)])
    assert resumed.exit_code == 0, resumed.output
    manifest = json.loads((run / "run.json").read_text())
    assert manifest["cache_enabled"] is cache_enabled
    assert manifest["dataset"]["clips"] == 12
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    assert [event["clip_id"] for event in events] == [str(index) for index in range(12)]
    assert not progress.exists()
