from __future__ import annotations

import json
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import cli


def test_resume_refuses_an_active_run_without_truncating_its_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [SimpleNamespace(shot_id=str(index), fold=0) for index in range(25)]
    context = multiprocessing.get_context("fork")
    parent, child = context.Pipe()

    def labels(_: Path) -> None:
        return None

    def population(items: list[SimpleNamespace], _: object) -> list[SimpleNamespace]:
        return items

    monkeypatch.setattr(cli, "read_labels", labels)
    monkeypatch.setattr(cli, "load_catalog", lambda: records)
    monkeypatch.setattr(cli, "_populacao", population)
    monkeypatch.setattr(cli, "new_run_id", lambda: "concurrent-fixture")

    def process(record: SimpleNamespace, config: Path) -> SimpleNamespace:
        if record.shot_id == "10":
            child.send("checkpoint-written")
            child.recv()
        return SimpleNamespace(
            event=SimpleNamespace(to_dict=lambda: {"clip_id": record.shot_id}),
            stages=(),
            shot_source="ball-break",
            ball_support="detected",
        )

    monkeypatch.setattr(cli, "_processa_registro", process)
    arguments = ["run", "--all", "--config", "config/pipeline.toml", "--out", str(tmp_path)]

    def original_run() -> None:
        try:
            result = CliRunner().invoke(cli.app, arguments)
            child.send(result.exit_code)
        finally:
            child.close()

    original = context.Process(target=original_run)
    original.start()
    child.close()
    directory = tmp_path / "concurrent-fixture"
    try:
        assert parent.poll(10), "a execucao nao chegou ao primeiro checkpoint"
        assert parent.recv() == "checkpoint-written"
        partial = directory / "events.jsonl.parcial"
        progress = directory / "progress.json"
        saved_events, saved_progress = partial.read_bytes(), progress.read_bytes()

        competing = CliRunner().invoke(cli.app, [*arguments, "--resume", str(directory)])

        assert competing.exit_code == 1
        assert "em uso" in competing.output
        assert partial.read_bytes() == saved_events
        assert progress.read_bytes() == saved_progress
        parent.send("continue")
        assert parent.poll(10), "a execucao original nao terminou"
        assert parent.recv() == 0
        original.join(10)
        assert not original.is_alive()
        assert original.exitcode == 0

        events = [
            json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()
        ]
        manifest = json.loads((directory / "run.json").read_text())
        provenance = (directory / "instant-observations.jsonl").read_text().splitlines()
        assert [event["clip_id"] for event in events] == [str(index) for index in range(25)]
        assert manifest["dataset"]["clips"] == len(events) == len(provenance) == 25
        assert not progress.exists()
    finally:
        if original.is_alive():
            original.kill()
        original.join(10)
        parent.close()
        original.close()
