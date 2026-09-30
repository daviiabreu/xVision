from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import cli
from edge.pipeline import ClipResult
from edge.testing.make_clip import make_synthetic_clip
from mlops.catalog import ShotRecord

from .test_run_resume import (
    _executa,  # pyright: ignore[reportPrivateUsage]
    _Interrompe,  # pyright: ignore[reportPrivateUsage]
    _prepara,  # pyright: ignore[reportPrivateUsage]
    _run_id,  # pyright: ignore[reportPrivateUsage]
)


def _interrupted_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, list[str]]:
    clip = make_synthetic_clip(tmp_path / "clip.mp4")
    records = [ShotRecord(str(i), "synthetic", clip, True, 0, {}) for i in range(12)]

    def selected(_limit: int, _gabarito: bool, _dobra: str | None) -> list[ShotRecord]:
        return records

    monkeypatch.setattr(cli, "_escolhidos", selected)
    spots = tmp_path / "spots.jsonl"
    spots.write_text(
        "".join(
            json.dumps({"clip_id": r.shot_id, "frame_index": 20, "confidence": 0.9}) + "\n"
            for r in records
        )
    )
    config = tmp_path / "pipeline.toml"
    config.write_text(
        '[frames.coarse]\nstrategy="uniform"\nfps=5\n'
        '[frames.dense]\nstrategy="uniform"\nfps=5\n'
        '[modules.ball]\nimpl="edge.modules.stubs:StubBall"\nframes="dense"\n'
        '[modules.shot]\nimpl="mlops.spots:PrecomputedShot"\nspots=' + json.dumps(str(spots)) + "\n"
    )
    process = cli._processa_registro  # pyright: ignore[reportPrivateUsage]
    processed: list[str] = []

    def interrupted(record: ShotRecord, config: Path, *, cache_enabled: bool = True) -> ClipResult:
        if len(processed) == 11:
            raise KeyboardInterrupt
        processed.append(record.shot_id)
        return process(record, config, cache_enabled=cache_enabled)

    args = ["run", "--all", "--no-cache", "--config", str(config), "--out", str(tmp_path / "runs")]
    with monkeypatch.context() as patch:
        patch.setattr(cli, "_processa_registro", interrupted)
        first = CliRunner().invoke(cli.app, args)
    assert first.exit_code == 130
    (run,) = (tmp_path / "runs").iterdir()
    assert json.loads((run / "progress.json").read_text())["ok"] == 10
    return run, spots, args


@pytest.mark.parametrize("change", ["content", "missing", "legacy"])
def test_resume_refuses_changed_or_unverifiable_fingerprints_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    run, spots, args = _interrupted_run(tmp_path, monkeypatch)
    checkpoint = run / "progress.json"
    if change == "content":
        spots.write_text(spots.read_text().replace('"frame_index": 20', '"frame_index": 10'))
    elif change == "missing":
        spots.unlink()
    else:
        document = json.loads(checkpoint.read_text())
        document.pop("module_tunings", None)
        checkpoint.write_text(json.dumps(document))
    partial = run / "events.jsonl.parcial"
    before = partial.read_bytes(), checkpoint.read_bytes()

    result = CliRunner().invoke(cli.app, [*args, "--resume", str(run)])

    assert result.exit_code == 1, result.output
    assert "fingerprints" in result.output
    assert (partial.read_bytes(), checkpoint.read_bytes()) == before
    assert not (run / "events.jsonl").exists()


def test_resume_with_identical_spots_preserves_events_and_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, _, args = _interrupted_run(tmp_path, monkeypatch)
    original = json.loads((run / "progress.json").read_text())

    result = CliRunner().invoke(cli.app, [*args, "--resume", str(run)])

    assert result.exit_code == 0, result.output
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    assert [event["shot_id"] for event in events] == [f"{i}:20" for i in range(12)]
    manifest = json.loads((run / "run.json").read_text())
    assert manifest["modules"]["shot"]["tuning"] == original["module_tunings"]["shot"]
    assert not (run / "progress.json").exists()


def test_invalid_module_preflight_reports_usage_error_without_modifying_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, _, args = _interrupted_run(tmp_path, monkeypatch)
    config = Path(args[args.index("--config") + 1])
    config.write_text(config.read_text().replace("edge.modules.stubs:StubBall", "absent:Module"))
    checkpoint = run / "progress.json"
    partial = run / "events.jsonl.parcial"
    before = checkpoint.read_bytes(), partial.read_bytes()

    result = CliRunner().invoke(cli.app, [*args, "--resume", str(run)])

    assert result.exit_code == 1
    assert "absent:Module" in result.output
    assert result.exception is not None and isinstance(result.exception, SystemExit)
    assert (checkpoint.read_bytes(), partial.read_bytes()) == before


def test_legacy_checkpoint_without_fingerprinted_modules_still_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepara(monkeypatch, 12, para_em=11)
    output = tmp_path / "runs"
    with pytest.raises(_Interrompe):
        _executa(output)
    run = _run_id(output)
    checkpoint = run / "progress.json"
    document = json.loads(checkpoint.read_text())
    document.pop("module_tunings", None)
    checkpoint.write_text(json.dumps(document))
    _prepara(monkeypatch, 12)

    result = _executa(output, "--resume", str(run))

    assert result.exit_code == 0, result.output
