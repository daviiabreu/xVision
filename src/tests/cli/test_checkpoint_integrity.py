"""A retomada recusa checkpoint corrompido antes de modificar eventos recuperaveis."""

import json
from pathlib import Path

import pytest

from .test_run_resume import (
    _executa,  # pyright: ignore[reportPrivateUsage]
    _Interrompe,  # pyright: ignore[reportPrivateUsage]
    _prepara,  # pyright: ignore[reportPrivateUsage]
    _run_id,  # pyright: ignore[reportPrivateUsage]
)


@pytest.mark.parametrize(
    "changes",
    [
        {"ok": -1},
        {"module_tunings": []},
        {"module_tunings": {"shot": 42}},
        {"module_tunings": {"shot": ""}},
        {"ok": "10"},
        {"ok": 10.9},
        {"failed": True},
        {"done": [str(index) for index in range(25)]},
        {"done": ["0"] * 10},
        {"done": "0123456789"},
        {
            "done": [*[str(index) for index in range(9)], "foreign"],
            "ok": 9,
            "failed": 1,
            "fontes": {str(index): "ball-break" for index in range(9)},
            "origens": {"ball-break": 9},
            "tempos": {"ball": [10.0] * 9},
        },
        {"fontes": {}},
        {"origens": {"midpoint": 10}},
        {"identidade": {"ball": ["yolo-ball"]}},
        {"tempos": {"ball": [float("nan")]}},
        {"tempos": {"ball": []}},
        {"identidade": {}},
        {"falhas": {"ball": ["invalid", None]}},
    ],
)
def test_invalid_checkpoint_preserves_partial_before_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changes: dict[str, object]
) -> None:
    _prepara(monkeypatch, 25, para_em=12)
    output = tmp_path / "runs"
    with pytest.raises(_Interrompe):
        _executa(output)
    run = _run_id(output)
    checkpoint = run / "progress.json"
    document = json.loads(checkpoint.read_text())
    document.update(changes)
    checkpoint.write_text(json.dumps(document))
    partial = run / "events.jsonl.parcial"
    before = partial.read_bytes(), checkpoint.read_bytes()
    processed = _prepara(monkeypatch, 25)

    result = _executa(output, "--resume", str(run))

    assert result.exit_code == 1, result.output
    assert "progress.json" in result.output or "checkpoint" in result.output
    assert not processed
    assert (partial.read_bytes(), checkpoint.read_bytes()) == before
    assert not (run / "events.jsonl").exists()


@pytest.mark.parametrize("replacement", ["foreign", "1"])
def test_replaced_event_prefix_preserves_checkpoint_and_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replacement: str
) -> None:
    _prepara(monkeypatch, 25, para_em=12)
    output = tmp_path / "runs"
    with pytest.raises(_Interrompe):
        _executa(output)
    run = _run_id(output)
    partial = run / "events.jsonl.parcial"
    lines = partial.read_text().splitlines()
    event = json.loads(lines[0])
    event["clip_id"] = replacement
    partial.write_text("\n".join([json.dumps(event), *lines[1:]]) + "\n")
    checkpoint = run / "progress.json"
    before = partial.read_bytes(), checkpoint.read_bytes()
    processed = _prepara(monkeypatch, 25)

    result = _executa(output, "--resume", str(run))

    assert result.exit_code == 1, result.output
    assert "prefixo" in result.output
    assert not processed
    assert (partial.read_bytes(), checkpoint.read_bytes()) == before
    assert not (run / "events.jsonl").exists()
