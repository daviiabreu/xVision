"""A prova ao vivo do card: o grafo inteiro roda, e o ciclo nomeia os dois culpados."""

from __future__ import annotations

from pathlib import Path

import pytest

from edge.graph.demo import main, run
from edge.graph.order import GraphError
from edge.modules.base import MODULE_KINDS
from edge.testing.make_clip import make_synthetic_clip

CONFIG = Path("config/pipeline.toml")


def test_the_default_configuration_runs_every_stage(tmp_path: Path) -> None:
    clipe = make_synthetic_clip(tmp_path / "synthetic.mp4")

    resultado = run(CONFIG, clipe, cycle=False)

    assert {etapa.kind for etapa in resultado.stages} == MODULE_KINDS
    assert all(etapa.status == "ok" for etapa in resultado.stages)


def test_a_cycle_names_both_modules_before_anything_runs(tmp_path: Path) -> None:
    clipe = make_synthetic_clip(tmp_path / "synthetic.mp4")

    with pytest.raises(GraphError) as erro:
        run(CONFIG, clipe, cycle=True)

    assert "players" in str(erro.value)
    assert "field" in str(erro.value)


def test_the_printed_table_carries_the_stage_the_set_and_the_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    clipe = make_synthetic_clip(tmp_path / "synthetic.mp4")
    monkeypatch.setattr("sys.argv", ["edge.graph.demo", "--clip", str(clipe)])

    main()

    saida = capsys.readouterr().out
    assert "ball-break@0.1.0" in saida
    assert "dense" in saida
    assert "bola em metros" in saida
