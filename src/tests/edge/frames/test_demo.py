"""A prova ao vivo do card precisa mostrar zero chamadas na segunda execucao."""

from __future__ import annotations

from pathlib import Path

import pytest

from edge.frames.demo import main, run
from edge.testing.make_clip import make_synthetic_clip


def test_the_warm_run_calls_no_module_and_the_new_version_calls_again(tmp_path: Path) -> None:
    clip = make_synthetic_clip(tmp_path / "prova.mp4")
    cache = tmp_path / "cache"

    fria = run(clip, frames=5, module_version="0.1.0", cache_root=cache)
    quente = run(clip, frames=5, module_version="0.1.0", cache_root=cache)
    trocada = run(clip, frames=5, module_version="0.2.0", cache_root=cache)

    assert (fria.chamadas_ao_modulo, quente.chamadas_ao_modulo) == (1, 0)
    assert trocada.chamadas_ao_modulo == 1
    assert fria.quadros == quente.quadros == 5


def test_the_printed_line_carries_the_counter_the_card_asks_for(tmp_path: Path) -> None:
    resultado = run(
        make_synthetic_clip(tmp_path / "prova.mp4"),
        frames=3,
        module_version="0.1.0",
        cache_root=tmp_path / "cache",
    )

    assert "chamadas_ao_modulo=1" in str(resultado)
    assert "modulo=example@0.1.0" in str(resultado)
    assert "quadros=3" in str(resultado)
    assert "tempo_ms=" in str(resultado)


def test_main_prints_the_line_with_the_options_the_card_asks_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    clip = make_synthetic_clip(tmp_path / "prova.mp4")
    argumentos = [
        "edge.frames.demo",
        "--clip",
        str(clip),
        "--frames",
        "4",
        "--module-version",
        "9.9.9",
        "--cache",
        str(tmp_path / "cache"),
    ]
    monkeypatch.setattr("sys.argv", argumentos)

    main()

    saida = capsys.readouterr().out
    assert "modulo=example@9.9.9" in saida
    assert "quadros=4" in saida
    assert "chamadas_ao_modulo=1" in saida


def test_the_clip_id_comes_from_the_content_and_not_from_the_file_name(tmp_path: Path) -> None:
    caminho = tmp_path / "synthetic.mp4"
    cache = tmp_path / "cache"
    make_synthetic_clip(caminho, seed=0)
    primeira = run(caminho, frames=3, module_version="0.1.0", cache_root=cache)

    make_synthetic_clip(caminho, seed=6)
    outra = run(caminho, frames=3, module_version="0.1.0", cache_root=cache)

    assert (primeira.chamadas_ao_modulo, outra.chamadas_ao_modulo) == (1, 1)
