"""`XVISION_CONFIG_DIR` precisa valer para quem roda o pipeline, e nao so para o mlops.

A variavel era aceita e ignorada pelo CLI e pela API, porque as duas fixavam o caminho numa
constante de modulo, avaliada no import antes de qualquer ambiente ser lido. O efeito e o pior
tipo: apontando a variavel para uma copia com outro `impl`, as sete etapas saiam `ok` e o evento
trazia o modulo antigo. O operador trocava o modelo, o relatorio dizia que rodou, e rodou o de
antes.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cli import app
from edge.paths import pipeline_config
from edge.testing.make_clip import make_synthetic_clip

runner = CliRunner()


@pytest.fixture
def config_alternativa(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Uma copia da configuracao real com `players` trocado por um modulo que quebra."""
    destino = tmp_path / "config"
    destino.mkdir()
    for arquivo in Path("config").glob("*.toml"):
        shutil.copy(arquivo, destino / arquivo.name)
    alvo = destino / "pipeline.toml"
    trocado = re.sub(
        r'(\[modules\.players\]\s*\n\s*impl\s*=\s*)"[^"]+"',
        r'\1"edge.testing.example_module:QuebraNoLoad"',
        alvo.read_text(encoding="utf-8"),
    )
    assert trocado != alvo.read_text(encoding="utf-8")
    alvo.write_text(trocado, encoding="utf-8")
    monkeypatch.setenv("XVISION_CONFIG_DIR", str(destino))
    return destino


def test_the_path_is_resolved_on_call_and_not_frozen_at_import(
    config_alternativa: Path,
) -> None:
    """Constante de modulo congela antes de o ambiente existir, e a variavel vira decoracao."""
    assert pipeline_config() == config_alternativa / "pipeline.toml"


def test_the_cli_runs_the_module_the_variable_points_at(
    config_alternativa: Path, tmp_path: Path
) -> None:
    clipe = make_synthetic_clip(tmp_path / "s.mp4")

    resultado = runner.invoke(app, ["run", "--clip", str(clipe)])

    assert "quebra-no-load" in resultado.output, (
        "o CLI rodou o modulo da configuracao de producao, ignorando XVISION_CONFIG_DIR"
    )


def test_an_explicit_option_still_wins_over_the_variable(
    config_alternativa: Path, tmp_path: Path
) -> None:
    """A guarda da correcao: resolver o padrao nao pode atropelar quem passou `--config`."""
    clipe = make_synthetic_clip(tmp_path / "s.mp4")

    resultado = runner.invoke(
        app, ["run", "--clip", str(clipe), "--config", "config/pipeline.toml"]
    )

    assert "quebra-no-load" not in resultado.output
    assert "stub-players" in resultado.output or "players" in resultado.output


def test_the_service_reads_the_same_variable(config_alternativa: Path) -> None:
    """A API tinha a mesma constante congelada, e o mesmo efeito silencioso."""
    from api import create_app

    app_http = create_app()

    assert app_http.title == "xVision"
    from api import CONFIG_ATUAL

    assert CONFIG_ATUAL[0] == config_alternativa / "pipeline.toml"
