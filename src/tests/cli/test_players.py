"""Os comandos de jogadores, que antes eram um executavel proprio em `argparse`.

O teste que existia conferia o `parse_args` do `argparse`, entao provava so que a bandeira estava
declarada. Aqui a chamada e a mesma que uma pessoa faz no terminal, e o que se cobra e que ela
chegue na funcao certa com os valores certos: `identify` continua sem exigir peso, que era o ponto
do teste antigo, e `run` e `batch` continuam exigindo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from cli import app

runner = CliRunner()


@pytest.fixture
def chamadas(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    registro: list[dict[str, Any]] = []

    def falso_identificar(*args: Any, **kwargs: Any) -> dict[str, Any]:
        registro.append({"args": args, "kwargs": kwargs})
        return {"tracks": [{"status": "available"}, {"status": "missing"}]}

    monkeypatch.setattr("cli.players.process_identification", falso_identificar)
    return registro


def test_identify_nao_exige_o_peso_do_detector(chamadas: list[dict[str, Any]]) -> None:
    """Classificar equipe trabalha sobre tracking pronto, entao pedir peso seria pedir a toa."""
    resultado = runner.invoke(app, ["players", "identify", "clipe.mp4", "deteccoes.json", "saida"])

    assert resultado.exit_code == 0, resultado.output
    assert "1 trajetorias classificadas" in resultado.output
    assert chamadas[0]["args"][:3] == (Path("clipe.mp4"), Path("deteccoes.json"), Path("saida"))
    assert chamadas[0]["kwargs"] == {"max_samples": 5, "minimum_samples": 2}


def test_run_exige_o_peso(chamadas: list[dict[str, Any]]) -> None:
    resultado = runner.invoke(app, ["players", "run", "clipe.mp4", "saida"])

    assert resultado.exit_code != 0
    assert not chamadas, "o comando chegou a rodar sem o peso declarado"


def test_os_quatro_comandos_estao_no_xvision() -> None:
    """Eles vinham de um executavel proprio, e o ponto da mudanca e existirem aqui dentro."""
    ajuda = runner.invoke(app, ["players", "--help"]).output

    for comando in ("run", "batch", "identify", "debug"):
        assert comando in ajuda, f"`xvision players {comando}` sumiu"
