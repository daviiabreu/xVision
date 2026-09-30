"""`xvision run` le `[decode].mode`, entrega ao provider e grava no `run.json`.

O defeito que estes testes existem para pegar e o da chave aceita e ignorada: o TOML diz
`redecode`, a execucao termina, e o provider rodou em `keep`. A medida de memoria sairia do modo
errado com a cara do certo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import cli
from cli import app
from edge.frames.provider import Decode, FrameProvider, FrameSource
from edge.testing.make_clip import make_synthetic_clip

runner = CliRunner()
CONFIG = Path("config/pipeline.toml")


@dataclass(frozen=True)
class _Registro:
    shot_id: str
    clip_path: Path
    fold: int = 0
    match_id: str = "900"


def _config(tmp_path: Path, mode: str | None) -> Path:
    texto = CONFIG.read_text(encoding="utf-8")
    if mode is not None:
        texto += f'\n[decode]\nmode = "{mode}"\n'
    destino = tmp_path / f"pipeline-{mode or 'padrao'}.toml"
    destino.write_text(texto, encoding="utf-8")
    return destino


@pytest.fixture
def modos(monkeypatch: pytest.MonkeyPatch) -> list[Decode]:
    vistos: list[Decode] = []
    original = cli.FrameProvider

    def espiao(source: FrameSource, *, decode: Decode = "keep") -> FrameProvider:
        vistos.append(decode)
        return original(source, decode=decode)

    monkeypatch.setattr(cli, "FrameProvider", espiao)
    return vistos


@pytest.fixture
def catalogo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[_Registro]:
    registros = [
        _Registro(str(700 + i), make_synthetic_clip(tmp_path / "cortes" / f"{700 + i}.mp4"))
        for i in range(2)
    ]

    def rotulos(_: Path) -> object:
        return object()

    def carrega() -> list[_Registro]:
        return registros

    def populacao(itens: Any, _tabela: Any) -> list[Any]:
        return list(itens)

    monkeypatch.setattr(cli, "read_labels", rotulos)
    monkeypatch.setattr(cli, "load_catalog", carrega)
    monkeypatch.setattr(cli, "_populacao", populacao)
    return registros


def _run_json(saida: Path) -> dict[str, Any]:
    (registro,) = list(saida.rglob("run.json"))
    return json.loads(registro.read_text(encoding="utf-8"))


@pytest.mark.parametrize(("mode", "esperado"), [(None, "keep"), ("redecode", "redecode")])
def test_run_all_hands_the_mode_to_the_provider_and_records_it(
    tmp_path: Path,
    catalogo: list[_Registro],
    modos: list[Decode],
    mode: str | None,
    esperado: Decode,
) -> None:
    saida = tmp_path / "execucoes"

    resultado = runner.invoke(
        app, ["run", "--all", "--config", str(_config(tmp_path, mode)), "--out", str(saida)]
    )

    assert resultado.exit_code == 0, resultado.output
    assert modos == [esperado] * len(catalogo)
    dados = _run_json(saida)
    assert dados["decode"] == esperado
    assert dados["dataset"]["clips"] == len(catalogo)


def test_run_all_refuses_an_unknown_mode_before_touching_a_clip(
    tmp_path: Path, catalogo: list[_Registro], modos: list[Decode]
) -> None:
    saida = tmp_path / "execucoes"

    resultado = runner.invoke(
        app, ["run", "--all", "--config", str(_config(tmp_path, "sim")), "--out", str(saida)]
    )

    assert resultado.exit_code == 1
    assert "[decode].mode = 'sim'" in resultado.output
    assert modos == []
    assert not saida.exists()


def _evento(tmp_path: Path, clipe: Path, mode: str | None) -> dict[str, Any]:
    """Cada modo com o proprio `var/`, senao o segundo le a percepcao do primeiro no cache e nao
    decodifica nada."""
    saida = tmp_path / f"saida-{mode}"
    with pytest.MonkeyPatch.context() as isolado:
        isolado.setenv("XVISION_VAR_DIR", str(tmp_path / f"var-{mode}"))
        resultado = runner.invoke(
            app,
            [
                "run",
                "--clip",
                str(clipe),
                "--config",
                str(_config(tmp_path, mode)),
                "--out",
                str(saida),
            ],
        )
    assert resultado.exit_code == 0, resultado.output
    (linha,) = (saida / clipe.stem / "events.jsonl").read_text(encoding="utf-8").splitlines()
    evento = json.loads(linha)
    for volatil in ("timings_ms", "started_at"):
        del evento["pipeline"][volatil]
    return evento


def test_run_clip_honours_the_mode_and_writes_the_same_event(
    tmp_path: Path, modos: list[Decode]
) -> None:
    clipe = make_synthetic_clip(tmp_path / "synthetic.mp4")

    guardado = _evento(tmp_path, clipe, None)
    redecodificado = _evento(tmp_path, clipe, "redecode")

    assert modos == ["keep", "redecode"]
    assert redecodificado == guardado


def test_run_clip_refuses_an_unknown_mode_in_one_line(tmp_path: Path) -> None:
    clipe = make_synthetic_clip(tmp_path / "synthetic.mp4")

    resultado = runner.invoke(
        app, ["run", "--clip", str(clipe), "--config", str(_config(tmp_path, "per_species"))]
    )

    assert resultado.exit_code == 1
    assert "[decode].mode = 'per_species'" in resultado.output
    assert "Traceback" not in resultado.output
