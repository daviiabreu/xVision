"""O filtro de dobra dos comandos que abrem clipe anotado.

Ele existe porque `run --gabarito` abre os 38 clipes anotados e 14 deles sao o pacote reservado.
Sem o filtro, medir uma melhoria durante o desenvolvimento consome o holdout sem ninguem decidir
isso, e o consumo nao aparece em lugar nenhum: a execucao termina com sucesso.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from cli import DOBRAS_DA_BOLA, app, dobras_de

runner = CliRunner()


def _consenso(valor: dict[str, int]) -> Any:
    def consensus(_: Any) -> dict[str, int]:
        return valor

    return consensus


def _nada(_: Any) -> None:
    return None


def _trinta(_: Any) -> float:
    return 30.0


def test_o_vocabulario_de_dobra_e_um_so() -> None:
    assert dobras_de("ajuste") == frozenset({0, 1, 2})
    assert dobras_de("reservado") == frozenset({3, 4})
    assert dobras_de("tudo") == frozenset({0, 1, 2, 3, 4})


def test_ajuste_e_reservado_nao_se_encostam() -> None:
    """A dobra e por partida: um clipe de ajuste e um reservado nunca vem do mesmo gramado."""
    assert not dobras_de("ajuste") & dobras_de("reservado")
    assert dobras_de("ajuste") | dobras_de("reservado") == dobras_de("tudo")


def test_dobra_desconhecida_lista_o_que_existe() -> None:
    resultado = runner.invoke(app, ["run", "--all", "--dobra", "desenvolvimento"])

    assert resultado.exit_code == 1
    assert "nao existe" in resultado.output
    for nome in DOBRAS_DA_BOLA:
        assert nome in resultado.output


def test_eval_instant_recusa_dobra_desconhecida_antes_de_abrir_o_gabarito(
    tmp_path: Path,
) -> None:
    """A recusa vem antes de ler a anotacao: erro de digitacao nao pode virar erro de dado
    faltando, que manda a pessoa procurar defeito no lugar errado."""
    execucao = tmp_path / "run"
    execucao.mkdir()
    (execucao / "events.jsonl").write_text("", encoding="utf-8")

    resultado = runner.invoke(app, ["eval-instant", str(execucao), "--dobra", "dev"])

    assert resultado.exit_code == 1
    assert "dobra 'dev' nao existe" in resultado.output


def test_eval_instant_filtra_o_gabarito_pela_dobra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Um `run` que contenha clipe reservado ainda pode ser lido so na parte de desenvolvimento."""
    import cli

    execucao = tmp_path / "run"
    execucao.mkdir()
    linhas = [
        {"clip_id": "dev", "shot_timestamp_s": 1.0},
        {"clip_id": "res", "shot_timestamp_s": 1.0},
    ]
    (execucao / "events.jsonl").write_text(
        "".join(json.dumps(linha) + "\n" for linha in linhas), encoding="utf-8"
    )

    class Registro:
        def __init__(self, shot_id: str, fold: int) -> None:
            self.shot_id = shot_id
            self.fold = fold
            self.big_enough = True

    monkeypatch.setattr(cli, "read", lambda: ())
    monkeypatch.setattr(cli, "consensus", _consenso({"dev": 30, "res": 30}))
    monkeypatch.setattr(cli, "agreement", _nada)
    monkeypatch.setattr(cli, "load_catalog", lambda: [Registro("dev", 0), Registro("res", 4)])
    monkeypatch.setattr(cli, "_fps_do_clipe", _trinta)

    vistos: list[dict[str, int]] = []

    def espia(previsto: dict[str, int], gabarito: dict[str, int], **_: Any) -> Any:
        vistos.append(dict(gabarito))
        raise SystemExit(0)

    monkeypatch.setattr(cli, "measure", espia)
    runner.invoke(app, ["eval-instant", str(execucao), "--dobra", "ajuste"])

    assert vistos == [{"dev": 30}]


def test_eval_instant_avisa_quando_a_dobra_fica_vazia(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cli

    execucao = tmp_path / "run"
    execucao.mkdir()
    (execucao / "events.jsonl").write_text("", encoding="utf-8")

    class Registro:
        def __init__(self) -> None:
            self.shot_id = "res"
            self.fold = 4
            self.big_enough = True

    monkeypatch.setattr(cli, "read", lambda: ())
    monkeypatch.setattr(cli, "consensus", _consenso({"res": 30}))
    monkeypatch.setattr(cli, "load_catalog", lambda: [Registro()])

    resultado = runner.invoke(app, ["eval-instant", str(execucao), "--dobra", "ajuste"])

    assert resultado.exit_code == 1
    assert "nenhum clipe anotado na dobra ajuste" in resultado.output
