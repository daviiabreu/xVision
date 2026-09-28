"""`xvision treino`: evidencia de uma execucao vira modelo, validado nas dobras 0 a 2."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cli import app
from edge.features.evidence_model import EvidenceModel

runner = CliRunner()


def _execucao(tmp_path: Path) -> tuple[Path, Path, Path]:
    execucao = tmp_path / "run"
    execucao.mkdir()
    linhas: list[str] = []
    verdade: dict[str, dict[str, str]] = {}
    for i in range(60):
        parada = i % 5 == 0
        evidencia = {"reinicio_s": 2.0 if parada else None}
        feature = {"name": "jogada", "version": "0.3.0", "status": "ok", "value": "open play"}
        feature |= {"confidence": None, "reason": None, "evidence": evidencia}
        linhas.append(json.dumps({"clip_id": f"c{i}", "features": [feature]}))
        verdade[f"c{i}"] = {"jogada": "set piece" if parada else "open play"}
    (execucao / "events.jsonl").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    labels = tmp_path / "labels.toml"
    labels.write_text(
        '[[label]]\nname = "jogada"\ntable = "t"\ncolumn = "jogada"\ntype = "enum"\n'
        'domain = ["open play", "set piece", "individual"]\nmissing = [""]\n'
        '[[metric]]\nfeature = "jogada"\nlabel = "jogada"\n',
        encoding="utf-8",
    )
    truth = tmp_path / "truth.json"
    truth.write_text(json.dumps(verdade), encoding="utf-8")
    return execucao, labels, truth


def test_treino_grava_o_modelo_e_mostra_regra_e_maioria_ao_lado(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cli

    execucao, labels, truth = _execucao(tmp_path)

    class Registro:
        def __init__(self, i: int) -> None:
            self.shot_id, self.fold = f"c{i}", i % 5

    monkeypatch.setattr(cli, "load_catalog", lambda: [Registro(i) for i in range(60)])
    saida = tmp_path / "modelo.json"

    resultado = runner.invoke(
        app,
        [
            "treino",
            str(execucao),
            "--feature",
            "jogada",
            "--labels",
            str(labels),
            "--truth",
            str(truth),
            "--saida",
            str(saida),
            "--reservado",
        ],
    )

    assert resultado.exit_code == 0, resultado.output
    assert "36 clipes nas dobras 0 a 2" in resultado.output
    assert "regra" in resultado.output and "maioria" in resultado.output
    assert "reservado (24 clipes)" in resultado.output
    modelo = EvidenceModel.load(saida)
    assert modelo.predict([("open play", {"reinicio_s": 2.0})]) == ["set piece"]
    assert json.loads(saida.with_suffix(".relatorio.json").read_text())["folds"] == [0, 1, 2]


def test_treino_sem_rotulo_para_a_feature_e_recusado(tmp_path: Path) -> None:
    execucao, labels, _ = _execucao(tmp_path)

    resultado = runner.invoke(
        app, ["treino", str(execucao), "--feature", "outra", "--labels", str(labels)]
    )

    assert resultado.exit_code == 1
    assert "nao tem rotulo" in resultado.output
