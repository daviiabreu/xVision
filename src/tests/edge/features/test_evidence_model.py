"""O modelo linear sobre a evidencia: leitura das colunas e ida e volta do artefato."""

from __future__ import annotations

from pathlib import Path

import pytest

from edge.features.evidence_model import VALOR, Column, EvidenceModel

EVIDENCIA = {
    "reinicio_s": 1.5,
    "ultima_acao": "CROSS",
    "acoes": {"PASS": {"p": 0.8}},
    "falta": None,
}


@pytest.mark.parametrize(
    ("coluna", "esperado"),
    [
        (Column(("reinicio_s",), "number"), 1.5),
        (Column(("acoes", "PASS", "p"), "number"), 0.8),
        (Column(("acoes", "CROSS", "p"), "number"), 0.0),
        (Column(("falta",), "missing"), 1.0),
        (Column(("reinicio_s",), "missing"), 0.0),
        (Column(("ultima_acao",), "equals", "CROSS"), 1.0),
        (Column(("ultima_acao",), "equals", "PASS"), 0.0),
        (Column((VALOR,), "equals", "open play"), 1.0),
        (Column(("ultima_acao",), "number"), 0.0),
    ],
)
def test_cada_tipo_de_coluna_le_a_evidencia(coluna: Column, esperado: float) -> None:
    assert coluna.read("open play", EVIDENCIA) == esperado


def test_o_artefato_volta_do_disco_prevendo_igual(tmp_path: Path) -> None:
    modelo = EvidenceModel(
        feature="jogada",
        classes=("aberta", "parada"),
        columns=(Column(("reinicio_s",), "number"),),
        mean=(0.5,),
        scale=(1.0,),
        coef=((-2.0,), (2.0,)),
        intercept=(0.0, 0.0),
    )
    modelo.save(tmp_path / "m.json")
    lido = EvidenceModel.load(tmp_path / "m.json")

    amostras = [(None, {"reinicio_s": 0.0}), (None, {"reinicio_s": 2.0})]
    assert lido == modelo
    assert lido.predict(amostras) == ["aberta", "parada"]
    assert lido.probabilities(amostras).sum(axis=1) == pytest.approx([1.0, 1.0])
