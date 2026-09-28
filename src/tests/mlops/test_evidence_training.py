"""Treino da softmax sobre a evidencia, com validacao deixando uma dobra de fora."""

from __future__ import annotations

import random
from typing import Any, cast

import pytest

from edge.features.evidence_model import VALOR, Column
from mlops.evidence_training import Exemplo, discover_columns, evaluate_model, train


def _exemplos(n: int = 90, raros: int = 12) -> list[Exemplo]:
    """`reinicio_s` separa bola parada; a regra publicada acerta so a classe comum."""
    sorteio = random.Random(0)
    exemplos: list[Exemplo] = []
    for i in range(n):
        classe = "set piece" if i < raros else "open play"
        reinicio = sorteio.uniform(1.0, 3.0) if classe == "set piece" else None
        evidencia = {"reinicio_s": reinicio, "ruido": sorteio.random(), "leitura": f"texto {i}"}
        exemplos.append(Exemplo(f"c{i}", i % 3, "open play", evidencia, classe))
    return exemplos


def test_colunas_descobertas_por_tipo_e_texto_livre_fica_de_fora() -> None:
    colunas = discover_columns(
        [
            ("open play", {"a": 1.0, "b": "x", "c": {"d": True}, "leitura": "livre"}),
            ("set piece", {"a": None, "b": "y", "c": {"d": False}}),
        ]
    )

    assert set(colunas) == {
        Column((VALOR,), "equals", "open play"),
        Column((VALOR,), "equals", "set piece"),
        Column(("a",), "number"),
        Column(("a",), "missing"),
        Column(("b",), "equals", "x"),
        Column(("b",), "equals", "y"),
        Column(("c", "d"), "number"),
    }


def test_o_modelo_aprende_a_classe_rara_que_a_regra_nao_ve() -> None:
    modelo, relatorio = train(_exemplos(), "type_of_play")

    escolhido = cast("dict[str, Any]", relatorio["escolhido"])
    regra = cast("dict[str, Any]", relatorio["regra"])
    assert escolhido["macro_f1"] > 0.9
    assert escolhido["macro_f1"] > regra["macro_f1"]
    assert escolhido["per_class"]["set piece"]["recall"] == 1.0
    assert relatorio["folds"] == [0, 1, 2]
    assert modelo.predict([("open play", {"reinicio_s": 2.0, "ruido": 0.5})]) == ["set piece"]


def test_a_passada_no_reservado_usa_o_modelo_congelado() -> None:
    modelo, _ = train(_exemplos(), "type_of_play")
    reservados = [Exemplo(e.shot_id, 3, e.valor, e.evidencia, e.rotulo) for e in _exemplos(30, 5)]

    resultado = evaluate_model(modelo, reservados)

    assert resultado["n"] == 30
    assert cast("dict[str, Any]", resultado["modelo"])["macro_f1"] > 0.9


def test_treino_sem_duas_classes_ou_duas_dobras_e_recusado() -> None:
    uma_classe = [e for e in _exemplos() if e.rotulo == "open play"]
    uma_dobra = [Exemplo(e.shot_id, 0, e.valor, e.evidencia, e.rotulo) for e in _exemplos()]

    for exemplos in (uma_classe, uma_dobra):
        with pytest.raises(ValueError, match="duas"):
            train(exemplos, "type_of_play")
