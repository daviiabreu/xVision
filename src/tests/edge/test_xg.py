"""xG da Footstats: a conta exata e a regra para feature que nao mediu."""

from __future__ import annotations

import csv
from collections.abc import Mapping

import pytest

from contracts import FeatureValue
from contracts.events import FeatureStatus
from edge.features.quadrant import celula_px, valor
from edge.paths import data_root
from edge.xg import (
    ASSISTENCIA_DECISIVA,
    ASSISTENCIA_REGULAR,
    ASSISTENCIA_SEM_NIVEL,
    MEDIA_OLHOU,
    arredondar,
    defesa,
    expected_goals,
    xg_footstats,
)
from mlops.catalog import SHOTS, XG


def _f(
    nome: str, valor: float | str | bool | None, status: FeatureStatus = "ok", **evidencia: object
) -> FeatureValue:
    return FeatureValue(
        name=nome,
        version="0",
        status=status,
        value=valor,
        confidence=None,
        reason=None if status == "ok" else "teste",
        evidence=evidencia,
    )


def _features(*fora: str, **troca: FeatureValue) -> Mapping[str, FeatureValue]:
    todas = {
        "quadrant": _f("quadrant", 0.49, linha=1),
        "type_of_play": _f("type_of_play", "open play"),
        "assist": _f("assist", False),
        "marked": _f("marked", True),
        "occupied_area": _f("occupied_area", False),
        "looked_before": _f("looked_before", True),
        "body_part": _f("body_part", "PERNA_DIREITA"),
        "goalkeeper_position": _f("goalkeeper_position", False),
    }
    todas.update(troca)
    return {nome: f for nome, f in todas.items() if nome not in fora}


@pytest.mark.parametrize(
    ("componentes", "goleiro", "esperado"),
    [
        ((0.49, 0.072, 0.21, 0.035, 0.111, 0.012), False, 0.790),
        ((0.02, 0.035, 0.0, 0.006, 0.014, 0.008), True, 0.035),
        ((0.15, 0.059, 0.075, 0.035, 0.111, 0.012), False, 0.376),
    ],
)
def test_exemplos_do_documento(
    componentes: tuple[float, ...], goleiro: bool, esperado: float
) -> None:
    assert xg_footstats(*componentes, goleiro_posicionado=goleiro) == esperado


def test_arredonda_o_valor_binario_como_o_javascript() -> None:
    assert (arredondar(0.7905), arredondar(0.0355), arredondar(0.07055)) == (0.79, 0.035, 0.071)


def test_tudo_medido_sai_ok() -> None:
    xg = expected_goals(_features())

    assert (xg.status, xg.estimados) == ("ok", ())
    assert xg.valor == xg_footstats(0.49, 0.035, 0.0, 0.012, 0.111, 0.012, False)


def test_sem_quadrante_nao_ha_xg() -> None:
    for features in (
        _features("quadrant"),
        _features(quadrant=_f("quadrant", None, "unavailable")),
    ):
        xg = expected_goals(features)

        assert (xg.valor, xg.status) == (None, "unavailable")
        assert xg.motivo


def test_assistencia_entra_sem_nivel() -> None:
    xg = expected_goals(_features(assist=_f("assist", True)))

    assert xg.valor == xg_footstats(0.49, 0.035, ASSISTENCIA_SEM_NIVEL, 0.012, 0.111, 0.012, False)
    assert xg.status == "ok"


def test_feature_que_nao_mediu_entra_pela_media() -> None:
    xg = expected_goals(_features(looked_before=_f("looked_before", None, "unavailable")))

    assert (xg.status, xg.estimados) == ("low_confidence", ("looked_before",))
    assert xg.valor == xg_footstats(0.49, 0.035, 0.0, 0.012, MEDIA_OLHOU, 0.012, False)


def test_defesa_usa_o_campo_que_mediu() -> None:
    xg = expected_goals(_features("occupied_area"))

    assert xg.estimados == ("occupied_area",)
    assert xg.valor == xg_footstats(0.49, 0.035, 0.0, 0.0077, 0.111, 0.012, False)


@pytest.mark.parametrize(("linha", "fator"), [(1, 0.755), (3, 0.577), (7, 0.516)])
def test_goleiro_faltando_usa_o_fator_da_linha(linha: int, fator: float) -> None:
    xg = expected_goals(
        _features("goalkeeper_position", quadrant=_f("quadrant", 0.49, linha=linha))
    )

    sem_goleiro = xg_footstats(0.49, 0.035, 0.0, 0.012, 0.111, 0.012, False)
    assert xg.valor == arredondar(sem_goleiro * fator)
    assert xg.estimados == ("goalkeeper_position",)


def test_cabecada_e_quadrante_inferido() -> None:
    xg = expected_goals(
        _features(
            body_part=_f("body_part", "CABECA"),
            quadrant=_f("quadrant", 0.01, "low_confidence", linha=9),
        )
    )

    assert xg.valor == xg_footstats(0.01, 0.035, 0.0, 0.012, 0.111, 0.008, False)
    assert (xg.status, xg.estimados) == ("low_confidence", ("quadrant",))


DB = data_root(required=False) / "db"
COMPLETAS = (
    "type_of_play",
    "assist",
    "marked",
    "looked_before",
    "used",
    "occupied_area",
    "goalkeaper_position",
    "expected_goals",
)


@pytest.mark.skipif(not (DB / XG).is_file(), reason="o dump da Footstats nao esta na maquina")
def test_reproduz_o_expected_goals_gravado() -> None:
    with (DB / SHOTS).open(encoding="utf-8") as arquivo:
        posicoes = {r["id"]: r for r in csv.DictReader(arquivo)}
    with (DB / XG).open(encoding="utf-8") as arquivo:
        linhas = [r for r in csv.DictReader(arquivo) if all(r[c] for c in COMPLETAS)]

    exatas = 0
    for r in linhas:
        s = posicoes[r["id_match_shot"]]
        quadrante = valor(*celula_px(float(s["field_position_x"]), float(s["field_position_y"])))
        jogada = {"open play": 0.035, "individual": 0.059, "set piece": 0.072}[r["type_of_play"]]
        niveis = (ASSISTENCIA_REGULAR, ASSISTENCIA_DECISIVA) if r["assist"] == "true" else (0.0,)
        resto = (
            defesa(r["marked"] == "true", r["occupied_area"] == "true"),
            0.111 if r["looked_before"] == "true" else 0.014,
            0.008 if r["used"] == "head" else 0.012,
        )
        goleiro = r["goalkeaper_position"] == "true"
        calculados = {xg_footstats(quadrante, jogada, n, *resto, goleiro) for n in niveis}
        exatas += float(r["expected_goals"]) in calculados

    # Ficam de fora as 89 gravadas com quadrante 0 e 5 que contradizem a propria celula.
    assert (exatas, len(linhas)) == (2414, 2508)
