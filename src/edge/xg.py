"""xG da Footstats sobre as features do chute, com a regra para campo que nao foi medido."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Literal

from contracts import FeatureValue

ASSISTENCIA_REGULAR = 0.075
ASSISTENCIA_DECISIVA = 0.21
# A feature nao separa regular de decisiva: 561 e 162 das 723 assistidas.
ASSISTENCIA_SEM_NIVEL = 0.1052
FATOR_DE_AJUSTE = 0.85

JOGADA: Mapping[object, float] = {"open play": 0.035, "individual": 0.059, "set piece": 0.072}
ASSISTENCIA: Mapping[object, float] = {True: ASSISTENCIA_SEM_NIVEL, False: 0.0}
OLHOU: Mapping[object, float] = {True: 0.111, False: 0.014}
USADO: Mapping[object, float] = {"CABECA": 0.008, "PERNA_ESQUERDA": 0.012, "PERNA_DIREITA": 0.012}

# Medias dos rotulos da Footstats (2.508 finalizacoes), no lugar da feature que nao mediu.
MEDIA_JOGADA = 0.0454
MEDIA_ASSISTENCIA = 0.0303
MEDIA_OLHOU = 0.0239
MEDIA_USADO = 0.0114
MEDIA_DEFESA = 0.0102
DEFESA_SO_MARCADO = {True: 0.0077, False: 0.0279}
DEFESA_SO_AREA = {True: 0.0063, False: 0.0180}
# Razao entre o xG gravado e o xG sem goleiro, pelas linhas 1, 2, 3 e 4 em diante do quadrante.
FATOR_GOLEIRO = (0.755, 0.629, 0.577, 0.516)

Status = Literal["ok", "low_confidence", "unavailable"]


@dataclass(frozen=True)
class ExpectedGoals:
    valor: float | None
    status: Status
    estimados: tuple[str, ...] = ()
    motivo: str | None = None


def arredondar(valor: float) -> float:
    """O `toFixed(3)` do JavaScript: meio para cima sobre o valor binario exato."""
    return float(Decimal(valor).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def defesa(marcado: bool, area_ocupada: bool) -> float:
    if marcado and area_ocupada:
        return 0.006
    if marcado or area_ocupada:
        return 0.012
    return 0.035


def xg_footstats(
    quadrante: float,
    jogada: float,
    assistencia: float,
    defesa: float,
    olhou: float,
    usado: float,
    goleiro_posicionado: bool,
) -> float:
    """A conta da Footstats, com os dois arredondamentos que reproduzem o valor gravado."""
    soma = (
        quadrante * 1000
        + jogada * 1000
        + assistencia * 1000
        + defesa * 1000
        + olhou * 1000
        + usado * 1000
    )
    xg = arredondar(soma / 1000 * FATOR_DE_AJUSTE)
    return arredondar(xg / 2) if goleiro_posicionado else xg


def expected_goals(features: Mapping[str, FeatureValue]) -> ExpectedGoals:
    quadrante = _medido(features, "quadrant")
    if not isinstance(quadrante, float):
        return ExpectedGoals(None, "unavailable", motivo="sem quadrante o xG nao e calculado")
    estimados = ["quadrant"] if features["quadrant"].status == "low_confidence" else []

    def componente(nome: str, tabela: Mapping[object, float], media: float) -> float:
        valor = _medido(features, nome)
        if valor in tabela:
            return tabela[valor]
        estimados.append(nome)
        return media

    sem_goleiro = xg_footstats(
        quadrante,
        componente("type_of_play", JOGADA, MEDIA_JOGADA),
        componente("assist", ASSISTENCIA, MEDIA_ASSISTENCIA),
        _defesa(features, estimados),
        componente("looked_before", OLHOU, MEDIA_OLHOU),
        componente("body_part", USADO, MEDIA_USADO),
        goleiro_posicionado=False,
    )
    goleiro = _medido(features, "goalkeeper_position")
    if isinstance(goleiro, bool):
        valor = arredondar(sem_goleiro / 2) if goleiro else sem_goleiro
    else:
        estimados.append("goalkeeper_position")
        linha = features["quadrant"].evidence.get("linha")
        fator = FATOR_GOLEIRO[min(linha, 4) - 1] if isinstance(linha, int) else FATOR_GOLEIRO[-1]
        valor = arredondar(sem_goleiro * fator)
    return ExpectedGoals(valor, "low_confidence" if estimados else "ok", tuple(estimados))


def _medido(features: Mapping[str, FeatureValue], nome: str) -> object:
    feature = features.get(nome)
    if feature is None or feature.status == "unavailable":
        return None
    return feature.value


def _defesa(features: Mapping[str, FeatureValue], estimados: list[str]) -> float:
    marcado = _medido(features, "marked")
    area = _medido(features, "occupied_area")
    if isinstance(marcado, bool) and isinstance(area, bool):
        return defesa(marcado, area)
    if isinstance(marcado, bool):
        estimados.append("occupied_area")
        return DEFESA_SO_MARCADO[marcado]
    if isinstance(area, bool):
        estimados.append("marked")
        return DEFESA_SO_AREA[area]
    estimados.extend(("marked", "occupied_area"))
    return MEDIA_DEFESA
