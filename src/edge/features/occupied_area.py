"""Area ocupada no chute: jogadores no cone ate as traves, na grande area e perto do finalizador.

Nao ha definicao do parceiro, entao a evidencia traz todas as medidas candidatas e a decisao e uma
regra preliminar, para ser trocada por um modelo treinado contra o rotulo.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

from contracts import FeatureValue, Perception, PlayerDetection
from edge.features.base import FeatureSpec, ShotOutput, unavailable
from edge.features.scene import SceneRefusal, shot_scene
from edge.modules.field.geometry import PITCH_105X68

MEIA_LARGURA_DO_GOL_M = 3.66
PROFUNDIDADE_DA_AREA_M = 16.5
MEIA_LARGURA_DA_AREA_M = 20.16
RAIOS_M = (2.0, 3.0, 5.0)

LIMIAR_PRELIMINAR = 1
"""Jogadores no cone a partir dos quais a area conta como ocupada. Preliminar, sem gabarito."""

Ponto = tuple[float, float]


class OccupiedArea:
    spec = FeatureSpec(
        name="occupied_area",
        version="0.1.0",
        requires=("players", "ball", "field", "tracking"),
        unit="bool",
    )

    def extract(self, perception: Perception, shot: ShotOutput) -> FeatureValue:
        cena = shot_scene(perception, shot)
        if isinstance(cena, SceneRefusal):
            return unavailable(self.spec, cena.detail)
        medidas = measure_occupation(cena.frame.players, cena.shooter, cena.anchor_xy)
        return FeatureValue(
            name=self.spec.name,
            version=self.spec.version,
            status="ok",
            value=cast("int", medidas["jogadores_no_cone"]) >= LIMIAR_PRELIMINAR,
            confidence=None,
            reason=None,
            evidence={
                **medidas,
                "quadro_de_referencia": cena.frame.frame_index,
                "defasagem_s": round(cena.offset_s, 4),
                "origem_do_finalizador": cena.shooter_source,
            },
        )


def measure_occupation(
    jogadores: Sequence[PlayerDetection], finalizador: PlayerDetection, ancora: Ponto
) -> dict[str, object]:
    """As medidas em metro em volta do finalizador; contagem de adversario sem time sai `None`."""
    meio = PITCH_105X68.length_m / 2.0
    gol_x = meio if ancora[0] >= 0 else -meio
    traves = ((gol_x, -MEIA_LARGURA_DO_GOL_M), (gol_x, MEIA_LARGURA_DO_GOL_M))
    outros = [
        j
        for j in jogadores
        if j.pitch_xy is not None and j.track_id != finalizador.track_id and j.role != "referee"
    ]
    goleiros = [j for j in outros if j.role == "goalkeeper"]
    linha = [j for j in outros if j.role != "goalkeeper"]
    time = finalizador.team
    adversarios = None if time is None else [j for j in linha if j.team not in (None, time)]

    def no_cone(j: PlayerDetection) -> bool:
        return _no_triangulo(_xy(j), ancora, *traves)

    def na_area(j: PlayerDetection) -> bool:
        x, y = _xy(j)
        return (
            x * gol_x > 0
            and abs(x) >= meio - PROFUNDIDADE_DA_AREA_M
            and abs(y) <= MEIA_LARGURA_DA_AREA_M
        )

    distancias = [math.dist(_xy(j), ancora) for j in adversarios or ()]
    return {
        "jogadores_no_cone": sum(no_cone(j) for j in linha),
        "adversarios_no_cone": None
        if adversarios is None
        else sum(no_cone(j) for j in adversarios),
        "goleiro_no_cone": any(no_cone(j) for j in goleiros) if goleiros else None,
        "jogadores_na_area": sum(na_area(j) for j in linha),
        "adversarios_na_area": None
        if adversarios is None
        else sum(na_area(j) for j in adversarios),
        **{
            f"jogadores_a_{raio:g}_m": sum(math.dist(_xy(j), ancora) <= raio for j in linha)
            for raio in RAIOS_M
        },
        "adversario_mais_proximo_m": round(min(distancias), 3) if distancias else None,
        "distancia_ao_gol_m": round(math.dist(ancora, (gol_x, 0.0)), 3),
    }


def _xy(jogador: PlayerDetection) -> Ponto:
    assert jogador.pitch_xy is not None
    return jogador.pitch_xy


def _no_triangulo(p: Ponto, a: Ponto, b: Ponto, c: Ponto) -> bool:
    lados = (_lado(p, a, b), _lado(p, b, c), _lado(p, c, a))
    return not (any(v < 0 for v in lados) and any(v > 0 for v in lados))


def _lado(p: Ponto, a: Ponto, b: Ponto) -> float:
    return (p[0] - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (p[1] - b[1])
