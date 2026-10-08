"""Quadrante do chute na grade 16x9 da Footstats e o valor base de xG dele."""

from __future__ import annotations

import math

from contracts import FeatureValue, Perception
from edge.features.base import FeatureSpec, ShotOutput, unavailable
from edge.features.scene import SceneRefusal, shot_scene
from edge.modules.field.geometry import PITCH_105X68

# Imagem de meio campo da Footstats: 802 x 405 px, gol no topo, calibrada pela grande area.
CENTRO_DO_GOL_PX = 401.0
PX_POR_M_LATERAL = 802 / 68
PX_POR_M_PROFUNDIDADE = 141.3 / 16.5
COLUNA_PX = 50
LINHA_PX = 45

# Em centesimos, reconstruida do expected_goals gravado.
# Linha 1 junto ao gol, coluna 1 a esquerda de quem ataca.
CENTESIMOS: tuple[tuple[int, ...], ...] = (
    (2, 4, 4, 4, 10, 10, 18, 49, 49, 19, 8, 8, 4, 4, 4, 20),
    (2, 4, 4, 4, 7, 7, 12, 15, 15, 12, 7, 7, 4, 4, 4, 2),
    (2, 4, 4, 4, 7, 7, 9, 10, 10, 10, 7, 7, 4, 4, 4, 2),
    (2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2),
    (2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2),
    (1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 1, 1),
    (1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1),
    (1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1),
    (1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1),
)

# Sem finalizacao no rotulo (valor do espelho ou do vizinho), mais a L1 C16, que contradiz o doc.
INFERIDAS = frozenset(
    {
        *((1, c) for c in (1, 2, 3, 16)),
        *((2, c) for c in (1, 2, 15, 16)),
        *((3, c) for c in (1, 15, 16)),
        *((4, c) for c in (1, 2, 15, 16)),
        *((5, c) for c in (1, 14, 16)),
        *((6, c) for c in (1, 2, 4, 5, 15, 16)),
        *((7, c) for c in (1, 2, 3, 4, 6, 8, 10, 11, 12, 13, 16)),
        *((8, c) for c in (1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16)),
        *((9, c) for c in (1, 2, 3, 4, 7, 10, 14, 15, 16)),
    }
)


class Quadrant:
    spec = FeatureSpec(
        name="quadrant",
        version="0.1.0",
        requires=("players", "ball", "field", "tracking"),
        unit="xG",
    )

    def extract(self, perception: Perception, shot: ShotOutput) -> FeatureValue:
        cena = shot_scene(perception, shot)
        if isinstance(cena, SceneRefusal):
            return unavailable(self.spec, cena.detail)
        lateral, profundidade = para_o_gol(cena.anchor_xy)
        linha, coluna = celula_px(
            CENTRO_DO_GOL_PX + lateral * PX_POR_M_LATERAL, profundidade * PX_POR_M_PROFUNDIDADE
        )
        inferida = (linha, coluna) in INFERIDAS
        return FeatureValue(
            name=self.spec.name,
            version=self.spec.version,
            status="low_confidence" if inferida else "ok",
            value=valor(linha, coluna),
            confidence=None,
            reason="valor inferido: celula sem finalizacao no rotulo" if inferida else None,
            evidence={
                "linha": linha,
                "coluna": coluna,
                "lateral_m": round(lateral, 2),
                "profundidade_m": round(profundidade, 2),
                "quadro_de_referencia": cena.frame.frame_index,
                "defasagem_s": round(cena.offset_s, 4),
                "origem_do_finalizador": cena.shooter_source,
            },
        )


def para_o_gol(xy: tuple[float, float]) -> tuple[float, float]:
    """Lateral (positiva a direita de quem ataca) e profundidade, em metro, no gol mais proximo."""
    x, y = xy
    meio = PITCH_105X68.length_m / 2.0
    # O template tem y crescendo para baixo, entao atacando o gol direito a direita e +y.
    if x >= 0:
        return y, meio - x
    return -y, meio + x


def celula_px(x: float, y: float) -> tuple[int, int]:
    """Linha e coluna de um ponto da imagem da Footstats, presas a grade."""
    coluna = min(len(CENTESIMOS[0]), max(1, math.ceil(int(x) / COLUNA_PX)))
    linha = min(len(CENTESIMOS), max(1, math.ceil(int(y) / LINHA_PX)))
    return linha, coluna


def valor(linha: int, coluna: int) -> float:
    return CENTESIMOS[linha - 1][coluna - 1] / 100
