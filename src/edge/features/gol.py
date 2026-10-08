"""A geometria do gol que as features compartilham, e o alvo escolhido pelo rumo da bola.

`gol_mais_proximo` estava copiado em `goalkeeper_position` e em `looked_before`. Duas copias da
mesma conta divergem caladas: bastaria uma trocar o template do campo para as duas features
passarem a medir contra balizas diferentes sem nenhuma excecao.

`gol_pelo_rumo_da_bola` e o criterio novo. A distancia nao basta para escolher o alvo: chute de
fora da area pode partir de um ponto mais perto da baliza que o proprio time defende, e ali o
criterio da distancia aponta para o gol errado. Quem sabe para onde o chute foi e a bola depois do
contato, e isso e medida em vez de suposicao.
"""

from __future__ import annotations

import math

from contracts import Perception
from edge.modules.field.geometry import PITCH_105X68

GOL_MEIA_LARGURA_M = 3.66
"""Meia largura da baliza oficial, 7,32 m de trave a trave. `occupied_area` usa o mesmo numero."""

JANELA_DEPOIS_DO_CONTATO_S = 1.0
"""Ate onde olhar a bola depois do contato. Alem disso ela pode ter batido em alguem e mudado de
rumo, e o rumo que diz qual era o alvo do chute e o imediatamente apos o pe."""

MIN_QUADROS_DA_BOLA = 2
MIN_DESLOCAMENTO_DA_BOLA_M = 1.0
"""Abaixo disso o deslocamento nao distingue rumo de ruido de deteccao: a bola tem 22 cm e a
projecao do seu centro erra mais que isso em campo distante."""

CRITERIO_PELA_BOLA = "rumo da bola depois do contato"
CRITERIO_PELA_DISTANCIA = "gol mais proximo do finalizador"


def gols() -> tuple[tuple[float, float], tuple[float, float]]:
    """Os centros das duas balizas no template do campo, em metros."""
    meio = PITCH_105X68.length_m / 2.0
    return ((-meio, 0.0), (meio, 0.0))


def gol_mais_proximo(posicao: tuple[float, float]) -> tuple[float, float]:
    return min(gols(), key=lambda gol: math.dist(posicao, gol))


def nome_do_gol(gol: tuple[float, float]) -> str:
    return "direito" if gol[0] > 0 else "esquerdo"


def gol_pelo_rumo_da_bola(
    perception: Perception, frame_do_contato: int, timestamp_do_contato: float
) -> tuple[float, float] | None:
    """O gol que a bola procurou depois do contato, ou `None` quando a trajetoria nao sustenta.

    So posicao `detected` entra. A interpolacao preenche lacuna com reta, e o contato e exatamente
    onde a trajetoria vira: uma reta tracada por cima da virada aponta para onde a bola nao foi.
    """
    amostras = tuple(
        quadro.ball.pitch_xy
        for quadro in perception.frames
        if quadro.frame_index >= frame_do_contato
        and quadro.timestamp_s <= timestamp_do_contato + JANELA_DEPOIS_DO_CONTATO_S
        and quadro.ball is not None
        and quadro.ball.pitch_xy is not None
        and quadro.ball.source == "detected"
    )
    if len(amostras) < MIN_QUADROS_DA_BOLA:
        return None
    origem, destino = amostras[0], amostras[-1]
    deslocamento = (destino[0] - origem[0], destino[1] - origem[1])
    if math.hypot(*deslocamento) < MIN_DESLOCAMENTO_DA_BOLA_M:
        return None

    def alinhamento(gol: tuple[float, float]) -> float:
        para_gol = (gol[0] - origem[0], gol[1] - origem[1])
        norma = math.hypot(*para_gol) * math.hypot(*deslocamento)
        if norma < 1e-9:
            return -1.0
        return (deslocamento[0] * para_gol[0] + deslocamento[1] * para_gol[1]) / norma

    melhor = max(gols(), key=alinhamento)
    return melhor if alinhamento(melhor) > 0.0 else None
