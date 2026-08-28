"""A folha de contato recortada na regiao do lance, que e a que da para anotar.

A folha de quadro inteiro nao serve e o motivo e aritmetico: um clipe de 1920 por 1080 numa celula
de 320 por 180 poe o jogador em cerca de 17 pixels de altura, e o contato entre pe e bola nao e
visivel nesse tamanho. Medido tentando anotar: da para dizer em que segundo o lance acontece, e nao
em que quadro.

O recorte segue os jogadores, e nao a bola. A bola e justamente o que esta em duvida: centralizar
nela mostraria a arquibancada quando a deteccao erra, que e o caso que a anotacao existe para
pegar. O envelope das caixas de `players` e estavel porque a deteccao de gente funciona, medida em
13 a 23 jogadores por quadro nos clipes do parceiro.

Nao decide o instante e nao anota nada. Produz a imagem que uma pessoa olha.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from contracts import Box

MARGEM = 0.35
"""Quanto o envelope cresce, em fracao do proprio lado, para a bola caber quando ela sai do pe."""

MINIMO_PX = 420.0
"""Piso do lado do recorte no quadro original, para um lance com dois jogadores nao virar macro."""


def envelope(registros: Sequence[Mapping[str, object]]) -> Box | None:
    """A caixa que contem todos os jogadores do quadro, ou `None` quando nao ha nenhum."""
    caixas = [cast("Any", r["box"]) for r in registros if "box" in r]
    if not caixas:
        return None
    return Box(
        min(float(c[0]) for c in caixas),
        min(float(c[1]) for c in caixas),
        max(float(c[2]) for c in caixas),
        max(float(c[3]) for c in caixas),
    )


def janela(alvo: Box, largura: int, altura: int) -> tuple[int, int, int, int]:
    """Um recorte quadrado o bastante em volta do alvo, preso dentro do quadro.

    Preso e nao deslizante nas bordas: um lance no canto do campo produz recorte menor, e mentir
    sobre a posicao para manter o tamanho moveria o lance de lugar entre uma celula e a proxima.
    """
    centro_x = (alvo.x1 + alvo.x2) / 2.0
    centro_y = (alvo.y1 + alvo.y2) / 2.0
    lado_x = max(MINIMO_PX, (alvo.x2 - alvo.x1) * (1.0 + 2 * MARGEM))
    lado_y = max(MINIMO_PX * altura / largura, (alvo.y2 - alvo.y1) * (1.0 + 2 * MARGEM))
    proporcao = largura / altura
    if lado_x / lado_y < proporcao:
        lado_x = lado_y * proporcao
    else:
        lado_y = lado_x / proporcao
    x1 = max(0, int(centro_x - lado_x / 2))
    y1 = max(0, int(centro_y - lado_y / 2))
    x2 = min(largura, int(centro_x + lado_x / 2))
    y2 = min(altura, int(centro_y + lado_y / 2))
    return x1, y1, x2, y2
