"""Movimento de camera entre dois quadros vizinhos, para levar a homografia de um ao outro."""

from __future__ import annotations

from typing import Any, cast

import cv2
import numpy as np

Array = np.ndarray[Any, Any]

MIN_INLIERS = 40
"""Correspondencias que o RANSAC precisa manter para o movimento valer. Preliminar."""

FAIXA_DO_PLACAR = 0.1
"""Fracao do alto do quadro ignorada, onde placar e logos ficam parados na tela."""

ESCALA_MAXIMA = 1.5
"""Mudanca de zoom entre vizinhos acima disso e corte, e nao movimento. Preliminar."""


def camera_motion(anterior: Array, atual: Array) -> Array | None:
    """Homografia de imagem que leva `anterior` em `atual`, ou `None` sem apoio suficiente."""
    orb = cast("Any", cv2).ORB_create(nfeatures=1500)
    altura, largura = int(anterior.shape[0]), int(anterior.shape[1])
    mascara = np.full((altura, largura), 255, dtype=np.uint8)
    mascara[: int(altura * FAIXA_DO_PLACAR)] = 0
    pontos_a, descritores_a = orb.detectAndCompute(_cinza(anterior), mascara)
    pontos_b, descritores_b = orb.detectAndCompute(_cinza(atual), mascara)
    if descritores_a is None or descritores_b is None:
        return None
    pares = (
        cast("Any", cv2)
        .BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        .match(descritores_a, descritores_b)
    )
    if len(pares) < MIN_INLIERS:
        return None
    origem = np.asarray([pontos_a[p.queryIdx].pt for p in pares], dtype=np.float32).reshape(
        -1, 1, 2
    )
    destino = np.asarray([pontos_b[p.trainIdx].pt for p in pares], dtype=np.float32).reshape(
        -1, 1, 2
    )
    matriz, aceitos = cast(
        "tuple[Array | None, Array | None]", cv2.findHomography(origem, destino, cv2.RANSAC, 3.0)
    )
    if matriz is None or aceitos is None or int(aceitos.sum()) < MIN_INLIERS:
        return None
    escala = abs(float(np.linalg.det(matriz[:2, :2])))
    if not 1 / ESCALA_MAXIMA**2 <= escala <= ESCALA_MAXIMA**2:
        return None
    return matriz


def _cinza(quadro: Array) -> Array:
    return quadro if quadro.ndim == 2 else cv2.cvtColor(quadro, cv2.COLOR_BGR2GRAY)
