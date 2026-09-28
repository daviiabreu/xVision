"""Fixture de video sintetica, desenhada a partir do gabarito de `edge.testing.pitch`.

Nenhum clipe da Footstats entra no repositorio, entao a fixture e gerada. Ela e
deterministica, e a posicao de cada elemento vem de metros de campo projetados pela mesma
homografia que o modulo `field` devolve. Por isso ela e gabarito e nao so entrada: quem
afirmar distancia, angulo ou erro de reprojeccao compara com um numero conhecido.

A bola sai numa cor que nada mais no quadro usa. Isso nao e enfeite: e o que permite afirmar,
pixel a pixel, que o desenho concorda com o gabarito, sem depender de deteccao.

A semente muda so o tom do gramado. Ela existe para dois clipes diferentes terem bytes
diferentes sem mexer na geometria, que e o que os testes de identidade de clipe usam.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from edge.testing.pitch import (
    BALL_RADIUS_PX,
    GOAL_AREA_HALF_WIDTH_M,
    GOAL_AREA_X_M,
    GOAL_HALF_WIDTH_M,
    GOAL_LINE_X_M,
    PENALTY_BOX_HALF_WIDTH_M,
    PENALTY_BOX_X_M,
    PENALTY_SPOT_X_M,
    ClipTruth,
    clip_truth,
    player_box,
    project,
)

BALL_COLOR = (255, 0, 255)
LINE_COLOR = (240, 240, 240)
TEAM_COLOR = {"team_a": (40, 40, 220), "team_b": (220, 60, 40)}
GOALKEEPER_COLOR = (40, 220, 220)

_PITCH_LINES: tuple[tuple[tuple[float, float], ...], ...] = (
    ((GOAL_LINE_X_M, -PENALTY_BOX_HALF_WIDTH_M), (GOAL_LINE_X_M, PENALTY_BOX_HALF_WIDTH_M)),
    (
        (PENALTY_BOX_X_M, -PENALTY_BOX_HALF_WIDTH_M),
        (PENALTY_BOX_X_M, PENALTY_BOX_HALF_WIDTH_M),
        (GOAL_LINE_X_M, PENALTY_BOX_HALF_WIDTH_M),
    ),
    ((PENALTY_BOX_X_M, -PENALTY_BOX_HALF_WIDTH_M), (GOAL_LINE_X_M, -PENALTY_BOX_HALF_WIDTH_M)),
    ((GOAL_AREA_X_M, -GOAL_AREA_HALF_WIDTH_M), (GOAL_AREA_X_M, GOAL_AREA_HALF_WIDTH_M)),
    ((GOAL_LINE_X_M, -GOAL_HALF_WIDTH_M), (GOAL_LINE_X_M, GOAL_HALF_WIDTH_M)),
)


def ball_pixel(frame_index: int) -> tuple[int, int]:
    gabarito = clip_truth()
    return _pixel(gabarito, gabarito.ball_pitch_xy[frame_index])


def render_frame(frame_index: int, seed: int = 0) -> np.ndarray[Any, Any]:
    gabarito = clip_truth()
    quadro = np.empty((gabarito.height, gabarito.width, 3), dtype=np.uint8)
    quadro[:, :] = (20, 90 + seed % 40, 20)
    _draw_pitch(quadro, gabarito)

    for jogador in gabarito.players:
        cor = GOALKEEPER_COLOR if jogador.role == "goalkeeper" else TEAM_COLOR[jogador.team]
        _draw_player(quadro, player_box(gabarito, jogador.pitch_xy[frame_index]), cor)
    cv2.circle(quadro, ball_pixel(frame_index), round(BALL_RADIUS_PX), BALL_COLOR, -1)
    return quadro


def make_synthetic_clip(path: str | Path, *, seed: int = 0) -> Path:
    gabarito = clip_truth()
    destino = Path(path)
    destino.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(destino),
        cv2.VideoWriter.fourcc(*"mp4v"),
        gabarito.fps,
        (gabarito.width, gabarito.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"nao foi possivel abrir o escritor de video em {destino}")
    try:
        for indice in range(gabarito.frame_count):
            writer.write(render_frame(indice, seed))
    finally:
        writer.release()
    return destino


def _draw_pitch(quadro: np.ndarray[Any, Any], gabarito: ClipTruth) -> None:
    for linha in _PITCH_LINES:
        pontos = np.array([_pixel(gabarito, ponto) for ponto in linha], dtype=np.int32)
        cv2.polylines(quadro, [pontos], False, LINE_COLOR, 1)
    cv2.circle(quadro, _pixel(gabarito, (PENALTY_SPOT_X_M, 0.0)), 1, LINE_COLOR, -1)


def _draw_player(
    quadro: np.ndarray[Any, Any],
    caixa: tuple[float, float, float, float],
    cor: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = caixa
    cv2.rectangle(quadro, (round(x1), round(y1)), (round(x2), round(y2)), cor, -1)


def _pixel(gabarito: ClipTruth, posicao: tuple[float, float]) -> tuple[int, int]:
    u, v = project(gabarito.homography, posicao[0], posicao[1])
    return (round(u), round(v))
