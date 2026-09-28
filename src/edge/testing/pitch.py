"""Gabarito da fixture: onde cada elemento esta, em metros, por construcao.

O clipe e desenhado a partir daqui, e nao o contrario. Isso e o que torna a fixture
gabarito de verdade: quem afirma distancia, angulo ou erro de reprojeccao compara com um
numero que ninguem mediu na imagem, e o modulo `field` devolve exatamente a homografia que
desenhou o quadro.

A origem e o centro do gramado, o que deixa as duas metades simetricas, e o eixo x aponta para
o gol atacado, que fica em x positivo. Nada aqui depende de aleatoriedade.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import cv2
import numpy as np

from edge.participants.identity import Team

WIDTH = 320
HEIGHT = 240
FPS = 10
SECONDS = 3

GOAL_LINE_X_M = 52.5
GOAL_HALF_WIDTH_M = 3.66
PENALTY_BOX_X_M = 36.0
PENALTY_BOX_HALF_WIDTH_M = 20.16
GOAL_AREA_X_M = 47.0
GOAL_AREA_HALF_WIDTH_M = 9.16
PENALTY_SPOT_X_M = 41.5

SHOT_FRAME_INDEX = 18
SHOOTER_TRACK_ID = 7

BALL_RADIUS_PX = 3.0
PLAYER_HEIGHT_PX = (10.0, 18.0)
PLAYER_WIDTH_RATIO = 0.22

_VISIBLE_PITCH = ((30.0, -22.0), (30.0, 22.0), (55.0, 22.0), (55.0, -22.0))
_VISIBLE_IMAGE = ((90.0, 40.0), (230.0, 40.0), (315.0, 225.0), (5.0, 225.0))

_DRIBBLE_FROM = (36.0, -9.0)
_CONTACT_AT = (PENALTY_SPOT_X_M, -6.0)


@dataclass(frozen=True)
class PlayerTruth:
    track_id: int
    team: Team
    role: str
    jersey_number: int
    pitch_xy: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class ClipTruth:
    width: int
    height: int
    fps: int
    frame_count: int
    homography: tuple[float, ...]
    shot_frame_index: int
    shooter_track_id: int
    ball_pitch_xy: tuple[tuple[float, float], ...]
    players: tuple[PlayerTruth, ...]


def project(homography: tuple[float, ...], x: float, y: float) -> tuple[float, float]:
    a, b, c, d, e, f, g, h, i = homography
    w = g * x + h * y + i
    return ((a * x + b * y + c) / w, (d * x + e * y + f) / w)


def player_box(
    truth: ClipTruth, pitch_xy: tuple[float, float]
) -> tuple[float, float, float, float]:
    u, v = project(truth.homography, pitch_xy[0], pitch_xy[1])
    base, ganho = PLAYER_HEIGHT_PX
    altura = base + ganho * (v / truth.height)
    meia_largura = max(2.0, altura * PLAYER_WIDTH_RATIO)
    return (u - meia_largura, v - altura, u + meia_largura, v)


def ball_box(truth: ClipTruth, pitch_xy: tuple[float, float]) -> tuple[float, float, float, float]:
    u, v = project(truth.homography, pitch_xy[0], pitch_xy[1])
    return (u - BALL_RADIUS_PX, v - BALL_RADIUS_PX, u + BALL_RADIUS_PX, v + BALL_RADIUS_PX)


def homography_matrix(homography: tuple[float, ...]) -> np.ndarray[Any, Any]:
    return np.array([homography[0:3], homography[3:6], homography[6:9]], dtype=np.float64)


@lru_cache(maxsize=1)
def clip_truth() -> ClipTruth:
    total = SECONDS * FPS
    return ClipTruth(
        width=WIDTH,
        height=HEIGHT,
        fps=FPS,
        frame_count=total,
        homography=_homography(),
        shot_frame_index=SHOT_FRAME_INDEX,
        shooter_track_id=SHOOTER_TRACK_ID,
        ball_pitch_xy=tuple(_ball_at(indice) for indice in range(total)),
        players=_players(total),
    )


def _homography() -> tuple[float, ...]:
    origem = np.array(_VISIBLE_PITCH, dtype=np.float32)
    destino = np.array(_VISIBLE_IMAGE, dtype=np.float32)
    matriz = cv2.getPerspectiveTransform(origem, destino)
    return tuple(
        round(float(matriz[linha][coluna]), 9) for linha in range(3) for coluna in range(3)
    )


def _ball_at(frame_index: int) -> tuple[float, float]:
    if frame_index <= SHOT_FRAME_INDEX:
        return _between(_DRIBBLE_FROM, _CONTACT_AT, frame_index / SHOT_FRAME_INDEX)
    restantes = SECONDS * FPS - 1 - SHOT_FRAME_INDEX
    avanco = (frame_index - SHOT_FRAME_INDEX) / restantes
    return _between(_CONTACT_AT, (GOAL_LINE_X_M, 0.0), avanco)


def _players(total: int) -> tuple[PlayerTruth, ...]:
    finalizador = PlayerTruth(
        track_id=SHOOTER_TRACK_ID,
        team="team_a",
        role="player",
        jersey_number=10,
        pitch_xy=tuple(_ball_at(min(indice, SHOT_FRAME_INDEX)) for indice in range(total)),
    )
    marcador = PlayerTruth(
        track_id=4,
        team="team_b",
        role="player",
        jersey_number=3,
        pitch_xy=tuple(
            _between((38.0, -12.0), (44.0, -7.5), indice / (total - 1)) for indice in range(total)
        ),
    )
    goleiro = PlayerTruth(
        track_id=1,
        team="team_b",
        role="goalkeeper",
        jersey_number=1,
        pitch_xy=tuple(
            _between((51.0, -1.5), (51.0, 1.0), indice / (total - 1)) for indice in range(total)
        ),
    )
    posicoes_paradas: tuple[tuple[int, Team, int, tuple[float, float]], ...] = (
        (2, "team_a", 9, (44.0, 6.0)),
        (3, "team_b", 5, (46.5, -2.0)),
        (5, "team_a", 7, (37.0, 12.0)),
    )
    parados = tuple(
        PlayerTruth(
            track_id=identificador,
            team=time,
            role="player",
            jersey_number=numero,
            pitch_xy=((posicao,) * total),
        )
        for identificador, time, numero, posicao in posicoes_paradas
    )
    return (finalizador, marcador, goleiro, *parados)


def _between(
    origem: tuple[float, float], destino: tuple[float, float], avanco: float
) -> tuple[float, float]:
    return (
        round(origem[0] + (destino[0] - origem[0]) * avanco, 6),
        round(origem[1] + (destino[1] - origem[1]) * avanco, 6),
    )
