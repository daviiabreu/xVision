"""A percepcao do quadro do contato, montada a partir do gabarito da fixture.

Ela nao passa pelo runner de proposito: o que estes testes afirmam e o que a feature faz com
uma percepcao, e o caminho do clipe ao evento e a prova ao vivo de `test_demo`.

Os numeros de confianca e de erro de reprojeccao vem dos stubs, e nao de constante escrita
aqui: assim o que o teste de unidade afirma e o que a prova ao vivo imprime sao o mesmo numero.

`contato_xy` troca so o metro que a homografia devolveu no quadro do contato, e nao a caixa em
pixel: e assim que homografia quase singular falha, com a deteccao boa e a projecao nao.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pytest

from contracts import BallDetection, Box, FieldGeometry, FramePerception, Perception
from edge.features.base import ShotOutput
from edge.modules.stubs.perception import CONFIDENCE, REPROJECTION_ERROR_PX
from edge.testing.pitch import FPS, SHOOTER_TRACK_ID, SHOT_FRAME_INDEX, ball_box, clip_truth

TRUTH = clip_truth()


@dataclass(frozen=True)
class Shot:
    frame_index: int = SHOT_FRAME_INDEX
    shooter_track_id: int | None = SHOOTER_TRACK_ID


def _perception(
    *,
    homography: tuple[float, ...] | None = TRUTH.homography,
    reprojection_error_px: float | None = REPROJECTION_ERROR_PX,
    ball: bool = True,
    contato_xy: tuple[float, float] | None = None,
) -> Perception:
    return Perception(
        clip_id="clipe-d4",
        frames=tuple(
            _frame(indice, homography, reprojection_error_px, ball, contato_xy)
            for indice in range(TRUTH.frame_count)
        ),
    )


@pytest.fixture
def perception() -> Perception:
    return _perception()


@pytest.fixture
def make_perception() -> Callable[..., Perception]:
    return _perception


@pytest.fixture
def shot() -> ShotOutput:
    return Shot()


@pytest.fixture
def make_shot() -> Callable[..., ShotOutput]:
    return Shot


def _frame(
    indice: int,
    homography: tuple[float, ...] | None,
    reprojection_error_px: float | None,
    ball: bool,
    contato_xy: tuple[float, float] | None,
) -> FramePerception:
    desenhada = TRUTH.ball_pitch_xy[indice]
    projetada = contato_xy if contato_xy is not None and indice == SHOT_FRAME_INDEX else desenhada
    return FramePerception(
        frame_index=indice,
        timestamp_s=indice / FPS,
        ball=BallDetection(
            box=Box(*ball_box(TRUTH, desenhada)),
            confidence=CONFIDENCE,
            pitch_xy=projetada if homography is not None else None,
        )
        if ball
        else None,
        field=FieldGeometry(homography=homography, reprojection_error_px=reprojection_error_px),
    )
