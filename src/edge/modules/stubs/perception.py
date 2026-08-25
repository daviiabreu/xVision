"""Um stub por especie, os sete, respondendo a partir do gabarito da fixture.

Os nomes de campo de cada registro espelham `contracts.perception`, mas os tipos nao: a caixa
sai como tupla de quatro floats, e nao como `Box`, porque o lote inteiro e serializado em JSON
pelo cache de percepcao e dataclass nao atravessa. Quem monta o `Box` e o runner.

Caixa e keypoints saem sempre em pixel do quadro original, nunca do lote pre-processado que o
modulo declarou, mesmo quando a `PreprocessSpec` pede outra resolucao.

Cada especie devolve so o que e dela: `players` da caixa, time, funcao e confianca,
e o identificador de trajetoria e de `tracking`, que o liga a deteccao por `detection_index`, a
posicao do registro dentro do quadro. O runner nunca preenche um a partir do outro.
"""

from __future__ import annotations

from edge.modules.base import PreprocessSpec
from edge.modules.stubs.base import Record, StubModule, stub_spec
from edge.testing.pitch import (
    GOAL_HALF_WIDTH_M,
    GOAL_LINE_X_M,
    PENALTY_BOX_HALF_WIDTH_M,
    PENALTY_BOX_X_M,
    ClipTruth,
    PlayerTruth,
    ball_box,
    player_box,
    project,
)

COARSE = PreprocessSpec(width=320, height=240, color="rgb", letterbox=False, normalize="none")
FINE = PreprocessSpec(width=192, height=192, color="rgb", letterbox=True, normalize="unit")

CONFIDENCE = 0.9
REPROJECTION_ERROR_PX = 1.2
KEYPOINTS = (
    ("goal_post_left", GOAL_LINE_X_M, -GOAL_HALF_WIDTH_M),
    ("goal_post_right", GOAL_LINE_X_M, GOAL_HALF_WIDTH_M),
    ("penalty_box_left", PENALTY_BOX_X_M, -PENALTY_BOX_HALF_WIDTH_M),
    ("penalty_box_right", PENALTY_BOX_X_M, PENALTY_BOX_HALF_WIDTH_M),
)


class StubPlayers(StubModule):
    spec = stub_spec("players", COARSE)

    def records(self, truth: ClipTruth, frame_index: int) -> tuple[Record, ...]:
        return tuple(_player_record(truth, jogador, frame_index) for jogador in truth.players)


class StubBall(StubModule):
    spec = stub_spec("ball", COARSE)

    def records(self, truth: ClipTruth, frame_index: int) -> tuple[Record, ...]:
        pitch_xy = truth.ball_pitch_xy[frame_index]
        return (
            {
                "box": ball_box(truth, pitch_xy),
                "confidence": CONFIDENCE,
                "pitch_xy": pitch_xy,
            },
        )


class StubField(StubModule):
    spec = stub_spec("field", COARSE)

    def records(self, truth: ClipTruth, frame_index: int) -> tuple[Record, ...]:
        return (
            {
                "homography": truth.homography,
                "reprojection_error_px": REPROJECTION_ERROR_PX,
                "keypoints": tuple(
                    (nome, *project(truth.homography, x, y)) for nome, x, y in KEYPOINTS
                ),
            },
        )


class StubTracking(StubModule):
    spec = stub_spec("tracking", COARSE)

    def records(self, truth: ClipTruth, frame_index: int) -> tuple[Record, ...]:
        return tuple(
            {
                "detection_index": posicao,
                "track_id": jogador.track_id,
                "box": _player_box(truth, jogador, frame_index),
            }
            for posicao, jogador in enumerate(truth.players)
        )


class StubPose(StubModule):
    spec = stub_spec("pose", FINE, frames="window")

    def records(self, truth: ClipTruth, frame_index: int) -> tuple[Record, ...]:
        return tuple(
            {
                "track_id": jogador.track_id,
                "keypoints": _keypoints(truth, jogador, frame_index),
                "confidence": CONFIDENCE,
            }
            for jogador in truth.players
        )


class StubIdentity(StubModule):
    spec = stub_spec("identity", FINE, frames="window")

    def records(self, truth: ClipTruth, frame_index: int) -> tuple[Record, ...]:
        return tuple(
            {
                "track_id": jogador.track_id,
                "jersey_number": jogador.jersey_number,
                "confidence": CONFIDENCE,
            }
            for jogador in truth.players
        )


class StubShot(StubModule):
    spec = stub_spec("shot", COARSE)

    def records(self, truth: ClipTruth, frame_index: int) -> tuple[Record, ...]:
        if frame_index != truth.shot_frame_index:
            return ()
        return (
            {
                "frame_index": truth.shot_frame_index,
                "shooter_track_id": truth.shooter_track_id,
                "confidence": CONFIDENCE,
            },
        )


STUBS: tuple[type[StubModule], ...] = (
    StubPlayers,
    StubBall,
    StubField,
    StubTracking,
    StubPose,
    StubIdentity,
    StubShot,
)


def _player_record(truth: ClipTruth, jogador: PlayerTruth, frame_index: int) -> Record:
    return {
        "box": _player_box(truth, jogador, frame_index),
        "confidence": CONFIDENCE,
        "team": jogador.team,
        "role": jogador.role,
        "pitch_xy": jogador.pitch_xy[frame_index],
    }


def _player_box(
    truth: ClipTruth, jogador: PlayerTruth, frame_index: int
) -> tuple[float, float, float, float]:
    return player_box(truth, jogador.pitch_xy[frame_index])


def _keypoints(
    truth: ClipTruth, jogador: PlayerTruth, frame_index: int
) -> tuple[tuple[str, float, float], ...]:
    x1, y1, x2, y2 = _player_box(truth, jogador, frame_index)
    meio = (x1 + x2) / 2
    return (("head", meio, y1), ("hip", meio, (y1 + y2) / 2), ("foot", meio, y2))
