"""O filtro de forma e tamanho da bola: aceita caixa quadrada de tamanho plausivel, rejeita o resto.

Puro e deterministico, sem YOLO nem GPU. Nao pretende separar marca de penalti de bola parada (as
duas passam no filtro); isso e trabalho da continuidade de movimento no rastreador.
"""

from __future__ import annotations

from contracts.perception import Box
from edge.ball.tracking import DetectorConfig, is_ball_shaped

_CONFIG = DetectorConfig(weights="fake.pt")
_FRAME_H = 720.0


def _square(side: float) -> Box:
    return Box(100.0, 100.0, 100.0 + side, 100.0 + side)


def test_accepts_plausible_ball() -> None:
    assert is_ball_shaped(_square(14.0), _FRAME_H, _CONFIG)


def test_rejects_too_large() -> None:
    assert not is_ball_shaped(_square(200.0), _FRAME_H, _CONFIG)


def test_rejects_too_small() -> None:
    assert not is_ball_shaped(_square(1.0), _FRAME_H, _CONFIG)


def test_rejects_elongated_box() -> None:
    # Faixa fina e comprida: aspecto muito longe de quadrado, tipico de artefato, nao de bola.
    assert not is_ball_shaped(Box(100.0, 100.0, 160.0, 108.0), _FRAME_H, _CONFIG)


def test_rejects_degenerate_box() -> None:
    assert not is_ball_shaped(Box(100.0, 100.0, 100.0, 100.0), _FRAME_H, _CONFIG)
