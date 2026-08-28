"""O rastreador da bola: selecao por movimento, rejeicao de salto e tratamento de lacuna.

Testado com deteccoes sinteticas, sem YOLO nem GPU: o valor do rastreador esta na logica de
trajetoria, e e ela que estes testes fixam. O detector fica atras de uma interface justamente para
que este nucleo seja exercitavel sozinho.
"""

from __future__ import annotations

from contracts.perception import BallDetection, Box
from edge.ball.tracking import BallObservation, BallTracker, TrackerConfig


def _ball(center: tuple[float, float], conf: float) -> BallDetection:
    cx, cy = center
    return BallDetection(box=Box(cx - 5.0, cy - 5.0, cx + 5.0, cy + 5.0), confidence=conf)


def _obs(frame: int, center: tuple[float, float] | None, *, conf: float = 0.9) -> BallObservation:
    timestamp_s = frame / 30.0
    candidates = () if center is None else (_ball(center, conf),)
    return BallObservation(frame, timestamp_s, candidates)


def test_constant_motion_keeps_every_detection() -> None:
    observations = [_obs(i, (float(i * 10), 100.0)) for i in range(5)]

    points = list(BallTracker().track(observations))

    assert [point.source for point in points] == ["detected"] * 5
    assert points[2].center == (20.0, 100.0)


def test_short_gap_is_interpolated() -> None:
    observations = [_obs(0, (0.0, 0.0)), _obs(1, None), _obs(2, (20.0, 0.0))]

    points = list(BallTracker().track(observations))

    assert [point.source for point in points] == ["detected", "interpolated", "detected"]
    assert points[1].center == (10.0, 0.0)


def test_improbable_jump_is_rejected() -> None:
    # A deteccao do quadro 1 salta muito alem do permitido pelo gate: vira lacuna e a posicao
    # e reconstruida por interpolacao, em vez de aceitar o salto como real.
    observations = [_obs(0, (0.0, 0.0)), _obs(1, (5000.0, 0.0)), _obs(2, (20.0, 0.0))]

    points = list(BallTracker().track(observations))

    assert points[1].source == "interpolated"
    assert points[1].center == (10.0, 0.0)


def test_long_gap_is_marked_missing() -> None:
    config = TrackerConfig(max_gap_frames=2)
    observations = [
        _obs(0, (0.0, 0.0)),
        _obs(1, None),
        _obs(2, None),
        _obs(3, None),
        _obs(10, (100.0, 0.0)),
    ]

    points = list(BallTracker(config).track(observations))
    sources = [point.source for point in points]

    assert sources[0] == "detected"
    assert sources.count("missing") == 3
    assert sources[-1] == "detected"


def test_moving_candidate_beats_static_high_confidence_one() -> None:
    # A bola vem de (0,0) para (10,0), entao a posicao prevista no quadro 2 e (20,0). Um candidato
    # parado perto da posicao antiga tem confianca maior, mas fica fora da trajetoria: a marca do
    # penalti perde para a bola em movimento.
    frame0 = BallObservation(0, 0.0, (_ball((0.0, 0.0), 0.9),))
    frame1 = BallObservation(1, 1 / 30.0, (_ball((10.0, 0.0), 0.9),))
    frame2 = BallObservation(2, 2 / 30.0, (_ball((20.0, 0.0), 0.5), _ball((2.0, 0.0), 0.95)))

    tracker = BallTracker()
    tracker.update(frame0)
    tracker.update(frame1)
    emitted = tracker.update(frame2)

    assert emitted[-1].center == (20.0, 0.0)


def test_stream_flush_marks_trailing_gap_as_missing() -> None:
    tracker = BallTracker()

    emitted = tracker.update(_obs(0, (0.0, 0.0)))
    emitted += tracker.update(_obs(1, None))
    assert [point.source for point in emitted] == ["detected"]

    tail = tracker.flush()
    assert [point.source for point in tail] == ["missing"]
