"""O rastreador da bola: selecao por movimento, rejeicao de salto e tratamento de lacuna.

Testado com deteccoes sinteticas, sem YOLO nem GPU: o valor do rastreador esta na logica de
trajetoria, e e ela que estes testes fixam. O detector fica atras de uma interface justamente para
que este nucleo seja exercitavel sozinho.
"""

from __future__ import annotations

import pytest

from contracts import BallDetection, Box
from edge.modules.ball import (
    BallObservation,
    BallTracker,
    TrackerConfig,
    TrajectoryOrderError,
)


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
    assert [point.trajectory_segment for point in points] == [0, 0, 0]
    assert [point.gap_frames for point in points] == [0, 1, 0]
    assert points[0].velocity_xy_px_s is None
    assert points[1].velocity_xy_px_s == pytest.approx((300.0, 0.0))
    assert points[1].velocity_source == "trajectory-model"
    assert points[2].velocity_xy_px_s == pytest.approx((300.0, 0.0))
    assert points[2].velocity_source == "finite-difference"


def test_improbable_jump_is_rejected() -> None:
    # A deteccao do quadro 1 salta muito alem do permitido pelo gate: vira lacuna e a posicao
    # e reconstruida por interpolacao, em vez de aceitar o salto como real.
    observations = [_obs(0, (0.0, 0.0)), _obs(1, (5000.0, 0.0)), _obs(2, (20.0, 0.0))]

    points = list(BallTracker().track(observations))

    assert points[1].source == "interpolated"
    assert points[1].center == (10.0, 0.0)


def test_leading_missing_frames_are_not_counted_as_interpolated() -> None:
    """Uma lacuna no inicio nunca sera fechada por interpolacao, entao contar aqueles quadros como
    interpolados inflava a telemetria e escondia perda de posicao dentro de um numero de sucesso."""
    tracker = BallTracker()

    points = list(tracker.track([_obs(0, None), _obs(1, (100, 100))]))

    assert [point.source for point in points] == ["missing", "detected"]
    assert tracker.telemetry.interpolated_points == 0
    assert tracker.telemetry.missing_points == 1


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
    assert points[0].trajectory_segment == 0
    assert points[-1].trajectory_segment == 1
    assert points[-1].velocity_xy_px_s is None
    assert points[-1].velocity_source is None


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


@pytest.mark.parametrize(
    "observations",
    [
        (BallObservation(0, 0.0, ()), BallObservation(0, 0.1, ())),
        (BallObservation(0, 0.0, ()), BallObservation(1, 0.0, ())),
        (BallObservation(1, 1.0, ()), BallObservation(2, 0.5, ())),
    ],
)
def test_frame_index_and_timestamp_must_be_strictly_increasing(
    observations: tuple[BallObservation, BallObservation],
) -> None:
    tracker = BallTracker()
    tracker.update(observations[0])

    with pytest.raises(TrajectoryOrderError):
        tracker.update(observations[1])


def test_camera_cut_closes_pending_gap_and_starts_a_new_segment() -> None:
    tracker = BallTracker()
    emitted = tracker.update(_obs(0, (0.0, 0.0)))
    emitted += tracker.update(_obs(1, None))
    emitted += tracker.update(
        BallObservation(2, 2 / 30.0, (_ball((20.0, 0.0), 0.9),), camera_cut=True)
    )

    assert [point.source for point in emitted] == ["detected", "missing", "detected"]
    assert emitted[0].trajectory_segment == 0
    assert emitted[-1].trajectory_segment == 1
    assert emitted[-1].velocity_xy_px_s is None
    assert tracker.telemetry.camera_cuts == 1
    assert tracker.telemetry.resets == 1


def test_telemetry_counts_rejected_jump_origins_and_largest_gap() -> None:
    tracker = BallTracker()
    points = list(
        tracker.track(
            [
                _obs(0, (0.0, 0.0)),
                _obs(1, (5000.0, 0.0)),
                _obs(2, (20.0, 0.0)),
            ]
        )
    )

    assert [point.source for point in points] == ["detected", "interpolated", "detected"]
    assert tracker.telemetry.detected_points == 2
    assert tracker.telemetry.interpolated_points == 1
    assert tracker.telemetry.missing_points == 0
    assert tracker.telemetry.rejected_jumps == 1
    assert tracker.telemetry.max_gap_frames == 1
    assert tracker.telemetry.segments == 1


def test_candidate_order_aceita_o_primeiro_que_passa_no_gate() -> None:
    """Com `candidate_order`, quem manda e a ordem da lista, e nao a proximidade da previsao.

    O segundo quadro traz um candidato longe em primeiro lugar e um perto em segundo. A regra
    padrao escolheria o perto; a nova escolhe o primeiro que sobrevive ao gate de salto.
    """
    perto, longe = _ball((110.0, 100.0), 0.9), _ball((160.0, 100.0), 0.5)
    observacoes = [
        BallObservation(0, 0.0, (_ball((100.0, 100.0), 0.9),)),
        BallObservation(1, 1 / 30, (longe, perto)),
    ]

    padrao = list(BallTracker().track(observacoes))
    ordenado = list(BallTracker(TrackerConfig(candidate_order=True)).track(observacoes))

    assert padrao[1].center == (110.0, 100.0)
    assert ordenado[1].center == (160.0, 100.0)


def test_candidate_order_ainda_recusa_o_que_o_gate_recusa() -> None:
    """A ordem escolhe entre os que passam, ela nao passa por cima do gate."""
    impossivel = _ball((5000.0, 100.0), 0.99)
    plausivel = _ball((110.0, 100.0), 0.1)
    observacoes = [
        BallObservation(0, 0.0, (_ball((100.0, 100.0), 0.9),)),
        BallObservation(1, 1 / 30, (impossivel, plausivel)),
    ]

    pontos = list(BallTracker(TrackerConfig(candidate_order=True)).track(observacoes))

    assert pontos[1].center == (110.0, 100.0)


def test_candidate_order_no_primeiro_quadro_usa_a_ordem_e_nao_a_confianca() -> None:
    primeiro, mais_confiante = _ball((100.0, 100.0), 0.2), _ball((300.0, 100.0), 0.9)
    observacoes = [BallObservation(0, 0.0, (primeiro, mais_confiante))]

    padrao = list(BallTracker().track(observacoes))
    ordenado = list(BallTracker(TrackerConfig(candidate_order=True)).track(observacoes))

    assert padrao[0].center == (300.0, 100.0)
    assert ordenado[0].center == (100.0, 100.0)


def test_candidate_order_com_um_candidato_nao_muda_nada() -> None:
    """A amarra que permite ligar `candidate_order` junto com o ranking sem remedir o caminho."""
    observacoes = [
        BallObservation(i, i / 30, (_ball((100.0 + 10 * i, 100.0), 0.9),)) for i in range(4)
    ]

    padrao = [p.center for p in BallTracker().track(observacoes)]
    ordenado = [
        p.center for p in BallTracker(TrackerConfig(candidate_order=True)).track(observacoes)
    ]

    assert padrao == ordenado


def _obs_multi(frame: int, *candidatos: tuple[tuple[float, float], float]) -> BallObservation:
    return BallObservation(
        frame, frame / 30.0, tuple(_ball(centro, conf) for centro, conf in candidatos)
    )


def _perde_a_trajetoria(tracker: BallTracker, ate: int) -> None:
    """Estabelece a trajetoria perto de (100, 100) e abre lacuna longa o bastante para resetar."""
    for frame in range(3):
        tracker.update(_obs(frame, (100.0 + frame * 5.0, 100.0)))
    for frame in range(3, ate):
        tracker.update(_obs(frame, None))


def test_a_reancora_respeita_a_ultima_posicao_conhecida() -> None:
    """Depois de perder a bola, reancorar pela maior confianca joga a trajetoria noutro objeto.

    Medido: o fp16 na bola divergia em 61 dos 360 quadros de um clipe, e a cascata inteira vinha de
    uma caixa a 3,7e-05 do corte `conf = 0,10`. Perdida ela, a lacuna nao fechava, a trajetoria
    morria e a reancora escolhia o candidato de maior confianca em qualquer lugar do quadro: 1641 px
    de distancia, outro objeto. O gate de velocidade que protege a continuacao nao valia aqui.
    """
    tracker = BallTracker()
    _perde_a_trajetoria(tracker, 10)

    pontos = tracker.update(_obs_multi(10, ((1700.0, 900.0), 0.95), ((120.0, 100.0), 0.30)))

    escolhido = [p for p in pontos if p.source == "detected"]
    assert len(escolhido) == 1
    assert escolhido[0].center == (120.0, 100.0)


def test_reancora_impossivel_nao_inventa_trajetoria() -> None:
    """Se o unico candidato esta longe demais para ser a mesma bola, a lacuna continua aberta.

    Ancorar nele seria trocar de objeto em silencio, que e pior que admitir que a bola foi perdida.
    """
    tracker = BallTracker()
    _perde_a_trajetoria(tracker, 10)

    pontos = tracker.update(_obs_multi(10, ((1700.0, 900.0), 0.95)))

    assert [p for p in pontos if p.source == "detected"] == []
    assert tracker.telemetry.rejected_jumps >= 1


def test_a_memoria_da_perda_expira_pela_propria_fisica() -> None:
    """Sem prazo, uma posse nova do outro lado do campo nunca ancoraria.

    O prazo nao precisa de parametro novo: o raio permitido e `3000 px/s * dt * 1,5`, entao passado
    cerca de meio segundo ele ja excede a diagonal de um quadro 1920x1080 e admite qualquer posicao.
    """
    tracker = BallTracker()
    _perde_a_trajetoria(tracker, 10)

    longe = _ball((1700.0, 900.0), 0.95)
    pontos = tracker.update(BallObservation(100, 100 / 30.0, (longe,)))

    escolhido = [p for p in pontos if p.source == "detected"]
    assert len(escolhido) == 1
    assert escolhido[0].center == (1700.0, 900.0)


def test_a_primeira_ancora_do_clipe_continua_pela_confianca() -> None:
    """Sem posicao anterior nao ha o que respeitar, e o criterio segue sendo a maior confianca."""
    tracker = BallTracker()

    pontos = tracker.update(_obs_multi(0, ((1700.0, 900.0), 0.95), ((120.0, 100.0), 0.30)))

    assert [p.center for p in pontos if p.source == "detected"] == [(1700.0, 900.0)]
