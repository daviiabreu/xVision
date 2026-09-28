"""Sinais de reinicio no clipe inteiro antes do chute."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pytest

from contracts import BallDetection, Box, FramePerception, Perception, PlayerDetection
from edge.features.play_restart import RestartEvidence, measure_restart

FPS = 50.0
CONTATO = 250


@dataclass(frozen=True)
class _Chute:
    frame_index: int
    shooter_track_id: int | None = None


def _bola(x: float, source: str = "detected") -> BallDetection:
    interpolada = source == "interpolated"
    return BallDetection(
        Box(x - 5, 95, x + 5, 105),
        0.0 if interpolada else 0.9,
        source=source,
        gap_frames=1 if interpolada else None,
    )


def _jogadores(
    tracks: tuple[int, ...] = (2, 3), altura: float = 100.0
) -> tuple[PlayerDetection, ...]:
    return tuple(
        PlayerDetection(Box(600 + 200 * i, 0, 640 + 200 * i, altura), 0.8, track_id=t)
        for i, t in enumerate(tracks)
    )


def _cena(
    bola_em: Callable[[int], BallDetection | None],
    jogadores_em: Callable[[int], tuple[PlayerDetection, ...]] = lambda _: _jogadores(),
) -> Perception:
    """Quadros no passo 4 do `dense`, de 0 ao contato em 5 s."""
    return Perception(
        clip_id="c",
        frames=tuple(
            FramePerception(
                frame_index=i, timestamp_s=i / FPS, players=jogadores_em(i), ball=bola_em(i)
            )
            for i in range(0, CONTATO + 1, 4)
        ),
    )


def _parada_ate(ultimo: int) -> Callable[[int], BallDetection]:
    """Bola parada ate o quadro `ultimo`, e depois andando 30 px por amostra."""
    return lambda i: _bola(100.0 if i <= ultimo else 100.0 + 30.0 * (i - ultimo) / 4)


def test_bola_parada_longe_do_chute_e_reinicio_com_a_antecedencia() -> None:
    reinicio = measure_restart(_cena(_parada_ate(80)), _Chute(CONTATO))

    assert reinicio.restart_s == pytest.approx(1.6)
    assert reinicio.restart_lead_s == pytest.approx(3.4)


def test_bola_sempre_andando_nao_tem_reinicio() -> None:
    reinicio = measure_restart(_cena(_parada_ate(-4)), _Chute(CONTATO))

    assert reinicio.restart_s is None
    assert reinicio.restart_lead_s is None


def test_parada_curta_nao_e_reinicio() -> None:
    assert measure_restart(_cena(_parada_ate(24)), _Chute(CONTATO)).restart_s is None


def test_lacuna_longa_na_deteccao_parte_o_trecho_parado() -> None:
    """Sem deteccao entre 0,6 e 1,2 s, os dois pedacos de 0,6 s nao somam um reinicio."""
    cena = _cena(lambda i: None if 30 < i < 60 else _parada_ate(90)(i))

    assert measure_restart(cena, _Chute(CONTATO)).restart_s is None


def test_bola_interpolada_parada_nao_e_reinicio() -> None:
    cena = _cena(lambda i: _bola(100.0, "interpolated") if i <= 80 else _parada_ate(80)(i))

    assert measure_restart(cena, _Chute(CONTATO)).restart_s is None


def test_plano_fechado_antes_do_chute() -> None:
    cena = _cena(
        _parada_ate(-4),
        lambda i: _jogadores((9,), altura=300.0) if 20 <= i <= 40 else _jogadores(),
    )

    assert measure_restart(cena, _Chute(CONTATO)).close_up_lead_s == pytest.approx(4.2)


def test_corte_de_camera_no_clipe_inteiro() -> None:
    cena = _cena(_parada_ate(-4), lambda i: _jogadores((5, 6)) if i >= 124 else _jogadores())

    assert measure_restart(cena, _Chute(CONTATO)).cut_lead_s == pytest.approx(2.52)


def test_sem_jogador_nao_ha_escala_para_bola_nem_para_plano() -> None:
    reinicio = measure_restart(_cena(_parada_ate(80), lambda _: ()), _Chute(CONTATO))

    assert (reinicio.restart_s, reinicio.close_up_lead_s, reinicio.cut_lead_s) == (None, None, None)


def test_bola_parada_ate_o_contato_e_falta_direta() -> None:
    reinicio = measure_restart(_cena(lambda _: _bola(100.0)), _Chute(CONTATO))

    assert reinicio.restart_s == pytest.approx(4.96)
    assert reinicio.restart_lead_s == pytest.approx(0.04)


def test_percepcao_sem_relogio_nao_tem_sinal_nenhum() -> None:
    uma = Perception(clip_id="c", frames=(FramePerception(frame_index=0, timestamp_s=0.0),))

    assert measure_restart(uma, _Chute(CONTATO)) == RestartEvidence(None, None, None, None)
