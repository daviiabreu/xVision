"""As acoes do T-DEED antes do chute, lidas por classe."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from contracts import FramePerception, Perception
from edge.features.base import ShotActions
from edge.features.play_actions import measure_actions

FPS = 50.0
CONTATO = 350
CLASSES = ("BACKGROUND", "PASS", "CROSS", "SHOT")


@dataclass(frozen=True)
class _Chute:
    frame_index: int
    shooter_track_id: int | None = None
    actions: ShotActions | None = None


def _relogio() -> Perception:
    """Quadros do `dense` de 0 a 8 s, so para dar tempo aos indices."""
    return Perception(
        clip_id="c",
        frames=tuple(FramePerception(frame_index=i, timestamp_s=i / FPS) for i in range(0, 404, 4)),
    )


def _acoes(picos: dict[int, dict[str, float]]) -> ShotActions:
    """Um passo a cada 0,1 s; cada passo sem pico e so fundo."""
    quadros = tuple(range(0, 400, 5))
    linhas = tuple(
        tuple(picos.get(q, {}).get(c, 1.0 if c == "BACKGROUND" else 0.0) for c in CLASSES)
        for q in quadros
    )
    return ShotActions(classes=CLASSES, frame_indices=quadros, probabilities=linhas)


def test_pico_de_cada_classe_na_janela_e_a_ultima_acao_acima_do_limiar() -> None:
    acoes = _acoes(
        {
            0: {"PASS": 0.99},
            250: {"PASS": 0.9},
            310: {"CROSS": 0.6},
            350: {"SHOT": 0.95, "CROSS": 0.7},
        }
    )

    evidencia = measure_actions(_relogio(), _Chute(CONTATO, actions=acoes))

    assert evidencia is not None
    assert set(evidencia.peaks) == {"PASS", "CROSS"}
    assert evidencia.peaks["PASS"].probability == 0.9
    assert evidencia.peaks["PASS"].lead_s == pytest.approx(2.0)
    assert evidencia.peaks["CROSS"].probability == 0.6
    assert evidencia.last_action == "CROSS"
    assert evidencia.last_action_lead_s == pytest.approx(0.8)


def test_acao_abaixo_do_limiar_nao_e_a_ultima() -> None:
    acoes = _acoes({250: {"PASS": 0.9}, 310: {"CROSS": 0.4}})

    evidencia = measure_actions(_relogio(), _Chute(CONTATO, actions=acoes))

    assert evidencia is not None
    assert evidencia.last_action == "PASS"
    assert evidencia.last_action_lead_s == pytest.approx(2.0)


def test_sem_acao_nenhuma_acima_do_limiar_nao_ha_ultima_acao() -> None:
    evidencia = measure_actions(_relogio(), _Chute(CONTATO, actions=_acoes({})))

    assert evidencia is not None
    assert evidencia.last_action is None
    assert evidencia.peaks["PASS"].probability == 0.0


def test_instante_que_nao_veio_do_tdeed_nao_tem_acao() -> None:
    assert measure_actions(_relogio(), _Chute(CONTATO)) is None
