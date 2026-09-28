"""Sinais de reinicio no clipe inteiro antes do chute: bola parada, plano fechado e corte.

Escanteio e falta cobrada para a area comecam de 2 a 4 s antes do chute, fora da janela de 1,5 s
que `play_evidence` le.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median

from contracts import FramePerception, Perception
from edge.features.base import ShotOutput
from edge.features.play_evidence import (
    LACUNA_MAX_S,
    MIN_TRACKS_PARA_CORTE,
    PARADA_ALTURAS_POR_S,
    contact_time_s,
    player_scale_px,
)

REINICIO_MIN_S = 1.0
"""Bola parada por pelo menos isso conta como reinicio. Preliminar."""

PLANO_FECHADO_FATOR = 2.5
"""Quadro cuja caixa mediana passa disso vezes a do clipe e plano fechado. Preliminar."""


@dataclass(frozen=True)
class RestartEvidence:
    """Antecedencias contadas do fim de cada sinal ate o contato; `None` se o sinal nao apareceu."""

    restart_s: float | None
    restart_lead_s: float | None
    close_up_lead_s: float | None
    cut_lead_s: float | None


def measure_restart(perception: Perception, shot: ShotOutput) -> RestartEvidence:
    t_contato = contact_time_s(perception, shot.frame_index)
    if t_contato is None:
        return RestartEvidence(None, None, None, None)
    antes = sorted(
        (quadro for quadro in perception.frames if quadro.timestamp_s <= t_contato),
        key=lambda quadro: quadro.timestamp_s,
    )
    escala = player_scale_px(antes)
    trecho = None if escala is None else _ultimo_trecho_parado(antes, escala)
    fechado = None if escala is None else _ultimo_plano_fechado(antes, escala)
    corte = _ultimo_corte(antes)
    return RestartEvidence(
        restart_s=None if trecho is None else _arredonda(trecho[1] - trecho[0]),
        restart_lead_s=None if trecho is None else _arredonda(t_contato - trecho[1]),
        close_up_lead_s=None if fechado is None else _arredonda(t_contato - fechado),
        cut_lead_s=None if corte is None else _arredonda(t_contato - corte),
    )


def _ultimo_trecho_parado(
    quadros: Sequence[FramePerception], escala: float
) -> tuple[float, float] | None:
    """Inicio e fim do ultimo trecho com a bola detectada parada por `REINICIO_MIN_S` ou mais."""
    detectados = [q for q in quadros if q.ball is not None and q.ball.source == "detected"]
    trechos: list[tuple[float, float]] = []
    inicio: float | None = None
    fim = 0.0
    for antes, depois in zip(detectados, detectados[1:], strict=False):
        decorrido = depois.timestamp_s - antes.timestamp_s
        if decorrido <= 0.0:
            continue
        parado = (
            decorrido <= LACUNA_MAX_S
            and _deslocamento_px(antes, depois) / decorrido / escala <= PARADA_ALTURAS_POR_S
        )
        if parado:
            inicio = antes.timestamp_s if inicio is None else inicio
            fim = depois.timestamp_s
        elif inicio is not None:
            trechos.append((inicio, fim))
            inicio = None
    if inicio is not None:
        trechos.append((inicio, fim))
    longos = [trecho for trecho in trechos if trecho[1] - trecho[0] >= REINICIO_MIN_S]
    return max(longos, key=lambda trecho: trecho[1]) if longos else None


def _ultimo_plano_fechado(quadros: Sequence[FramePerception], escala: float) -> float | None:
    fechados = [
        quadro.timestamp_s
        for quadro in quadros
        if quadro.players
        and median(j.box.y2 - j.box.y1 for j in quadro.players) >= PLANO_FECHADO_FATOR * escala
    ]
    return max(fechados) if fechados else None


def _ultimo_corte(quadros: Sequence[FramePerception]) -> float | None:
    """O ultimo par de quadros vizinhos que perdeu todas as trajetorias de uma vez."""
    ids = [
        frozenset(j.track_id for j in quadro.players if j.track_id is not None)
        for quadro in quadros
    ]
    cortes = [
        quadros[posicao + 1].timestamp_s
        for posicao, (de, para) in enumerate(zip(ids, ids[1:], strict=False))
        if len(de) >= MIN_TRACKS_PARA_CORTE and len(para) >= MIN_TRACKS_PARA_CORTE and not de & para
    ]
    return max(cortes) if cortes else None


def _deslocamento_px(antes: FramePerception, depois: FramePerception) -> float:
    assert antes.ball is not None and depois.ball is not None
    a, b = antes.ball.box, depois.ball.box
    return math.hypot((b.x1 + b.x2 - a.x1 - a.x2) / 2.0, (b.y1 + b.y2 - a.y1 - a.y2) / 2.0)


def _arredonda(valor: float) -> float:
    return round(valor, 4)
