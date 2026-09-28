"""O quadro em que a geometria do chute e medida, o finalizador nele e a ancora no pe dele.

A ancora e o pe, e nao a bola, porque o runner nunca projeta a bola no gramado.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from contracts import FramePerception, Perception, PlayerDetection
from edge.features.base import ShotOutput
from edge.features.play_evidence import contact_time_s
from edge.features.touch import shooter_at_contact

TOLERANCIA_S = 0.2
"""Distancia maxima, em segundos, entre o quadro de referencia e o contato. Preliminar."""

DECLARADO = "declared"
PELO_TOQUE = "touch"

Recusa = Literal["sem-relogio", "sem-quadro-com-campo", "sem-finalizador", "finalizador-fora"]


@dataclass(frozen=True)
class ShotScene:
    frame: FramePerception
    shooter: PlayerDetection
    anchor_xy: tuple[float, float]
    shooter_source: Literal["declared", "touch"]
    offset_s: float
    """Tempo do quadro de referencia menos o do contato: negativo e antes do chute."""


@dataclass(frozen=True)
class SceneRefusal:
    reason: Recusa
    detail: str


def shot_scene(
    perception: Perception, shot: ShotOutput, *, tolerance_s: float = TOLERANCIA_S
) -> ShotScene | SceneRefusal:
    """A cena do chute, ou a recusa com o motivo. Nunca troca o finalizador por outro jogador."""
    t_contato = contact_time_s(perception, shot.frame_index)
    if t_contato is None:
        return SceneRefusal(
            "sem-relogio", "a percepcao tem menos de duas amostras para situar o contato no tempo"
        )
    quadro = _quadro_de_referencia(perception, t_contato, tolerance_s)
    if quadro is None:
        return SceneRefusal(
            "sem-quadro-com-campo",
            f"nenhum quadro a ate {tolerance_s:.2f} s do contato tem homografia e jogador em metro",
        )
    declarado = shot.shooter_track_id
    if declarado is not None:
        track_id, origem = declarado, DECLARADO
    else:
        contato = shooter_at_contact(perception, shot.frame_index)
        if contato is None:
            return SceneRefusal(
                "sem-finalizador", "nenhum jogador tinha a bola no episodio do contato"
            )
        track_id, origem = contato.player_track_id, PELO_TOQUE
    finalizador = next(
        (j for j in quadro.players if j.track_id == track_id and j.pitch_xy is not None), None
    )
    if finalizador is None or finalizador.pitch_xy is None:
        return SceneRefusal(
            "finalizador-fora",
            f"o finalizador nao aparece em metro no quadro {quadro.frame_index}, o de referencia",
        )
    return ShotScene(
        frame=quadro,
        shooter=finalizador,
        anchor_xy=finalizador.pitch_xy,
        shooter_source=origem,
        offset_s=quadro.timestamp_s - t_contato,
    )


def _quadro_de_referencia(
    perception: Perception, t_contato: float, tolerancia_s: float
) -> FramePerception | None:
    """O quadro com homografia e jogador em metro mais perto do contato; no empate, o anterior."""
    candidatos = [
        quadro
        for quadro in perception.frames
        if quadro.field is not None
        and quadro.field.homography is not None
        and any(jogador.pitch_xy is not None for jogador in quadro.players)
        and abs(quadro.timestamp_s - t_contato) <= tolerancia_s
    ]
    if not candidatos:
        return None
    return min(
        candidatos, key=lambda quadro: (abs(quadro.timestamp_s - t_contato), quadro.timestamp_s)
    )
