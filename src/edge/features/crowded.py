"""Conta jogadores dentro de um raio real (metros) da bola, no frame do chute.

Precisa de `players` e `ball` com `pitch_xy` (ou seja, homografia resolvida nesse
frame) — sem isso, a plataforma nem chama esta feature (players/ball ausentes
via `requires`) ou ela mesma sai `unavailable` (bola sem pitch_xy resolvido).
"""

from __future__ import annotations

import math

from contracts import FeatureValue, FramePerception, Perception
from edge.features.base import FeatureSpec, ShotOutput, unavailable

RADIUS_M = 3.0


class Crowded:
    spec = FeatureSpec(
        name="crowded",
        version="0.1.0",
        requires=("players", "ball"),
        unit="jogadores",
    )

    def __init__(self, radius_m: float = RADIUS_M) -> None:
        self.radius_m = radius_m

    def extract(self, perception: Perception, shot: ShotOutput) -> FeatureValue:
        frame = _frame_do_chute(perception, shot.frame_index)
        if frame is None:
            return unavailable(self.spec, f"sem percepcao pro frame {shot.frame_index}")

        if frame.ball is None or frame.ball.pitch_xy is None:
            return unavailable(
                self.spec,
                "bola sem pitch_xy nesse frame (homografia ausente ou bola nao detectada)",
            )

        jogadores_com_posicao = [p for p in frame.players if p.pitch_xy is not None]
        if not jogadores_com_posicao:
            return unavailable(self.spec, "nenhum jogador com pitch_xy nesse frame")

        bola_xy = frame.ball.pitch_xy
        count = sum(
            1
            for p in jogadores_com_posicao
            if p.pitch_xy is not None and math.dist(p.pitch_xy, bola_xy) <= self.radius_m
        )
        return FeatureValue(
            name=self.spec.name,
            version=self.spec.version,
            status="ok",
            value=count,
            confidence=None,
            reason=None,
            evidence={"radius_m": self.radius_m, "player_count": len(jogadores_com_posicao)},
        )


def _frame_do_chute(perception: Perception, frame_index: int) -> FramePerception | None:
    for frame in perception.frames:
        if frame.frame_index == frame_index:
            return frame
    return None
