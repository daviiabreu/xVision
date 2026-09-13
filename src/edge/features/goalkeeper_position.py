"""Decide se o goleiro esta lateralmente posicionado para o chute."""

from __future__ import annotations

import math

from contracts import FeatureValue, FramePerception, Perception, PlayerDetection
from edge.features.base import FeatureSpec, ShotOutput, unavailable
from edge.modules.field.geometry import PITCH_105X68

GOAL_HALF_WIDTH_M = 3.66
GOALKEEPER_POSITION_THRESHOLD = 0.5


class GoalkeeperPosition:
    spec = FeatureSpec(
        name="goalkeeper_position",
        version="0.2.0",
        requires=("players", "tracking", "field", "shot"),
        unit="bool",
    )

    def extract(self, perception: Perception, shot: ShotOutput) -> FeatureValue:
        referencia = _quadro_de_referencia(perception, shot)
        if referencia is None:
            return unavailable(
                self.spec, "nenhum quadro proximo ao chute tem homografia e jogadores em pitch_xy"
            )
        finalizador = _finalizador(referencia, shot)
        if finalizador is None or finalizador.pitch_xy is None:
            return unavailable(self.spec, "nao foi possivel resolver o finalizador no chute")
        gol = _gol_mais_proximo(finalizador.pitch_xy)
        goleiro = _goleiro_mais_proximo(referencia, gol)
        if goleiro is None or goleiro.pitch_xy is None:
            return unavailable(self.spec, "nenhum goleiro com pitch_xy no quadro do chute")
        desvio = _distancia_da_linha(goleiro.pitch_xy, finalizador.pitch_xy, gol)
        escore = round(max(0.0, 1.0 - desvio / GOAL_HALF_WIDTH_M), 6)
        return FeatureValue(
            name=self.spec.name,
            version=self.spec.version,
            status="ok",
            value=escore >= GOALKEEPER_POSITION_THRESHOLD,
            confidence=min(finalizador.confidence, goleiro.confidence),
            reason=None,
            evidence={
                "score": escore,
                "threshold": GOALKEEPER_POSITION_THRESHOLD,
                "decision": escore >= GOALKEEPER_POSITION_THRESHOLD,
                "frame_do_chute": shot.frame_index,
                "frame_utilizado": referencia.frame_index,
                "shooter_track_id": finalizador.track_id,
                "goalkeeper_track_id": goleiro.track_id,
                "gol": "direito" if gol[0] > 0 else "esquerdo",
                "desvio_lateral_m": round(desvio, 4),
                "distancia_goleiro_gol_m": round(math.dist(goleiro.pitch_xy, gol), 4),
            },
        )


def _quadro_de_referencia(perception: Perception, shot: ShotOutput) -> FramePerception | None:
    candidatos = tuple(
        q
        for q in perception.frames
        if q.field is not None
        and q.field.homography is not None
        and any(j.pitch_xy is not None for j in q.players)
    )
    if shot.shooter_track_id is not None:
        com_finalizador = tuple(
            q
            for q in candidatos
            if any(
                j.track_id == shot.shooter_track_id and j.pitch_xy is not None for j in q.players
            )
        )
        if com_finalizador:
            candidatos = com_finalizador
    return min(candidatos, key=lambda q: abs(q.frame_index - shot.frame_index), default=None)


def _finalizador(quadro: FramePerception, shot: ShotOutput) -> PlayerDetection | None:
    candidatos = tuple(j for j in quadro.players if j.pitch_xy is not None)
    if shot.shooter_track_id is not None:
        declarado = next((j for j in candidatos if j.track_id == shot.shooter_track_id), None)
        if declarado is not None:
            return declarado
    com_pose = tuple(j for j in candidatos if j.keypoints)
    if len(com_pose) == 1:
        return com_pose[0]
    if quadro.ball is None or quadro.ball.source == "interpolated":
        return None
    centro_bola = (
        (quadro.ball.box.x1 + quadro.ball.box.x2) / 2.0,
        (quadro.ball.box.y1 + quadro.ball.box.y2) / 2.0,
    )
    jogadores = tuple(j for j in candidatos if j.role != "goalkeeper")
    return min(jogadores, key=lambda j: math.dist(_ponto_dos_pes(j), centro_bola), default=None)


def _goleiro_mais_proximo(
    quadro: FramePerception, gol: tuple[float, float]
) -> PlayerDetection | None:
    goleiros = tuple(j for j in quadro.players if j.role == "goalkeeper" and j.pitch_xy is not None)
    return min(goleiros, key=lambda j: math.dist(j.pitch_xy or gol, gol), default=None)


def _gol_mais_proximo(posicao: tuple[float, float]) -> tuple[float, float]:
    meio = PITCH_105X68.length_m / 2.0
    return min(((-meio, 0.0), (meio, 0.0)), key=lambda gol: math.dist(posicao, gol))


def _distancia_da_linha(
    ponto: tuple[float, float], inicio: tuple[float, float], fim: tuple[float, float]
) -> float:
    direcao = (fim[0] - inicio[0], fim[1] - inicio[1])
    comprimento = math.hypot(*direcao)
    if comprimento < 1e-9:
        raise ValueError("finalizador coincide com o centro do gol")
    deslocamento = (ponto[0] - inicio[0], ponto[1] - inicio[1])
    return abs(direcao[0] * deslocamento[1] - direcao[1] * deslocamento[0]) / comprimento


def _ponto_dos_pes(jogador: PlayerDetection) -> tuple[float, float]:
    return ((jogador.box.x1 + jogador.box.x2) / 2.0, jogador.box.y2)
