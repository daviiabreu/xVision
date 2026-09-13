"""Decide se o finalizador estava orientado para o gol antes do contato.

O escore geometrico continua nas evidencias para auditoria. O valor publico do evento e um
booleano, usando o limiar semantico de 0,5 enquanto a calibracao estatistica da amostra maior nao
estiver disponivel.
"""

from __future__ import annotations

import math
from statistics import median

from contracts import FeatureValue, FramePerception, Keypoint, Perception, PlayerDetection
from edge.features.base import FeatureSpec, ShotOutput, unavailable
from edge.modules.field.geometry import PITCH_105X68, FieldProjectionError, pitch_to_image

MIN_KEYPOINT_CONFIDENCE = 0.5
MIN_VALID_FRAMES = 3
WINDOW_START_S = 0.5
WINDOW_END_S = 0.1
LOOKED_BEFORE_THRESHOLD = 0.5

_HEAD_POINTS = ("nose", "eye_left", "eye_right", "shoulder_left", "shoulder_right")


class LookedBefore:
    spec = FeatureSpec(
        name="looked_before",
        version="0.2.0",
        requires=("players", "tracking", "field", "shot", "pose"),
        unit="bool",
    )

    def extract(self, perception: Perception, shot: ShotOutput) -> FeatureValue:
        contato = _frame_mais_proximo(perception, shot.frame_index)
        if contato is None:
            return unavailable(self.spec, f"sem percepcao pro frame {shot.frame_index}")

        track_id = _finalizador(perception, shot)
        if track_id is None:
            return unavailable(self.spec, "nao foi possivel resolver um unico finalizador")

        referencia = _referencia_do_finalizador(perception, shot.frame_index, track_id)
        if referencia is None:
            return unavailable(
                self.spec,
                "o finalizador nao tem pitch_xy e homografia proximos ao frame do chute",
            )
        quadro_referencia, jogador_referencia = referencia
        if jogador_referencia.pitch_xy is None:
            return unavailable(self.spec, "o finalizador nao tem pitch_xy")
        gol = _gol_mais_proximo(jogador_referencia.pitch_xy)

        inicio = contato.timestamp_s - WINDOW_START_S
        fim = contato.timestamp_s - WINDOW_END_S
        candidatos = tuple(
            quadro for quadro in perception.frames if inicio <= quadro.timestamp_s <= fim
        )
        amostras: list[tuple[float, float, int]] = []
        for quadro in candidatos:
            jogador = _jogador(quadro, track_id)
            if jogador is None or quadro.field is None or quadro.field.homography is None:
                continue
            try:
                gol_imagem = pitch_to_image(quadro.field.homography, gol)
            except FieldProjectionError:
                continue
            orientacao = _orientacao(jogador.keypoints, gol_imagem)
            if orientacao is not None:
                angulo, confianca = orientacao
                amostras.append((angulo, confianca, quadro.frame_index))

        if len(amostras) < MIN_VALID_FRAMES:
            return unavailable(
                self.spec,
                f"somente {len(amostras)} quadros validos na janela anterior ao chute, "
                f"e a medida exige {MIN_VALID_FRAMES}",
            )

        angulo_mediano = median(amostra[0] for amostra in amostras)
        escore = (math.cos(math.radians(angulo_mediano)) + 1.0) / 2.0
        cobertura = len(amostras) / max(1, len(candidatos))
        confianca = median(amostra[1] for amostra in amostras) * cobertura
        escore = round(escore, 6)
        return FeatureValue(
            name=self.spec.name,
            version=self.spec.version,
            status="ok",
            value=escore >= LOOKED_BEFORE_THRESHOLD,
            confidence=round(min(1.0, confianca), 6),
            reason=None,
            evidence={
                "score": escore,
                "threshold": LOOKED_BEFORE_THRESHOLD,
                "decision": escore >= LOOKED_BEFORE_THRESHOLD,
                "track_id": track_id,
                "frame_do_chute": shot.frame_index,
                "frame_da_posicao": quadro_referencia.frame_index,
                "gol": "direito" if gol[0] > 0 else "esquerdo",
                "angulo_mediano_deg": round(angulo_mediano, 3),
                "quadros_validos": len(amostras),
                "quadros_na_janela": len(candidatos),
                "janela_s": [-WINDOW_START_S, -WINDOW_END_S],
                "frames": [amostra[2] for amostra in amostras],
            },
        )


def _frame_mais_proximo(perception: Perception, frame_index: int) -> FramePerception | None:
    if not perception.frames:
        return None
    return min(perception.frames, key=lambda quadro: abs(quadro.frame_index - frame_index))


def _finalizador(perception: Perception, shot: ShotOutput) -> int | None:
    com_pose = {
        jogador.track_id
        for quadro in perception.frames
        for jogador in quadro.players
        if jogador.track_id is not None and jogador.keypoints
    }
    if shot.shooter_track_id is not None and shot.shooter_track_id in com_pose:
        return shot.shooter_track_id
    if len(com_pose) == 1:
        return next(iter(com_pose))

    quadro = _frame_mais_proximo(perception, shot.frame_index)
    if quadro is None or quadro.ball is None or quadro.ball.source == "interpolated":
        return None
    centro_bola = _centro(
        quadro.ball.box.x1, quadro.ball.box.y1, quadro.ball.box.x2, quadro.ball.box.y2
    )
    candidatos = tuple(
        j for j in quadro.players if j.track_id is not None and j.role != "goalkeeper"
    )
    mais_proximo = min(
        candidatos, key=lambda j: math.dist(_ponto_dos_pes(j), centro_bola), default=None
    )
    return None if mais_proximo is None else mais_proximo.track_id


def _referencia_do_finalizador(
    perception: Perception, frame_index: int, track_id: int
) -> tuple[FramePerception, PlayerDetection] | None:
    candidatos = tuple(
        (quadro, jogador)
        for quadro in perception.frames
        for jogador in quadro.players
        if jogador.track_id == track_id
        and jogador.pitch_xy is not None
        and quadro.field is not None
        and quadro.field.homography is not None
    )
    return min(candidatos, key=lambda par: abs(par[0].frame_index - frame_index), default=None)


def _jogador(quadro: FramePerception, track_id: int) -> PlayerDetection | None:
    return next((j for j in quadro.players if j.track_id == track_id and j.keypoints), None)


def _gol_mais_proximo(posicao: tuple[float, float]) -> tuple[float, float]:
    meio = PITCH_105X68.length_m / 2.0
    return min(((-meio, 0.0), (meio, 0.0)), key=lambda gol: math.dist(posicao, gol))


def _orientacao(
    keypoints: tuple[Keypoint, ...], gol: tuple[float, float]
) -> tuple[float, float] | None:
    pontos = {ponto.name: ponto for ponto in keypoints}
    selecionados = tuple(pontos.get(nome) for nome in _HEAD_POINTS)
    if any(ponto is None for ponto in selecionados):
        return None
    confiancas = tuple(
        p.confidence for p in selecionados if p is not None and p.confidence is not None
    )
    if len(confiancas) != len(_HEAD_POINTS) or min(confiancas) < MIN_KEYPOINT_CONFIDENCE:
        return None
    nariz = pontos["nose"]
    olhos = _meio(pontos["eye_left"], pontos["eye_right"])
    ombros = _meio(pontos["shoulder_left"], pontos["shoulder_right"])
    rosto = ((nariz.x + olhos[0]) / 2.0, (nariz.y + olhos[1]) / 2.0)
    frente = (rosto[0] - ombros[0], rosto[1] - ombros[1])
    para_gol = (gol[0] - rosto[0], gol[1] - rosto[1])
    norma = math.hypot(*frente) * math.hypot(*para_gol)
    if norma < 1e-9:
        return None
    cosseno = max(-1.0, min(1.0, _produto(frente, para_gol) / norma))
    return math.degrees(math.acos(cosseno)), min(confiancas)


def _meio(a: Keypoint, b: Keypoint) -> tuple[float, float]:
    return ((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)


def _produto(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _centro(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _ponto_dos_pes(jogador: PlayerDetection) -> tuple[float, float]:
    return _centro(jogador.box.x1, jogador.box.y1, jogador.box.x2, jogador.box.y2)[
        0
    ], jogador.box.y2
