"""Gera evidencias visuais pequenas a partir da percepcao ja calculada."""

from __future__ import annotations

import base64
from collections.abc import Iterable
from pathlib import Path

import cv2

from contracts import Box, FramePerception, Perception, PlayerDetection
from edge.frames.preprocess import Frame
from edge.frames.provider import VideoSource

LARGURA_MAXIMA = 720
QUALIDADE_JPEG = 72
AMARELO = (46, 217, 248)
VERDE = (92, 181, 76)
VERMELHO = (88, 88, 224)
AZUL = (215, 139, 52)
BRANCO = (255, 255, 255)
PRETO = (20, 20, 20)


def gerar_evidencias(
    caminho: Path,
    frame_index: int,
    percepcao: Perception,
    etapas: Iterable[str],
) -> dict[str, str]:
    referencia = _mais_proximo(percepcao, frame_index)
    indice = referencia.frame_index if referencia is not None else frame_index
    origem = VideoSource(caminho)
    try:
        quadro = origem.read((indice,))[indice]
    finally:
        origem.close()
    return {etapa: _imagem(etapa, quadro, referencia) for etapa in etapas}


def _mais_proximo(percepcao: Perception, frame_index: int) -> FramePerception | None:
    if not percepcao.frames:
        return None
    return min(percepcao.frames, key=lambda item: abs(item.frame_index - frame_index))


def _imagem(etapa: str, original: Frame, referencia: FramePerception | None) -> str:
    quadro = original.copy()
    if referencia is not None:
        _desenhar(etapa, quadro, referencia)
    _cabecalho(quadro, etapa, referencia)
    if quadro.shape[1] > LARGURA_MAXIMA:
        escala = LARGURA_MAXIMA / quadro.shape[1]
        quadro = cv2.resize(
            quadro,
            (LARGURA_MAXIMA, max(1, round(quadro.shape[0] * escala))),
            interpolation=cv2.INTER_AREA,
        )
    ok, codificado = cv2.imencode(".jpg", quadro, (cv2.IMWRITE_JPEG_QUALITY, QUALIDADE_JPEG))
    if not ok:
        raise RuntimeError("o OpenCV nao codificou a evidencia em JPEG")
    conteudo = base64.b64encode(codificado.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{conteudo}"


def _desenhar(etapa: str, quadro: Frame, referencia: FramePerception) -> None:
    if etapa in {"players", "tracking", "identity", "pose", "shot", "assemble"}:
        for jogador in referencia.players:
            _jogador(quadro, jogador, etapa)
    if etapa in {"ball", "shot", "assemble"} and referencia.ball is not None:
        _caixa(quadro, referencia.ball.box, AMARELO)
        _rotulo(quadro, referencia.ball.box, "bola", AMARELO)
    if etapa in {"field", "assemble"} and referencia.field is not None:
        for _, x, y in referencia.field.keypoints:
            cv2.circle(quadro, (round(x), round(y)), 7, VERDE, -1, cv2.LINE_AA)


def _jogador(quadro: Frame, jogador: PlayerDetection, etapa: str) -> None:
    cor = _cor_do_time(jogador.team) if etapa == "identity" else AZUL
    _caixa(quadro, jogador.box, cor)
    if etapa == "tracking" and jogador.track_id is not None:
        _rotulo(quadro, jogador.box, f"id {jogador.track_id}", cor)
    elif etapa == "identity":
        _rotulo(quadro, jogador.box, jogador.team or "equipe indisponivel", cor)
    elif etapa == "players" and jogador.role:
        _rotulo(quadro, jogador.box, jogador.role, cor)
    if etapa == "pose":
        for ponto in jogador.keypoints:
            if ponto.confidence is None or ponto.confidence >= 0.25:
                cv2.circle(quadro, (round(ponto.x), round(ponto.y)), 5, AMARELO, -1, cv2.LINE_AA)


def _caixa(quadro: Frame, caixa: Box, cor: tuple[int, int, int]) -> None:
    cv2.rectangle(
        quadro,
        (round(caixa.x1), round(caixa.y1)),
        (round(caixa.x2), round(caixa.y2)),
        cor,
        3,
        cv2.LINE_AA,
    )


def _rotulo(quadro: Frame, caixa: Box, texto: str, cor: tuple[int, int, int]) -> None:
    origem = (round(caixa.x1), max(24, round(caixa.y1) - 8))
    cv2.putText(quadro, texto, origem, cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2, cv2.LINE_AA)


def _cor_do_time(equipe: str | None) -> tuple[int, int, int]:
    if equipe == "team_a":
        return AZUL
    if equipe == "team_b":
        return VERMELHO
    return AMARELO


def _cabecalho(quadro: Frame, etapa: str, referencia: FramePerception | None) -> None:
    cv2.rectangle(quadro, (0, 0), (quadro.shape[1], 62), PRETO, -1)
    indice = "sem percepcao" if referencia is None else f"frame {referencia.frame_index}"
    cv2.putText(
        quadro,
        f"{etapa} | {indice}",
        (22, 41),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        BRANCO,
        2,
        cv2.LINE_AA,
    )
