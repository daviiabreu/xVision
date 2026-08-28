"""A folha de contato que substitui a janela na anotacao do instante.

O `opencv-python-headless` que o lock trava sai com `GUI: NONE` no wheel de Linux, entao
`cv2.imshow` nao roda para ninguem do time que nao esteja num Mac, e ali funciona por acidente de
empacotamento. Em vez de depender disso, a ferramenta escreve uma grade de quadros numerados em
PNG e a pessoa responde por linha de comando.

A grade e grosseira na primeira passada e fina na segunda. Achar o segundo em que o pe encosta na
bola e facil olhando o clipe inteiro a 5 fps; achar o quadro exato dentro daquele segundo pede
taxa nativa numa vizinhanca curta. Uma folha unica em taxa nativa teria 400 celulas e ninguem
consegue ler.

O decode passa pelo `FrameProvider`, que e o unico do projeto: nem esta ferramenta abre video por
conta propria. As folhas vao para `var/`, que e ignorado pelo git, porque cada celula e um quadro
de partida e isso e material restrito da Footstats.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from contracts import Clip
from edge.frames.provider import FrameProvider, VideoSource
from edge.modules.base import PreprocessSpec

CELULA = PreprocessSpec(width=320, height=180, color="bgr", letterbox=True, normalize="none")
COLUNAS = 6
CELULA_ZOOM = (480, 270)
COLUNAS_ZOOM = 5
MARGEM = 22
COR_DO_TEXTO = (255, 255, 255)
COR_DA_FAIXA = (0, 0, 0)


def contact_sheet(clip: Clip, indices: tuple[int, ...], destino: Path) -> Path:
    """Escreve a grade dos quadros pedidos, com o indice de cada um queimado na celula."""
    if not indices:
        raise ValueError("uma folha de contato sem quadro nenhum nao ajuda ninguem")
    provider = FrameProvider(VideoSource(clip.path))
    try:
        lote = provider.frames(indices, CELULA)
        folha = _montar(lote.tensor, indices)
    finally:
        provider.close()
    destino.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destino), folha)
    return destino


def _montar(tensor: Any, indices: tuple[int, ...]) -> Any:
    linhas = (len(indices) + COLUNAS - 1) // COLUNAS
    altura, largura = CELULA.height + MARGEM, CELULA.width
    folha = np.zeros((linhas * altura, COLUNAS * largura, 3), dtype=np.uint8)
    for posicao, indice in enumerate(indices):
        linha, coluna = divmod(posicao, COLUNAS)
        topo, esquerda = linha * altura, coluna * largura
        folha[topo : topo + CELULA.height, esquerda : esquerda + largura] = tensor[posicao]
        _rotular(folha, indice, topo + CELULA.height, esquerda, largura)
    return folha


def _rotular(folha: Any, indice: int, topo: int, esquerda: int, largura: int) -> None:
    cv2.rectangle(
        folha, (esquerda, topo), (esquerda + largura, topo + MARGEM), COR_DA_FAIXA, thickness=-1
    )
    cv2.putText(
        folha,
        f"quadro {indice}",
        (esquerda + 6, topo + MARGEM - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        COR_DO_TEXTO,
        1,
        cv2.LINE_AA,
    )


def coarse_indices(clip: Clip, fps: float = 5.0) -> tuple[int, ...]:
    passo = max(1, round(clip.fps / fps))
    return tuple(range(0, clip.frame_count, passo))


def fine_indices(clip: Clip, ancora: int, raio: int = 15) -> tuple[int, ...]:
    inicio = max(0, ancora - raio)
    fim = min(clip.frame_count - 1, ancora + raio)
    return tuple(range(inicio, fim + 1))


def zoomed_sheet(
    clip: Clip,
    indices: tuple[int, ...],
    janelas: Mapping[int, tuple[int, int, int, int]],
    destino: Path,
) -> Path:
    """A folha recortada na regiao do lance, que e a unica que da para anotar.

    A folha de quadro inteiro poe o jogador em cerca de 17 pixels de altura numa celula de 320 por
    180, e o contato entre pe e bola nao e visivel nesse tamanho: da para dizer em que segundo o
    lance acontece, e nao em que quadro. Medido tentando anotar.

    A janela por quadro vem de fora porque quem sabe onde esta o lance e a percepcao, e este modulo
    nao importa modelo nenhum.
    """
    if not indices:
        raise ValueError("uma folha de contato sem quadro nenhum nao ajuda ninguem")
    largura, altura = CELULA_ZOOM
    linhas = (len(indices) + COLUNAS_ZOOM - 1) // COLUNAS_ZOOM
    folha = np.zeros((linhas * (altura + MARGEM), COLUNAS_ZOOM * largura, 3), dtype=np.uint8)
    capture = cv2.VideoCapture(str(clip.path))
    try:
        for posicao, indice in enumerate(indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, indice)
            lido, quadro = capture.read()
            if not lido:
                continue
            x1, y1, x2, y2 = janelas.get(indice, (0, 0, clip.width, clip.height))
            celula = cv2.resize(quadro[y1:y2, x1:x2], (largura, altura))
            linha, coluna = divmod(posicao, COLUNAS_ZOOM)
            topo, esquerda = linha * (altura + MARGEM), coluna * largura
            folha[topo : topo + altura, esquerda : esquerda + largura] = celula
            _rotular(folha, indice, topo + altura, esquerda, largura)
    finally:
        capture.release()
    destino.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destino), folha, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return destino
