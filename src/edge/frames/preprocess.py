"""Aplica a PreprocessSpec declarada pelo modulo, sempre na mesma ordem.

Cor, enquadramento, tratamentos e normalizacao. Os tratamentos vem depois do
enquadramento porque `clahe:2.0:8` fala em blocos da imagem de entrada do modelo, e o
mesmo texto teria significado diferente em cada resolucao de origem se rodasse antes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import cv2
import numpy as np

from edge.modules.base import Color, Normalize, PreprocessSpec, channels

Frame = np.ndarray[Any, Any]
"""Quadro em memoria.

Os dois parametros ficam em `Any` de proposito. Sem parametro nenhum o `pyright` recusa no piso
3.10, onde o numpy ainda nao declara valor padrao para eles, e com parametro concreto o `cv2`
recusa, porque as assinaturas dele nao aceitam `ndarray` estreitado. `Any` e o unico ponto em
que os dois passam.
"""

LETTERBOX_FILL = 0

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_FROM_BGR: dict[Color, int | None] = {
    "bgr": None,
    "rgb": cv2.COLOR_BGR2RGB,
    "gray": cv2.COLOR_BGR2GRAY,
    "ycrcb": cv2.COLOR_BGR2YCrCb,
}


class PreprocessError(ValueError): ...


def prepare(frame: Frame, spec: PreprocessSpec) -> Frame:
    pronto = _recolor(frame, spec.color)
    pronto = _fit(pronto, spec)
    for op in spec.ops:
        pronto = _apply_op(pronto, op, spec.color)
    return _normalize(pronto, spec.normalize, spec.color)


def to_source_box(
    box: tuple[float, float, float, float],
    spec: PreprocessSpec,
    source_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    """Desfaz o enquadramento e devolve a caixa em pixel do quadro original.

    E o inverso exato de `_fit`, e existe porque `edge.modules.records` cobra caixa no espaco do
    quadro original enquanto o modulo so ve o tensor redimensionado. Sem ela, todo detector emitiria
    quatro floats plausiveis no espaco errado, e nem o schema nem a suite de conformidade teriam
    como perceber.

    Nao mexe em cor, em `ops` nem em normalizacao: nenhuma das tres move um pixel de lugar.
    """
    largura, altura = source_size
    if largura <= 0 or altura <= 0:
        raise PreprocessError(
            f"source_size={source_size!r}, e o quadro de origem tem medida positiva"
        )
    x1, y1, x2, y2 = box
    if not spec.letterbox:
        return (
            x1 * largura / spec.width,
            y1 * altura / spec.height,
            x2 * largura / spec.width,
            y2 * altura / spec.height,
        )
    escala = min(spec.width / largura, spec.height / altura)
    conteudo_w = max(1, round(largura * escala))
    conteudo_h = max(1, round(altura * escala))
    esquerda = (spec.width - conteudo_w) // 2
    topo = (spec.height - conteudo_h) // 2
    return (
        (x1 - esquerda) / escala,
        (y1 - topo) / escala,
        (x2 - esquerda) / escala,
        (y2 - topo) / escala,
    )


def to_source_image(
    tensor: np.ndarray[Any, Any],
    spec: PreprocessSpec,
    source_size: tuple[int, int],
) -> np.ndarray[Any, Any]:
    """Desfaz o enquadramento do tensor e devolve a imagem no espaco de pixel do quadro original.

    E a contraparte de `to_source_box` para quem precisa do pixel, e nao so da caixa: medida de cor
    e de residuo de movimento so valem se a imagem e a caixa estiverem no mesmo espaco.

    Na spec de `ball`, 1920x1080 dentro de 1920x1088, a escala e exatamente 1 e a volta e um recorte
    das quatro linhas de preenchimento, sem reamostrar. Quando houve reducao a volta reamostra, e ai
    ela e aproximada: quem chama precisa saber que recuperar pixel perdido nao e possivel.
    """
    largura, altura = source_size
    if largura <= 0 or altura <= 0:
        raise PreprocessError(
            f"source_size={source_size!r}, e o quadro de origem tem medida positiva"
        )
    if not spec.letterbox:
        return _resized(tensor, largura, altura)
    escala = min(spec.width / largura, spec.height / altura)
    conteudo_w = max(1, round(largura * escala))
    conteudo_h = max(1, round(altura * escala))
    esquerda = (spec.width - conteudo_w) // 2
    topo = (spec.height - conteudo_h) // 2
    recorte = tensor[topo : topo + conteudo_h, esquerda : esquerda + conteudo_w]
    return _resized(recorte, largura, altura)


def _resized(tensor: np.ndarray[Any, Any], largura: int, altura: int) -> np.ndarray[Any, Any]:
    if (int(tensor.shape[1]), int(tensor.shape[0])) == (largura, altura):
        return tensor
    return cast("np.ndarray[Any, Any]", cv2.resize(tensor, (largura, altura)))


def to_tensor_box(
    box: tuple[float, float, float, float],
    spec: PreprocessSpec,
    source_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    """Leva a caixa do quadro original para o espaco do tensor, que e a ida de `to_source_box`.

    Serve a quem precisa recortar: a especie `players` entrega caixa em pixel do quadro original,
    e quem vai olhar aquele jogador de perto tem em maos o tensor ja redimensionado.
    """
    largura, altura = source_size
    if largura <= 0 or altura <= 0:
        raise PreprocessError(
            f"source_size={source_size!r}, e o quadro de origem tem medida positiva"
        )
    x1, y1, x2, y2 = box
    if not spec.letterbox:
        return (
            x1 * spec.width / largura,
            y1 * spec.height / altura,
            x2 * spec.width / largura,
            y2 * spec.height / altura,
        )
    escala = min(spec.width / largura, spec.height / altura)
    esquerda = (spec.width - max(1, round(largura * escala))) // 2
    topo = (spec.height - max(1, round(altura * escala))) // 2
    return (
        x1 * escala + esquerda,
        y1 * escala + topo,
        x2 * escala + esquerda,
        y2 * escala + topo,
    )


def _recolor(frame: Frame, color: Color) -> Frame:
    code = _FROM_BGR[color]
    convertido = frame if code is None else cv2.cvtColor(frame, code)
    return convertido[:, :, np.newaxis] if convertido.ndim == 2 else convertido


def _fit(frame: Frame, spec: PreprocessSpec) -> Frame:
    if not spec.letterbox:
        return _keep_axis(cv2.resize(frame, (spec.width, spec.height)))
    altura, largura = frame.shape[0], frame.shape[1]
    escala = min(spec.width / largura, spec.height / altura)
    destino = (max(1, round(largura * escala)), max(1, round(altura * escala)))
    conteudo = _keep_axis(cv2.resize(frame, destino))
    tela: Frame = np.empty((spec.height, spec.width, frame.shape[2]), dtype=frame.dtype)
    tela[...] = LETTERBOX_FILL
    topo = (spec.height - conteudo.shape[0]) // 2
    esquerda = (spec.width - conteudo.shape[1]) // 2
    tela[topo : topo + conteudo.shape[0], esquerda : esquerda + conteudo.shape[1]] = conteudo
    return tela


def _keep_axis(frame: Frame) -> Frame:
    return frame[:, :, np.newaxis] if frame.ndim == 2 else frame


def _apply_op(frame: Frame, op: str, color: Color) -> Frame:
    nome, _, resto = op.partition(":")
    construtor = _OPS.get(nome)
    if construtor is None:
        raise PreprocessError(f"op {op!r} desconhecida: conhecidas sao {', '.join(sorted(_OPS))}")
    argumentos = resto.split(":") if resto else []
    try:
        return construtor(frame, argumentos, color)
    except (ValueError, cv2.error) as erro:
        raise PreprocessError(f"op {op!r} nao pode ser aplicada: {erro}") from erro


def _clahe(frame: Frame, argumentos: list[str], color: Color) -> Frame:
    limite, blocos = argumentos
    filtro = cv2.createCLAHE(clipLimit=float(limite), tileGridSize=(int(blocos), int(blocos)))
    if channels(color) == 1:
        return filtro.apply(frame[:, :, 0])[:, :, np.newaxis]
    luminancia, para, volta = _luminance_codes(color)
    convertido = _sem_alias(frame) if para is None else cv2.cvtColor(frame, para)
    convertido[:, :, luminancia] = filtro.apply(convertido[:, :, luminancia])
    return convertido if volta is None else cv2.cvtColor(convertido, volta)


def _bilateral(frame: Frame, argumentos: list[str], color: Color) -> Frame:
    diametro, cor, espaco = argumentos
    filtrado = cv2.bilateralFilter(frame, int(diametro), float(cor), float(espaco))
    return _keep_axis(filtrado)


def _sem_alias(frame: Frame) -> Frame:
    copia = np.empty_like(frame)
    copia[...] = frame
    return copia


def _luminance_codes(color: Color) -> tuple[int, int | None, int | None]:
    if color == "ycrcb":
        return 0, None, None
    para = cv2.COLOR_BGR2YCrCb if color == "bgr" else cv2.COLOR_RGB2YCrCb
    volta = cv2.COLOR_YCrCb2BGR if color == "bgr" else cv2.COLOR_YCrCb2RGB
    return 0, para, volta


_OPS: dict[str, Callable[[Frame, list[str], Color], Frame]] = {
    "clahe": _clahe,
    "bilateral": _bilateral,
}


def _normalize(frame: Frame, normalize: Normalize, color: Color) -> Frame:
    if normalize == "none":
        return frame
    unit = frame.astype(np.float32) / 255.0
    if normalize == "unit":
        return unit
    if color not in ("rgb", "bgr"):
        raise PreprocessError(
            f"normalize imagenet tem estatistica por canal de rgb e nao diz nada em {color!r}"
        )
    media = np.array(IMAGENET_MEAN, dtype=np.float32)
    desvio = np.array(IMAGENET_STD, dtype=np.float32)
    if color == "bgr":
        media, desvio = media[::-1], desvio[::-1]
    return (unit - media) / desvio
