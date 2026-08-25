"""Aplica a PreprocessSpec declarada pelo modulo, sempre na mesma ordem.

Cor, enquadramento, tratamentos e normalizacao. Os tratamentos vem depois do
enquadramento porque `clahe:2.0:8` fala em blocos da imagem de entrada do modelo, e o
mesmo texto teria significado diferente em cada resolucao de origem se rodasse antes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

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
