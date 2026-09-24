"""Tempo e concordancia de planos TensorRT do mesmo peso em precisoes diferentes, medidos na placa.

A quantizacao da Sprint 3 mediu cada modelo por um video, com o Ultralytics, e nao guardou o dado
bruto; o INT8 da Sprint 4 so tinha sido exportado. Aqui cada plano recebe os mesmos quadros reais,
e sai o tempo de cada passada, sem pre e pos-processamento, e o quanto as deteccoes concordam com
as do plano de referencia, que e o fp32 do mesmo peso. Concordancia nao e acuracia: ela diz quanto
a precisao menor muda a resposta, e nao se a resposta esta certa.

O par por IoU de 0,5 e o padrao de deteccao, e e duro com objeto pequeno: numa caixa de 8 px, um
deslocamento de 3 px ja derruba o IoU para 0,45. Por isso a concordancia sai tambem pela distancia
entre os centros, que separa a bola que andou alguns pixels da bola que apareceu ou sumiu.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np

from edge.runtime import RuntimeModel

Tensor = np.ndarray[Any, np.dtype[np.float32]]
IOU_PAR = 0.5
CONF = 0.25
IOU_NMS = 0.45
RAIO_CENTRO = 4.0


class BenchError(ValueError): ...


@dataclass(frozen=True)
class Deteccao:
    caixa: tuple[float, float, float, float]
    score: float
    classe: int
    pontos: Tensor | None = None


def entrada_do_plano(plano: Path) -> tuple[str, tuple[int, ...]]:
    """Nome e forma da entrada, lidos do relatorio que `mlops.build_engine` grava ao lado."""
    relatorio = plano.with_suffix(".run.json")
    if not relatorio.is_file():
        raise BenchError(f"{relatorio} nao existe: construa o plano com mlops.build_engine")
    dados = cast("dict[str, Any]", json.loads(relatorio.read_text(encoding="utf-8")))
    entradas = cast("list[dict[str, Any]]", dados["inputs"])
    if len(entradas) != 1:
        raise BenchError(f"{plano.name} tem {len(entradas)} entradas, e o banco mede uma")
    return str(entradas[0]["name"]), tuple(int(v) for v in entradas[0]["shape"])


def prepara(quadro: np.ndarray[Any, Any], altura: int, largura: int) -> Tensor:
    """Letterbox para a entrada do plano, em RGB, NCHW e escala 0 a 1, como o Ultralytics faz."""
    h, w = quadro.shape[:2]
    ganho = min(altura / h, largura / w)
    novo_h, novo_w = round(h * ganho), round(w * ganho)
    tela = np.full((altura, largura, 3), 114, dtype=np.uint8)
    topo, esquerda = (altura - novo_h) // 2, (largura - novo_w) // 2
    redimensionado = cv2.resize(quadro, (novo_w, novo_h), interpolation=cv2.INTER_LINEAR)
    tela[topo : topo + novo_h, esquerda : esquerda + novo_w] = redimensionado
    rgb = cv2.cvtColor(tela, cv2.COLOR_BGR2RGB)
    return (rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0).astype(np.float32)


def deteccoes(saida: Tensor, *, pontos: int = 0, conf: float = CONF) -> list[Deteccao]:
    """Decodifica a saida crua de um YOLO do Ultralytics, `(1, 4 + classes + pontos*3, ancoras)`.

    Com `pontos` maior que zero e a cabeca de pose, que tem uma classe so e os pontos no fim.
    """
    matriz = np.asarray(saida, dtype=np.float32)[0].T
    classes = matriz.shape[1] - 4 - pontos * 3
    if classes < 1:
        raise BenchError(f"saida de {matriz.shape[1]} canais nao comporta {pontos} pontos")
    scores = matriz[:, 4 : 4 + classes]
    melhor = scores.argmax(axis=1)
    confianca = scores[np.arange(len(scores)), melhor]
    mantidos = np.flatnonzero(confianca >= conf)
    if not len(mantidos):
        return []
    xywh = matriz[mantidos, :4]
    caixas = [[float(x - w / 2), float(y - h / 2), float(w), float(h)] for x, y, w, h in xywh]
    escolhidos = cv2.dnn.NMSBoxes(caixas, confianca[mantidos].tolist(), conf, IOU_NMS)
    saidas: list[Deteccao] = []
    for i in np.asarray(escolhidos).reshape(-1):
        linha = matriz[mantidos[i]]
        x, y, w, h = (float(v) for v in linha[:4])
        extras = linha[4 + classes :].reshape(pontos, 3) if pontos else None
        saidas.append(
            Deteccao(
                caixa=(x - w / 2, y - h / 2, x + w / 2, y + h / 2),
                score=float(confianca[mantidos[i]]),
                classe=int(melhor[mantidos[i]]),
                pontos=None if extras is None else extras.astype(np.float32),
            )
        )
    return saidas


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    uniao = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / uniao if uniao > 0 else 0.0


def concordancia(
    referencia: Sequence[Deteccao], outra: Sequence[Deteccao]
) -> tuple[int, int, int, list[float]]:
    """Pares por IoU e classe, gulosos pelo score: (casados, so na referencia, so na outra, pontos).

    A ultima lista e a distancia, em pixel da entrada, entre os pontos visiveis nos dois lados de
    cada par, quando a cabeca e de pose.
    """
    livres = list(outra)
    casados = 0
    distancias: list[float] = []
    for ref in sorted(referencia, key=lambda d: -d.score):
        candidatos = [
            (iou(ref.caixa, o.caixa), n) for n, o in enumerate(livres) if o.classe == ref.classe
        ]
        if not candidatos:
            continue
        valor, n = max(candidatos)
        if valor < IOU_PAR:
            continue
        par = livres.pop(n)
        casados += 1
        if ref.pontos is not None and par.pontos is not None:
            visiveis = (ref.pontos[:, 2] > 0.5) & (par.pontos[:, 2] > 0.5)
            dif = np.linalg.norm(ref.pontos[visiveis, :2] - par.pontos[visiveis, :2], axis=1)
            distancias.extend(float(d) for d in dif)
    return casados, len(referencia) - casados, len(livres), distancias


def por_centro(referencia: Sequence[Deteccao], outra: Sequence[Deteccao], raio: float) -> int:
    """Pares da mesma classe com centros a ate `raio` px, gulosos pelo score da referencia."""
    livres = list(outra)
    casados = 0
    for ref in sorted(referencia, key=lambda d: -d.score):
        centro = _centro(ref.caixa)
        candidatos = [
            (math.dist(centro, _centro(o.caixa)), n)
            for n, o in enumerate(livres)
            if o.classe == ref.classe
        ]
        if not candidatos:
            continue
        distancia, n = min(candidatos)
        if distancia <= raio:
            livres.pop(n)
            casados += 1
    return casados


def _centro(caixa: Sequence[float]) -> tuple[float, float]:
    return (caixa[0] + caixa[2]) / 2, (caixa[1] + caixa[3]) / 2


def compara(
    planos: Mapping[str, Path],
    quadros: Sequence[np.ndarray[Any, Any]],
    carrega: Callable[[Path], RuntimeModel],
    *,
    referencia: str,
    pontos: int = 0,
    aquecimento: int = 5,
    raio_centro: float = RAIO_CENTRO,
    relogio: Callable[[], float] = time.perf_counter,
) -> dict[str, object]:
    """Roda os mesmos quadros em cada plano e compara as deteccoes com as do plano de referencia."""
    if referencia not in planos:
        raise BenchError(f"a referencia {referencia} nao esta entre {', '.join(planos)}")
    if not quadros:
        raise BenchError("o banco precisa de pelo menos um quadro")
    saidas: dict[str, list[list[Deteccao]]] = {}
    relatorio: dict[str, object] = {"quadros": len(quadros), "referencia": referencia}
    por_plano: dict[str, object] = {}
    for nome, plano in planos.items():
        entrada, forma = entrada_do_plano(plano)
        modelo = carrega(plano)
        lotes = [prepara(q, forma[2], forma[3]) for q in quadros]
        for lote in lotes[: min(aquecimento, len(lotes))]:
            modelo({entrada: lote})
        tempos: list[float] = []
        resultados: list[list[Deteccao]] = []
        for lote in lotes:
            inicio = relogio()
            saida = modelo({entrada: lote})
            tempos.append((relogio() - inicio) * 1000)
            resultados.append(deteccoes(cast("Tensor", next(iter(saida.values()))), pontos=pontos))
        saidas[nome] = resultados
        ordenados = sorted(tempos)
        por_plano[nome] = {
            "plano": str(plano),
            "entrada": list(forma),
            "ms_por_passada": [round(t, 3) for t in tempos],
            "ms_mediana": round(ordenados[len(ordenados) // 2], 3),
            "ms_p95": round(ordenados[min(len(ordenados) - 1, int(0.95 * len(ordenados)))], 3),
            "deteccoes_por_quadro": round(sum(map(len, resultados)) / len(resultados), 2),
        }
    for nome in planos:
        if nome == referencia:
            continue
        casados = so_ref = so_outra = centro = 0
        distancias: list[float] = []
        for ref, outra in zip(saidas[referencia], saidas[nome], strict=True):
            c, r, o, d = concordancia(ref, outra)
            casados, so_ref, so_outra = casados + c, so_ref + r, so_outra + o
            distancias.extend(d)
            centro += por_centro(ref, outra, raio_centro)
        na_referencia, neste = casados + so_ref, casados + so_outra
        cast("dict[str, object]", por_plano[nome])["contra_referencia"] = {
            "casados": casados,
            "so_na_referencia": so_ref,
            "so_neste": so_outra,
            "revocacao": round(casados / na_referencia, 4) if na_referencia else None,
            "precisao": round(casados / neste, 4) if neste else None,
            "pontos_distancia_mediana_px": (
                round(float(np.median(distancias)), 3) if distancias else None
            ),
            "raio_centro_px": raio_centro,
            "revocacao_centro": round(centro / na_referencia, 4) if na_referencia else None,
            "precisao_centro": round(centro / neste, 4) if neste else None,
        }
    relatorio["planos"] = por_plano
    return relatorio
