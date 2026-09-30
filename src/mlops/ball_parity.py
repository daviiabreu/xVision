"""Paridade da bola entre dois runtimes no quadro do contato, candidato a candidato (TB-04).

O TB-04 no agregado comparava PCK, mediana e p95, e uma troca de candidato que ganha num clipe e
perde noutro some nesses numeros. Aqui as duas configuracoes recebem os mesmos quadros decodificados
e, no quadro do contato anotado, o mesmo que o `eval-ball` usa, saem os candidatos do detector antes
do ranking e o ponto que o rastreador escolheu. Os candidatos sao pareados pelo centro, e o
relatorio diz quanto a confianca e o centro andaram, o que ficou sem par e em quantos contatos o
ponto escolhido mudou.

Concordancia, e nao acuracia: o numero diz quanto o runtime muda a resposta, e nao se ela esta
certa. Quem julga acerto no contato continua sendo o `eval-ball`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np

from contracts import BallDetection, Box, Clip
from edge.graph.runner import assemble
from edge.modules import inference
from edge.modules.base import FrameBatch, ModuleInputs, ModuleOutput, ModuleSpec
from mlops.ball_quality import quadro_do_contato

RAIO_PAR_PX = 4.0
"""Distancia maxima entre centros, em pixel do quadro original, para dois candidatos serem um so.

A bola mede de 13,5 a 14,1 px de lado no quadro de 1920x1080, e o NMS da juncao dos ladrilhos
(`nms_iou = 0,1`) funde duas caixas desse tamanho ate cerca de 11 px entre centros. Com 4 px, abaixo
de metade disso, cada candidato tem no maximo um vizinho possivel do outro lado, e o par nao depende
da ordem. E quatro vezes o limite de 1 px do criterio, de proposito: candidato que andou 2 px sai
pareado e reprova, em vez de sumir como sem par. E o mesmo raio do par pelo centro de
`mlops.bench_plano`, que separou bola deslocada de bola nova no INT8.
"""

LIMITE_CONFIANCA = 0.02
LIMITE_CENTRO_PX = 1.0
"""O criterio do TB-04 entre candidatos pareados, fixado na especificacao antes de medir."""


class ParidadeError(ValueError): ...


class BolaComCandidatos(Protocol):
    spec: ModuleSpec

    def infer(self, batch: FrameBatch, ctx: ModuleInputs) -> ModuleOutput: ...

    def candidatos(self, batch: FrameBatch) -> tuple[tuple[BallDetection, ...], ...]: ...


@dataclass(frozen=True)
class LadoDoContato:
    """O que uma configuracao viu no quadro do contato: os candidatos e o ponto escolhido."""

    clip_id: str
    frame_index: int | None
    candidatos: tuple[BallDetection, ...]
    escolhido: BallDetection | None


@dataclass(frozen=True)
class ContatoPareado:
    clip_id: str
    frame_index: int | None
    pares: tuple[tuple[BallDetection, BallDetection], ...]
    so_na_referencia: int
    so_na_comparada: int
    escolha_mudou: bool

    @property
    def diferencas_de_confianca(self) -> list[float]:
        return [abs(a.confidence - b.confidence) for a, b in self.pares]

    @property
    def distancias_px(self) -> list[float]:
        return [_distancia(a.box, b.box) for a, b in self.pares]


@dataclass(frozen=True)
class ParidadeReport:
    contatos: tuple[ContatoPareado, ...]
    raio_px: float

    @property
    def dentro_do_criterio(self) -> bool:
        """O criterio exige pares e um quadro medido em cada contato selecionado."""
        confiancas = [d for c in self.contatos for d in c.diferencas_de_confianca]
        distancias = [d for c in self.contatos for d in c.distancias_px]
        return (
            bool(confiancas)
            and all(c.frame_index is not None for c in self.contatos)
            and max(confiancas) <= LIMITE_CONFIANCA
            and max(distancias) <= LIMITE_CENTRO_PX
            and not any(c.escolha_mudou for c in self.contatos)
        )

    def to_dict(self) -> dict[str, object]:
        confiancas = [d for c in self.contatos for d in c.diferencas_de_confianca]
        distancias = [d for c in self.contatos for d in c.distancias_px]
        pareados = sum(len(c.pares) for c in self.contatos)
        so_ref = sum(c.so_na_referencia for c in self.contatos)
        so_cmp = sum(c.so_na_comparada for c in self.contatos)
        mudaram = [c.clip_id for c in self.contatos if c.escolha_mudou]
        return {
            "n_contatos": len(self.contatos),
            "sem_quadro_no_contato": sum(c.frame_index is None for c in self.contatos),
            "candidatos_referencia": pareados + so_ref,
            "candidatos_comparada": pareados + so_cmp,
            "pareados": pareados,
            "so_na_referencia": so_ref,
            "so_na_comparada": so_cmp,
            "raio_par_px": self.raio_px,
            "confianca_diferenca_max": _maximo(confiancas),
            "confianca_diferenca_p95": _p95(confiancas),
            "centro_distancia_max_px": _maximo(distancias),
            "centro_distancia_p95_px": _p95(distancias),
            "escolha_mudou": len(mudaram),
            "clipes_com_escolha_mudada": mudaram,
            "limite_confianca": LIMITE_CONFIANCA,
            "limite_centro_px": LIMITE_CENTRO_PX,
            "dentro_do_criterio": self.dentro_do_criterio,
        }


def lado_do_contato(
    bola: BolaComCandidatos, clip: Clip, lote: FrameBatch, contato: int
) -> LadoDoContato:
    """Roda o modulo no lote inteiro e le o quadro do contato, com o cache de inferencia vazio.

    O lote inteiro porque o ponto escolhido depende do rastreador, que precisa dos quadros vizinhos.
    O cache e esvaziado antes porque a chave dele e peso, `conf`, `imgsz` e precisao, e nao o
    runtime: o `.pt` em fp32 e o plano TensorRT construido dele casam na chave, e a segunda
    configuracao leria as deteccoes da primeira e sairia identica por construcao.
    """
    inference.reset()
    saida = bola.infer(lote, {})
    quadro = quadro_do_contato(assemble(clip, {"ball": saida}).frames, contato)
    if quadro is None:
        return LadoDoContato(clip.clip_id, None, (), None)
    posicao = lote.indices.index(quadro.frame_index)
    so_o_contato = replace(
        lote, indices=(quadro.frame_index,), tensor=lote.tensor[posicao : posicao + 1]
    )
    (candidatos,) = bola.candidatos(so_o_contato)
    return LadoDoContato(clip.clip_id, quadro.frame_index, candidatos, quadro.ball)


def parear(
    referencia: Sequence[BallDetection], comparada: Sequence[BallDetection], raio_px: float
) -> tuple[tuple[tuple[BallDetection, BallDetection], ...], int, int]:
    """Pares pelo centro livre mais proximo ate o raio, na ordem da confianca da referencia.

    Devolve os pares, quantos ficaram so na referencia e quantos so na comparada.
    """
    livres = list(comparada)
    pares: list[tuple[BallDetection, BallDetection]] = []
    for candidato in sorted(referencia, key=lambda d: -d.confidence):
        perto = min(
            ((_distancia(candidato.box, outro.box), n) for n, outro in enumerate(livres)),
            default=None,
        )
        if perto is not None and perto[0] <= raio_px:
            pares.append((candidato, livres.pop(perto[1])))
    return tuple(pares), len(referencia) - len(pares), len(livres)


def escolha_mudou(a: BallDetection | None, b: BallDetection | None, raio_px: float) -> bool:
    """Um lado com bola e o outro sem, a origem trocada, ou os centros alem do raio de par.

    A origem conta porque o TB-02 so aceita posicao detectada: a mesma posicao, interpolada de um
    lado e detectada do outro, muda o resultado do contato.
    """
    if a is None or b is None:
        return (a is None) != (b is None)
    return a.source != b.source or _distancia(a.box, b.box) > raio_px


def compara(
    referencia: Sequence[LadoDoContato],
    comparada: Sequence[LadoDoContato],
    raio_px: float = RAIO_PAR_PX,
) -> ParidadeReport:
    if not math.isfinite(raio_px) or raio_px <= 0:
        raise ParidadeError(f"o raio de par e distancia em pixel, e veio {raio_px!r}")
    chaves = [(lado.clip_id, lado.frame_index) for lado in referencia]
    if chaves != [(lado.clip_id, lado.frame_index) for lado in comparada]:
        raise ParidadeError("as duas configuracoes nao mediram os mesmos quadros de contato")
    contatos: list[ContatoPareado] = []
    for a, b in zip(referencia, comparada, strict=True):
        pares, so_a, so_b = parear(a.candidatos, b.candidatos, raio_px)
        contatos.append(
            ContatoPareado(
                clip_id=a.clip_id,
                frame_index=a.frame_index,
                pares=pares,
                so_na_referencia=so_a,
                so_na_comparada=so_b,
                escolha_mudou=escolha_mudou(a.escolhido, b.escolhido, raio_px),
            )
        )
    return ParidadeReport(contatos=tuple(contatos), raio_px=raio_px)


def _distancia(a: Box, b: Box) -> float:
    return math.dist(((a.x1 + a.x2) / 2, (a.y1 + a.y2) / 2), ((b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2))


def _maximo(valores: Sequence[float]) -> float | None:
    return max(valores) if valores else None


def _p95(valores: Sequence[float]) -> float | None:
    return float(np.percentile(valores, 95)) if valores else None
