"""A especie `tracking`: liga as caixas de `players` ao longo do tempo, sem peso e sem video.

Um `track_id` estavel e o que permite dizer "o mesmo jogador" entre dois quadros, e sem ele nada
que fale de movimento existe: nem a pose sabe de quem e o corpo, nem uma feature consegue medir se
o finalizador girou a cabeca antes do contato.

A associacao e gulosa por sobreposicao, com previsao de posicao pela velocidade dos dois ultimos
quadros. Nao usa `model.track` do Ultralytics de proposito, e o motivo e o contrato: aquele
rastreador guarda estado entre chamadas com `persist=True`, e a suite de conformidade cobra que
duas chamadas com a mesma entrada devolvam a mesma saida. Um rastreador com memoria entre chamadas
reprova nesse teste, e reprovaria com razao: o mesmo lote entregaria numeros diferentes na segunda
vez. Aqui o estado nasce e morre dentro de um `infer`, porque `PerFrame` entrega o conjunto
inteiro de quadros num lote so.

Tambem nao redetecta. Redetectar dentro de `tracking` daria duas listas de jogadores por quadro,
uma dela e outra de `players`, e o `assemble` casa `track_id` a deteccao por `detection_index`, que
e a posicao na lista de `players`. Duas listas, dois indices, e o identificador cairia noutro
jogador em silencio.

Nao mede velocidade, nao reidentifica apos oclusao longa e nao sabe quem chutou.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from contracts import Box
from edge.modules.base import (
    FrameBatch,
    ModuleInputs,
    ModuleNotLoaded,
    ModuleOutput,
    ModuleSpec,
    PreprocessSpec,
    check_preprocess,
)

PREPROCESS = PreprocessSpec(width=32, height=32, color="gray", letterbox=False, normalize="none")
"""O menor lote que o contrato aceita: ele decide pelas caixas de `players`, e nao por pixel."""

MIN_AFINIDADE = 0.2
"""Piso da afinidade para dizer que duas caixas sao o mesmo jogador em quadros vizinhos."""

ALCANCE_EM_DIAGONAIS = 1.5
"""Quantas diagonais de caixa a distancia entre centros pode ter e ainda ser o mesmo jogador.

Medir em diagonais de caixa, e nao em pixel, faz o limite valer igual para o jogador perto da
camera e para o do fundo do campo, cuja caixa tem um terco do tamanho.
"""

MAX_GAP_FRAMES = 15
"""Quantos quadros uma trajetoria sobrevive sem par antes de ser dada por perdida.

A 59,94 fps isso e um quarto de segundo, tempo de um jogador passar atras de outro. Acima disso a
chance de reancorar no jogador errado passa a ser maior que o ganho de manter o numero.
"""


@dataclass
class _Trajetoria:
    track_id: int
    box: Box
    anterior: Box | None
    ultimo_quadro: int


class GreedyTracking:
    spec = ModuleSpec(
        kind="tracking",
        name="greedy-iou",
        version="0.1.0",
        frames="window",
        preprocess=PREPROCESS,
        needs=("players",),
    )

    def __init__(
        self, *, minafinidade: float = MIN_AFINIDADE, max_gap: int = MAX_GAP_FRAMES
    ) -> None:
        self._minafinidade = minafinidade
        self._max_gap = max_gap
        self._pronto = False

    def load(self) -> None:
        self._pronto = True

    def close(self) -> None:
        self._pronto = False

    def infer(self, batch: FrameBatch, ctx: ModuleInputs) -> ModuleOutput:
        if not self._pronto:
            raise ModuleNotLoaded(self.spec.name)
        check_preprocess(batch, self.spec.preprocess)
        if not batch.indices:
            return ModuleOutput(kind=self.spec.kind, frames=(), records=())

        jogadores = _por_quadro(ctx.get("players"))
        vivas: list[_Trajetoria] = []
        proximo = 1
        registros: list[tuple[Mapping[str, object], ...]] = []
        for indice in batch.indices:
            caixas = _caixas(jogadores.get(indice, ()))
            do_quadro, vivas, proximo = self._associar(caixas, vivas, proximo, indice)
            registros.append(do_quadro)
        return ModuleOutput(kind=self.spec.kind, frames=batch.indices, records=tuple(registros))

    def _associar(
        self,
        caixas: Sequence[Box],
        vivas: list[_Trajetoria],
        proximo: int,
        indice: int,
    ) -> tuple[tuple[Mapping[str, object], ...], list[_Trajetoria], int]:
        """Guloso pela melhor sobreposicao, e cada trajetoria casa com uma deteccao so.

        Guloso e nao otimo global porque o custo de uma atribuicao hungara nao se paga com 22
        caixas, e porque a ordem de `players` e estavel: o resultado e deterministico, que e o que
        a conformidade cobra.
        """
        candidatos = sorted(
            (
                (afinidade(_previsto(t), caixa), posicao, ordem)
                for ordem, t in enumerate(vivas)
                for posicao, caixa in enumerate(caixas)
            ),
            key=lambda par: (-par[0], par[1], par[2]),
        )
        usadas: dict[int, int] = {}
        casadas: set[int] = set()
        for valor, posicao, ordem in candidatos:
            if valor < self._minafinidade or posicao in usadas or ordem in casadas:
                continue
            usadas[posicao] = ordem
            casadas.add(ordem)

        registros: list[Mapping[str, object]] = []
        sobreviventes: list[_Trajetoria] = []
        for posicao, caixa in enumerate(caixas):
            ordem = usadas.get(posicao)
            if ordem is None:
                trajetoria = _Trajetoria(proximo, caixa, None, indice)
                proximo += 1
            else:
                antiga = vivas[ordem]
                trajetoria = _Trajetoria(antiga.track_id, caixa, antiga.box, indice)
            sobreviventes.append(trajetoria)
            registros.append({"detection_index": posicao, "track_id": trajetoria.track_id})

        for ordem, trajetoria in enumerate(vivas):
            if ordem not in casadas and indice - trajetoria.ultimo_quadro <= self._max_gap:
                sobreviventes.append(trajetoria)
        return tuple(registros), sobreviventes, proximo


def _previsto(trajetoria: _Trajetoria) -> Box:
    """Onde a caixa estaria se o jogador mantivesse a velocidade dos dois ultimos quadros."""
    if trajetoria.anterior is None:
        return trajetoria.box
    dx = trajetoria.box.x1 - trajetoria.anterior.x1
    dy = trajetoria.box.y1 - trajetoria.anterior.y1
    return Box(
        trajetoria.box.x1 + dx,
        trajetoria.box.y1 + dy,
        trajetoria.box.x2 + dx,
        trajetoria.box.y2 + dy,
    )


def _por_quadro(saida: ModuleOutput | None) -> dict[int, tuple[Mapping[str, object], ...]]:
    if saida is None:
        return {}
    return dict(zip(saida.frames, saida.records, strict=True))


def _caixas(registros: Sequence[Mapping[str, object]]) -> tuple[Box, ...]:
    return tuple(Box(*cast("Any", r["box"])) for r in registros if "box" in r)


def afinidade(a: Box, b: Box) -> float:
    """Sobreposicao quando as caixas se tocam, proximidade dos centros quando nao se tocam.

    So a sobreposicao nao basta, e o caso tem numero: com caixa de 40 px, duas caixas separadas por
    mais de 27 px ja tem IoU abaixo de 0,2, e um jogador amostrado a taxa baixa anda mais que isso
    entre dois quadros. Pior, ele nunca ganharia velocidade para a previsao ajudar, porque a
    velocidade so nasce depois de um casamento que a sobreposicao ja impediu.

    So a proximidade tambem nao basta: duas caixas de tamanhos muito diferentes no mesmo centro
    sao o jogador perto e o do fundo, e a sobreposicao distingue os dois porque usa a forma.
    """
    return max(iou(a, b), _proximidade(a, b))


def _proximidade(a: Box, b: Box) -> float:
    diagonal = (_diagonal(a) + _diagonal(b)) / 2.0
    if diagonal <= 0:
        return 0.0
    distancia = math.hypot(_centro(a)[0] - _centro(b)[0], _centro(a)[1] - _centro(b)[1])
    return max(0.0, 1.0 - distancia / (ALCANCE_EM_DIAGONAIS * diagonal))


def _centro(caixa: Box) -> tuple[float, float]:
    return ((caixa.x1 + caixa.x2) / 2.0, (caixa.y1 + caixa.y2) / 2.0)


def _diagonal(caixa: Box) -> float:
    return math.hypot(caixa.x2 - caixa.x1, caixa.y2 - caixa.y1)


def iou(a: Box, b: Box) -> float:
    largura = min(a.x2, b.x2) - max(a.x1, b.x1)
    altura = min(a.y2, b.y2) - max(a.y1, b.y1)
    if largura <= 0 or altura <= 0:
        return 0.0
    intersecao = largura * altura
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    uniao = area_a + area_b - intersecao
    return 0.0 if uniao <= 0 else intersecao / uniao
