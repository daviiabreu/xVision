"""O instante do contato lido do gabarito manual, para separar dois erros que se confundem.

Uma feature geometrica errada tem duas causas possiveis e o numero final nao distingue: ou a regra
esta errada, ou ela esta certa e olhando o quadro errado. Medido no detector cinematico contra 38
clipes anotados a mao, a mediana do erro e 63 quadros e 45% dos contatos caem fora da janela de
extracao, que pega 90 quadros antes e 30 depois. Com o contato fora da janela, `body_part` le a
pose de outro momento da jogada, e acertar viraria coincidencia.

Este modulo troca o detector pelo gabarito e deixa o resto do grafo igual. O que sobrar de erro nao
vem mais do instante. E o que a diferenca entre as duas medidas mostrar e exatamente o quanto o
instante custa, que e o numero que decide se o proximo esforco vai para o detector ou para a regra.

Nao e modulo de producao e nao pode virar um: ele le o rotulo do proprio conjunto de avaliacao, e
usa-lo fora deste experimento seria vazamento. Mora em `mlops` por isso, onde a fronteira de import
do projeto (`edge` nao importa `mlops`) impede que uma configuracao de borda o alcance por engano.

Clipe sem anotacao devolve saida vazia em vez de estimar. Estimar poria de volta no experimento
exatamente o erro que ele existe para eliminar.
"""

from __future__ import annotations

from collections.abc import Mapping

from edge.modules.base import (
    FrameBatch,
    ModuleInputs,
    ModuleNotLoaded,
    ModuleOutput,
    ModuleSpec,
    PreprocessSpec,
    check_preprocess,
)
from mlops.instant import consensus, read

PREPROCESS = PreprocessSpec(width=640, height=384, color="bgr", letterbox=True, normalize="none")
GABARITO = "gabarito"


class AnnotatedShot:
    """Devolve o quadro que uma pessoa marcou, sem depender de bola nem de jogador.

    `needs` vazio de proposito: o detector cinematico depende de `ball`, e uma falha ali mudaria o
    resultado do experimento por um motivo que nao e o que esta sendo medido.
    """

    spec = ModuleSpec(
        kind="shot",
        name="annotated-shot",
        version="0.1.0",
        frames="coarse",
        preprocess=PREPROCESS,
    )

    def __init__(self, gabarito: Mapping[str, int] | None = None) -> None:
        self._gabarito = gabarito
        self._pronto = False

    def load(self) -> None:
        if self._gabarito is None:
            self._gabarito = consensus(read())
        self._pronto = True

    def close(self) -> None:
        self._pronto = False

    def infer(self, batch: FrameBatch, ctx: ModuleInputs) -> ModuleOutput:
        if not self._pronto:
            raise ModuleNotLoaded(self.spec.name)
        check_preprocess(batch, self.spec.preprocess)
        if not batch.indices:
            return ModuleOutput(kind=self.spec.kind, frames=(), records=())
        quadro = (self._gabarito or {}).get(batch.clip_id or "")
        if quadro is None:
            return ModuleOutput(
                kind=self.spec.kind,
                frames=batch.indices,
                records=((),) * len(batch.indices),
            )
        registro: Mapping[str, object] = {
            "frame_index": quadro,
            "confidence": 1.0,
            "source": GABARITO,
        }
        portador = min(range(len(batch.indices)), key=lambda i: abs(batch.indices[i] - quadro))
        return ModuleOutput(
            kind=self.spec.kind,
            frames=batch.indices,
            records=tuple(
                (registro,) if posicao == portador else () for posicao in range(len(batch.indices))
            ),
        )
