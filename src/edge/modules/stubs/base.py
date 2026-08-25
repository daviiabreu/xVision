"""O esqueleto que os sete stubs dividem.

Todo stub responde a partir do gabarito da fixture, indexado pelo numero do quadro, e nunca
a partir dos pixels. Precisa ser assim para a suite de conformidade passar: ela monta lotes
sinteticos com todos os pixels iguais, onde nao existe nada para detectar, e cobra que duas
chamadas com a mesma entrada devolvam a mesma saida.

Nenhum stub declara `needs`. Ele nao consome outro modulo, le o gabarito. Quem quiser
exercitar dependencia no grafo declara `needs` na entrada `[modules.<especie>]` da
configuracao, que `edge.modules.registry.load_modules` aplica por cima da spec.
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
from edge.testing.pitch import ClipTruth, clip_truth

STUB_VERSION = "0.1.0"

Record = Mapping[str, object]


def stub_spec(kind: str, preprocess: PreprocessSpec, frames: str = "coarse") -> ModuleSpec:
    return ModuleSpec(
        kind=kind,
        name=f"stub-{kind}",
        version=STUB_VERSION,
        frames=frames,
        preprocess=preprocess,
    )


class StubModule:
    spec: ModuleSpec

    def __init__(self) -> None:
        self._truth: ClipTruth | None = None

    def load(self) -> None:
        self._truth = clip_truth()

    def infer(self, batch: FrameBatch, ctx: ModuleInputs) -> ModuleOutput:
        if self._truth is None:
            raise ModuleNotLoaded(self.spec.name)
        check_preprocess(batch, self.spec.preprocess)
        return ModuleOutput(
            kind=self.spec.kind,
            frames=batch.indices,
            records=tuple(self._at(self._truth, indice) for indice in batch.indices),
        )

    def close(self) -> None:
        self._truth = None

    def records(self, truth: ClipTruth, frame_index: int) -> tuple[Record, ...]:
        raise NotImplementedError  # pragma: no cover - as sete subclasses sobrescrevem

    def _at(self, truth: ClipTruth, frame_index: int) -> tuple[Record, ...]:
        if not 0 <= frame_index < truth.frame_count:
            return ()
        return self.records(truth, frame_index)
