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
    ModuleError,
    ModuleInputs,
    ModuleNotLoaded,
    ModuleOutput,
    ModuleSpec,
    PreprocessSpec,
    check_preprocess,
)
from edge.testing.pitch import FPS, HEIGHT, WIDTH, ClipTruth, clip_truth

STUB_VERSION = "0.1.1"


class UnexpectedClip(ModuleError):
    """O lote contradiz a geometria da fixture que o stub representa."""


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
        _reject_foreign_clip(self.spec.name, batch)
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


def _reject_foreign_clip(name: str, batch: FrameBatch) -> None:
    """Recusa outra geometria; lotes sinteticos de conformidade omitem os metadados."""
    if batch.source_size is not None and batch.source_size != (WIDTH, HEIGHT):
        raise UnexpectedClip(
            f"{name} serve a fixture {WIDTH}x{HEIGHT}; "
            f"o lote veio de {batch.source_size[0]}x{batch.source_size[1]}"
        )
    if batch.source_fps is not None and round(batch.source_fps, 3) != float(FPS):
        raise UnexpectedClip(
            f"{name} serve a fixture a {FPS} fps; o lote veio a {batch.source_fps:.3f} fps"
        )
