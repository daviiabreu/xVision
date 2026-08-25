"""Tres modulos de exemplo: um que cumpre o contrato, um que viola a propria spec, e um que
quebra ao carregar.

Nao sao os stubs que o pipeline usa. Existem para exercitar a suite de conformidade nos dois
sentidos, aprovando e reprovando, e para dar ao teste da fronteira HTTP uma etapa que falha de
verdade: sem um modulo que quebre, nao ha como afirmar que a etapa falha chega ao frontend como
falha, e cravar `status` em verdadeiro passava na suite inteira.
"""

from __future__ import annotations

from dataclasses import replace

from edge.modules.base import (
    FrameBatch,
    ModuleInputs,
    ModuleNotLoaded,
    ModuleOutput,
    ModuleSpec,
    PreprocessSpec,
    check_preprocess,
)

SPEC = ModuleSpec(
    kind="players",
    name="example",
    version="0.1.0",
    frames="coarse",
    preprocess=PreprocessSpec(width=64, height=64, color="rgb", letterbox=True, normalize="unit"),
)


class ExampleModule:
    spec = SPEC

    def __init__(self) -> None:
        self._threshold: float | None = None

    def load(self) -> None:
        self._threshold = 0.5

    def infer(self, batch: FrameBatch, ctx: ModuleInputs) -> ModuleOutput:
        if self._threshold is None:
            raise ModuleNotLoaded(self.spec.name)
        check_preprocess(batch, self.spec.preprocess)
        records = tuple(
            ({"box": (0.0, 0.0, 1.0, 1.0), "confidence": 1.0},)
            if float(frame.max()) > self._threshold
            else ()
            for frame in batch.tensor
        )
        return ModuleOutput(kind=self.spec.kind, frames=batch.indices, records=records)

    def close(self) -> None:
        self._threshold = None


class ViolatingModule:
    spec = replace(SPEC, name="violating")

    def __init__(self) -> None:
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    def infer(self, batch: FrameBatch, ctx: ModuleInputs) -> ModuleOutput:
        if not self._loaded:
            raise ModuleNotLoaded(self.spec.name)
        return ModuleOutput(
            kind=self.spec.kind, frames=batch.indices, records=((),) * len(batch.indices)
        )

    def close(self) -> None:
        self._loaded = False


class QuebraNoLoad:
    """Falha ao carregar, que e o modo de falha mais comum de um modelo de verdade.

    Peso ausente, biblioteca de GPU incompativel, memoria insuficiente: nos tres o modulo morre
    em `load` e nao chega a inferir. A especie e `players` porque o registro exige que a `kind` da
    classe bata com a secao da configuracao, entao ela so pode substituir `modules.players`.
    """

    spec = replace(SPEC, name="quebra-no-load")

    def load(self) -> None:
        raise ModuleNotLoaded(f"{self.spec.name}: o peso nao pode ser carregado")

    def infer(self, batch: FrameBatch, ctx: ModuleInputs) -> ModuleOutput:
        raise ModuleNotLoaded(self.spec.name)

    def close(self) -> None: ...
