"""Prova ao vivo do decode unico e do cache de percepcao, em duas execucoes do mesmo clipe.

A execucao fria conta as chamadas ao modulo; a quente precisa contar zero. Trocar
`--module-version` mostra a invalidacao acontecendo.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from dataclasses import dataclass, replace
from pathlib import Path

from edge.cache.perception import PerceptionCache, run_module
from edge.frames.provider import FrameProvider, VideoSource
from edge.modules.base import FrameBatch, ModuleInputs, ModuleOutput
from edge.testing.example_module import ExampleModule


class ModuloContado(ExampleModule):
    def __init__(self) -> None:
        super().__init__()
        self.execucoes = 0

    def infer(self, batch: FrameBatch, ctx: ModuleInputs) -> ModuleOutput:
        self.execucoes += 1
        return super().infer(batch, ctx)


@dataclass(frozen=True)
class Execucao:
    modulo: str
    quadros: int
    chamadas_ao_modulo: int
    tempo_ms: float

    def __str__(self) -> str:
        return (
            f"modulo={self.modulo} quadros={self.quadros} "
            f"chamadas_ao_modulo={self.chamadas_ao_modulo} tempo_ms={self.tempo_ms:.1f}"
        )


def clip_id(clip: Path) -> str:
    digest = hashlib.sha256()
    with clip.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1 << 20), b""):
            digest.update(bloco)
    return digest.hexdigest()


def run(clip: Path, frames: int, module_version: str, cache_root: Path | None) -> Execucao:
    modulo = ModuloContado()
    modulo.spec = replace(modulo.spec, version=module_version)
    modulo.load()
    provider = FrameProvider(VideoSource(clip))

    inicio = time.perf_counter()
    try:
        lote = provider.frames(tuple(range(frames)), modulo.spec.preprocess)
        saida = run_module(
            modulo, lote, {}, cache=PerceptionCache(cache_root), clip_id=clip_id(clip)
        )
    finally:
        provider.close()
        modulo.close()

    return Execucao(
        modulo=f"{modulo.spec.name}@{modulo.spec.version}",
        quadros=len(saida.frames),
        chamadas_ao_modulo=modulo.execucoes,
        tempo_ms=(time.perf_counter() - inicio) * 1000,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--module-version", default="0.1.0")
    parser.add_argument("--cache", type=Path, default=None)
    arguments = parser.parse_args()

    print(run(arguments.clip, arguments.frames, arguments.module_version, arguments.cache))


if __name__ == "__main__":
    main()
