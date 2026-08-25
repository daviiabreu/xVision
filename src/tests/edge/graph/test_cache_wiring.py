"""O cache precisa estar ligado ao pipeline de verdade, e nao so existir e ter teste proprio.

A revisao encontrou o caso: `perception_key`, `implementation_id`, `artifact_digest_of` e
`PerceptionCache` tinham teste, passavam, e nada em producao os chamava. O runner ia direto em
`modulo.infer`, entao `var/cache/perception` ficava vazio depois de uma execucao completa
enquanto o README prometia que a segunda execucao igual nao chamaria modulo nenhum.

Teste de unidade do cache nao pega isso, porque ele chama `run_module` de proposito. So pega
quem entra pela porta da frente, que e o que este arquivo faz.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from contracts import Clip
from edge.cache.perception import PerceptionCache
from edge.frames.provider import FrameProvider, VideoSource, probe
from edge.graph.framesets import read_framesets
from edge.graph.runner import run_pipeline
from edge.modules.base import FrameBatch, ModuleInputs, ModuleOutput, PerceptionModule
from edge.modules.registry import load_modules, read_config
from edge.testing.make_clip import make_synthetic_clip

CONFIG = Path("config/pipeline.toml")


class ModuloEspia:
    """Envolve um modulo de verdade e conta quantas vezes o pipeline chegou a inferir."""

    def __init__(self, interno: PerceptionModule) -> None:
        self.interno = interno
        self.chamadas = 0

    @property
    def spec(self) -> object:
        return self.interno.spec

    def load(self) -> None:
        self.interno.load()

    def close(self) -> None:
        self.interno.close()

    def infer(self, batch: FrameBatch, ctx: ModuleInputs) -> ModuleOutput:
        self.chamadas += 1
        return self.interno.infer(batch, ctx)


@pytest.fixture
def clipe(tmp_path: Path) -> Clip:
    return probe(make_synthetic_clip(tmp_path / "s.mp4"), clip_id="clipe-do-teste")


def _rodar(clipe: Clip, espioes: dict[str, ModuloEspia], cache: PerceptionCache | None) -> None:
    provider = FrameProvider(VideoSource(Path(clipe.path)))
    try:
        run_pipeline(
            clip=clipe,
            modules={kind: espiao for kind, espiao in espioes.items()},  # pyright: ignore[reportArgumentType]
            framesets=read_framesets(CONFIG),
            provider=provider,
            cache=cache,
        )
    finally:
        provider.close()


def _espioes() -> dict[str, ModuloEspia]:
    return {kind: ModuloEspia(modulo) for kind, modulo in load_modules(read_config(CONFIG)).items()}


def test_the_second_identical_run_does_not_reach_any_module(clipe: Clip, tmp_path: Path) -> None:
    """E a promessa que o README faz, e ela precisa valer pela porta da frente."""
    cache = PerceptionCache(tmp_path / "cache")
    primeiro = _espioes()
    _rodar(clipe, primeiro, cache)
    assert sum(e.chamadas for e in primeiro.values()) > 0, "a primeira execucao tem de inferir"

    segundo = _espioes()
    _rodar(clipe, segundo, cache)

    assert sum(e.chamadas for e in segundo.values()) == 0


def test_without_a_cache_the_pipeline_infers_every_time(clipe: Clip) -> None:
    """O parametro e opcional, e sem ele o comportamento e o de antes, sem tocar o disco."""
    primeiro = _espioes()
    _rodar(clipe, primeiro, None)
    segundo = _espioes()
    _rodar(clipe, segundo, None)

    assert sum(e.chamadas for e in segundo.values()) == sum(e.chamadas for e in primeiro.values())


def test_the_run_leaves_the_cache_on_disk(clipe: Clip, tmp_path: Path) -> None:
    """Prova direta do que a revisao mediu como zero depois de uma execucao completa."""
    raiz = tmp_path / "cache"

    _rodar(clipe, _espioes(), PerceptionCache(raiz))

    assert list(raiz.glob("*.json")), "a execucao nao gravou nada no cache"


def test_a_different_clip_does_not_read_the_first_clips_entry(clipe: Clip, tmp_path: Path) -> None:
    """Chave sem `clip_id` serviria a percepcao de um clipe como se fosse a de outro."""
    cache = PerceptionCache(tmp_path / "cache")
    _rodar(clipe, _espioes(), cache)

    outro = _espioes()
    _rodar(probe(Path(clipe.path), clip_id="outro-clipe"), outro, cache)

    assert sum(e.chamadas for e in outro.values()) > 0
