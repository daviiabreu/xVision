"""As quatro promessas nao negociaveis do runner, cobradas uma a uma.

O docstring de `graph/runner.py` lista quatro coisas que nao sao negociaveis. A suite cobria o
relatorio delas e nao o efeito: um teste de mutacao mostrou que dava para quebrar cada uma sem
derrubar teste nenhum. Como todas produzem resultado errado com cara de certo, que e o modo de
falha que este codigo inteiro existe para recusar, elas passam a ter teste proprio.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest

from contracts import Clip
from edge.frames.provider import FrameProvider, VideoSource, probe
from edge.graph.framesets import AroundFrames, FrameSetError, read_framesets, resolve
from edge.graph.runner import run_pipeline
from edge.modules.base import FrameBatch, ModuleInputs, ModuleOutput, PerceptionModule
from edge.modules.registry import load_modules, read_config
from edge.testing.make_clip import make_synthetic_clip

CONFIG = Path("config/pipeline.toml")


class SaidaIncompleta:
    """Devolve registro sem as chaves que a especie promete, e o `check_records` recusa.

    O ponto do teste nao e a recusa, que ja tem cobertura, e sim que a saida recusada nao pode
    entrar no dicionario que alimenta as etapas seguintes e a montagem.
    """

    def __init__(self, interno: PerceptionModule) -> None:
        self.interno = interno

    @property
    def spec(self) -> object:
        return self.interno.spec

    def load(self) -> None:
        self.interno.load()

    def close(self) -> None:
        self.interno.close()

    def infer(self, batch: FrameBatch, ctx: ModuleInputs) -> ModuleOutput:
        completa = self.interno.infer(batch, ctx)
        sem_confianca = tuple(
            tuple({k: v for k, v in registro.items() if k != "confidence"} for registro in quadro)
            for quadro in completa.records
        )
        return ModuleOutput(kind=completa.kind, frames=completa.frames, records=sem_confianca)


@pytest.fixture
def clipe(tmp_path: Path) -> Clip:
    return probe(make_synthetic_clip(tmp_path / "s.mp4"), clip_id="clipe-do-teste")


def _rodar(clipe: Clip, modulos: Mapping[str, Any]) -> Any:
    provider = FrameProvider(VideoSource(Path(clipe.path)))
    try:
        return run_pipeline(
            clip=clipe,
            modules=cast("Any", modulos),
            framesets=read_framesets(CONFIG),
            provider=provider,
        )
    finally:
        provider.close()


def test_the_output_of_a_stage_that_failed_never_reaches_the_next_ones(clipe: Clip) -> None:
    """Etapa que falhou reporta `failed` e a saida dela fica de fora, ou o erro vira acerto.

    Com a saida entrando mesmo assim, `assemble` roda com registro malformado e o clipe morre
    fora do `try`, ou pior, mede sobre nada e sai `ok`. E o modo de falha que `records.py` diz
    existir para impedir.
    """
    modulos = dict(load_modules(read_config(CONFIG)))
    modulos["players"] = cast("Any", SaidaIncompleta(modulos["players"]))

    execucao = _rodar(clipe, modulos)

    etapa = next(e for e in execucao.stages if e.kind == "players")
    assert etapa.status == "failed"
    assert "players" not in execucao.outputs, "a saida recusada entrou no contexto das seguintes"


def test_a_window_whose_anchor_produced_nothing_is_refused_instead_of_falling_back(
    clipe: Clip,
) -> None:
    """Ancora ausente nao pode virar o quadro 0 em silencio.

    O teste que existia cobre a ancora que **falhou**. Esta e a ancora que rodou com sucesso e
    nao apontou instante nenhum: a janela abriria em volta do quadro 0, a etapa reportaria `ok`,
    e o `pose` mediria a janela errada da partida.
    """
    janela = AroundFrames(anchor="shot", before_s=1.5, after_s=0.5, fps=None)

    with pytest.raises(FrameSetError, match="shot"):
        resolve(janela, clipe, anchors={})

    with pytest.raises(FrameSetError, match="shot"):
        resolve(janela, clipe, anchors=None)


def test_a_stage_that_failed_does_not_claim_provenance_in_the_event(clipe: Clip) -> None:
    """`versions` e a procedencia, e o schema diz que ela lista o modulo que rodou.

    Com a etapa que falhou dentro dela, o evento que cruza para a nuvem afirma que o modulo
    produziu saida, e quem ler o evento depois nao consegue distinguir execucao completa de
    execucao em que metade dos modulos morreu.
    """
    modulos = dict(load_modules(read_config(CONFIG)))
    modulos["players"] = cast("Any", SaidaIncompleta(modulos["players"]))

    execucao = _rodar(clipe, modulos)

    assert "players" not in execucao.pipeline.versions, "etapa que falhou declarou procedencia"
    assert "players" in execucao.pipeline.timings_ms, "o tempo medido e verdadeiro e deve ficar"
    assert "ball" in execucao.pipeline.versions, "as que rodaram continuam declarando"
