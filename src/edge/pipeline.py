"""A cola que vai do clipe ao evento, entregue no pacote e nao dentro de um arquivo de teste.

Ela roda o grafo, le da especie `shot` o quadro do contato, monta o `ShotOutput` que as features
consomem, chama a selecao declarada e devolve o evento mais as etapas. Ate aqui isso existia so
dentro de `src/tests/edge/graph/test_integration.py`, com duas consequencias: mudanca de
assinatura no runner nao quebrava nada visivel, e a demonstracao da sprint dependia de codigo de
teste.

Nao abre video, quem abre e o `FrameSource`. Nao decide quais features rodam, recebe a sequencia
pronta. Nao decide `clip_id`, que ja vem no `Clip`. Nao le CSV, nao sabe o que e particao e nao
escreve relatorio.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from contracts import Clip, ShotEvent
from contracts.events import make_shot_id
from contracts.schema import SCHEMA_VERSION
from edge.cache.perception import PerceptionCache
from edge.features.base import FeatureExtractor, run_features
from edge.graph.framesets import FrameSet
from edge.graph.runner import FrameSource, RunResult, StageRun, run_pipeline
from edge.modules.base import PerceptionModule

SHOT = "shot"
FRAME_INDEX = "frame_index"
SHOOTER = "shooter_track_id"
ASSEMBLE = "assemble"


class PipelineError(Exception): ...


class NoShotDetected(PipelineError): ...


class AmbiguousContact(PipelineError): ...


class ContactOutOfRange(PipelineError): ...


class AssemblyFailed(PipelineError): ...


class _Shot:
    def __init__(self, frame_index: int, shooter_track_id: int | None) -> None:
        self._frame_index = frame_index
        self._shooter_track_id = shooter_track_id

    @property
    def frame_index(self) -> int:
        return self._frame_index

    @property
    def shooter_track_id(self) -> int | None:
        return self._shooter_track_id


def process_clip(
    clip: Clip,
    modules: Mapping[str, PerceptionModule],
    framesets: Mapping[str, FrameSet],
    provider: FrameSource,
    extractors: Sequence[FeatureExtractor],
    cache: PerceptionCache | None = None,
) -> tuple[ShotEvent, tuple[StageRun, ...]]:
    execucao = run_pipeline(
        clip=clip, modules=modules, framesets=framesets, provider=provider, cache=cache
    )
    _check_assembly(execucao)
    contato = _contact(execucao, clip)
    evento = ShotEvent(
        schema_version=SCHEMA_VERSION,
        shot_id=make_shot_id(clip.clip_id, contato.frame_index),
        clip_id=clip.clip_id,
        match_id=clip.match_id,
        shot_timestamp_s=contato.frame_index / clip.fps,
        features=run_features(extractors, execucao.perception, contato, execucao.outputs),
        pipeline=execucao.pipeline,
    )
    return evento, execucao.stages


def _check_assembly(execucao: RunResult) -> None:
    falha = next((e for e in execucao.stages if e.kind == ASSEMBLE), None)
    if falha is not None:
        raise AssemblyFailed(
            f"a percepcao nao foi montada: {falha.reason}. Sem ela as features mediriam sobre "
            "nada e sairiam `ok`, entao o evento nao e produzido"
        )


def _contact(execucao: RunResult, clip: Clip) -> _Shot:
    saida = execucao.outputs.get(SHOT)
    if saida is None:
        motivo = next(
            (e.reason for e in execucao.stages if e.kind == SHOT and e.reason), "nao executou"
        )
        raise NoShotDetected(f"a especie {SHOT!r} nao produziu saida: {motivo}")
    encontrados = [registro for registros in saida.records for registro in registros]
    if not encontrados:
        raise NoShotDetected(
            f"a especie {SHOT!r} rodou e nao apontou instante de contato em nenhum quadro"
        )
    quadros = {_frame_index(registro, clip) for registro in encontrados}
    if len(quadros) > 1:
        raise AmbiguousContact(
            f"a especie {SHOT!r} apontou {len(quadros)} instantes de contato, "
            f"{sorted(quadros)}, e o evento descreve uma finalizacao so"
        )
    registro = encontrados[0]
    atirador = registro.get(SHOOTER)
    return _Shot(quadros.pop(), None if atirador is None else int(cast("Any", atirador)))


def _frame_index(registro: Mapping[str, object], clip: Clip) -> int:
    valor = registro.get(FRAME_INDEX)
    if not isinstance(valor, int) or isinstance(valor, bool):
        raise ContactOutOfRange(
            f"a especie {SHOT!r} devolveu {FRAME_INDEX} {valor!r}, e o instante precisa de inteiro"
        )
    if not 0 <= valor < clip.frame_count:
        raise ContactOutOfRange(
            f"a especie {SHOT!r} devolveu {FRAME_INDEX} {valor}, fora dos {clip.frame_count} "
            "quadros do clipe"
        )
    return valor
