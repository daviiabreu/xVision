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

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from contracts import Clip, Perception, ShotEvent
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


@dataclass(frozen=True)
class ClipResult:
    """O que o clipe produziu, nomeado em vez de posicional.

    Ate aqui isto era uma tupla de dois, e o terceiro campo mostrou por que isso nao escala: a
    procedencia do instante, se ele foi medido na trajetoria ou estimado no ponto medio, e a
    diferenca entre um numero e um preenchimento, e nao cabia em lugar nenhum do evento. Tupla que
    cresce faz todo chamador desempacotar de novo e nao diz o que o campo novo significa.
    """

    event: ShotEvent
    stages: tuple[StageRun, ...]
    shot_source: str | None
    ball_support: str | None
    perception: Perception


class _Shot:
    """O contato que as features recebem, com a procedencia do proprio registro de `shot`.

    A procedencia atravessa por aqui porque `run_features` so passa `perception` e `shot`, e sem
    ela uma feature que le a trajetoria antes do contato nao consegue distinguir janela ancorada em
    contato medido de janela ancorada em ponto medio estimado. As duas parecem iguais e uma delas
    nao tem um antes.
    """

    def __init__(
        self,
        frame_index: int,
        shooter_track_id: int | None,
        *,
        source: str | None = None,
        confidence: float | None = None,
        ball_support: str | None = None,
    ) -> None:
        self._frame_index = frame_index
        self._shooter_track_id = shooter_track_id
        self._source = source
        self._confidence = confidence
        self._ball_support = ball_support

    @property
    def frame_index(self) -> int:
        return self._frame_index

    @property
    def shooter_track_id(self) -> int | None:
        return self._shooter_track_id

    @property
    def source(self) -> str | None:
        return self._source

    @property
    def confidence(self) -> float | None:
        return self._confidence

    @property
    def ball_support(self) -> str | None:
        return self._ball_support


def process_clip(
    clip: Clip,
    modules: Mapping[str, PerceptionModule],
    framesets: Mapping[str, FrameSet],
    provider: FrameSource,
    extractors: Sequence[FeatureExtractor],
    cache: PerceptionCache | None = None,
    on_perception: Callable[[Perception], None] | None = None,
    on_stage_start: Callable[[str, int, int], None] | None = None,
) -> ClipResult:
    execucao = run_pipeline(
        clip=clip,
        modules=modules,
        framesets=framesets,
        provider=provider,
        cache=cache,
        on_stage_start=on_stage_start,
    )
    _check_assembly(execucao)
    if on_perception is not None:
        on_perception(execucao.perception)
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
    return ClipResult(
        event=evento,
        stages=execucao.stages,
        shot_source=_source(execucao, "source"),
        ball_support=_source(execucao, "ball_support"),
        perception=execucao.perception,
    )


def _source(execucao: RunResult, campo: str) -> str | None:
    """Um campo textual do registro de instante, para a execucao poder separar clipe de clipe.

    Sem `source`, cem clipes que cairam no ponto medio e cem que mediram a trajetoria chegam ao
    relatorio com a mesma cara, e a cobertura do card parece o dobro do que e. Sem `ball_support`, a
    mesma coisa acontece um nivel abaixo: um instante decidido sobre posicao interpolada chega igual
    a um decidido sobre posicao medida, e nenhum agregado separa os dois.
    """
    saida = execucao.outputs.get(SHOT)
    if saida is None:
        return None
    for registros in saida.records:
        for registro in registros:
            valor = registro.get(campo)
            if isinstance(valor, str):
                return valor
    return None


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
    return _Shot(
        quadros.pop(),
        None if atirador is None else int(cast("Any", atirador)),
        source=_texto(registro, "source"),
        confidence=_numero(registro, "confidence"),
        ball_support=_texto(registro, "ball_support"),
    )


def _texto(registro: Mapping[str, object], campo: str) -> str | None:
    valor = registro.get(campo)
    return valor if isinstance(valor, str) else None


def _numero(registro: Mapping[str, object], campo: str) -> float | None:
    """Numero real e finito, ou ausencia. `json.loads` aceita `NaN` por padrao, e um `nan` que
    atravessa vira confianca maxima ou logit envenenado no consumidor."""
    valor = registro.get(campo)
    if isinstance(valor, bool) or not isinstance(valor, int | float):
        return None
    if not math.isfinite(valor):
        return None
    return float(valor)


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
