"""Executa o grafo declarado e devolve `Perception` mais `PipelineMetadata`.

A procedencia do evento so lista modulo que rodou, que e o que o schema diz e o que era falso:
etapa que levantou excecao entrava em `versions` igual as outras, e o evento que cruza para a
nuvem afirmava que ela produziu saida. Quem lesse o evento depois nao teria como distinguir uma
execucao completa de uma em que metade dos modulos morreu. O tempo continua sendo registrado para
todas, inclusive as que falharam, porque medida e verdadeira nos dois casos e saber que a etapa
gastou 40 ms antes de morrer e informacao.

Quatro coisas nao sao negociaveis, e cada uma vira uma linha aqui. O decode e unico e passa
pelo `FrameProvider`, entao nenhum modulo abre video. A degradacao e declarada: modulo que
levanta vira etapa com motivo, e quem dependia dele e pulado com o motivo, em vez de rodar com
entrada vazia e nao conseguir distinguir isso de ausencia de deteccao. O tempo de cada etapa e
medido, inclusive o das que falharam. E coordenada em metros so sobrevive onde o modulo de campo
produziu homografia, porque o contrato recusa `pitch_xy` sem ela e cair para pixel calado faria a
feature medir noutra unidade sem ninguem notar.
"""

from __future__ import annotations

import platform
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Protocol, cast

from contracts import (
    BallDetection,
    Box,
    Clip,
    FieldGeometry,
    FramePerception,
    Perception,
    PipelineMetadata,
    PlayerDetection,
)
from edge.cache.perception import PerceptionCache, run_module
from edge.graph.framesets import AroundFrames, FrameSet, FrameSetError, resolve
from edge.graph.order import topological_order
from edge.modules.base import (
    MODULE_KINDS,
    FrameBatch,
    ModuleOutput,
    ModuleSpec,
    PerceptionModule,
    PerFrame,
    PreprocessSpec,
    Temporal,
)
from edge.modules.records import check_records

StageStatus = Literal["ok", "failed", "skipped"]


class FrameSource(Protocol):
    def frames(self, indices: Sequence[int], spec: PreprocessSpec) -> FrameBatch: ...


@dataclass(frozen=True)
class StageRun:
    kind: str
    name: str
    version: str
    frames: str
    status: StageStatus
    reason: str | None
    duration_ms: float


@dataclass(frozen=True)
class RunResult:
    perception: Perception
    pipeline: PipelineMetadata
    outputs: Mapping[str, ModuleOutput]
    stages: tuple[StageRun, ...]


def run_pipeline(
    *,
    clip: Clip,
    modules: Mapping[str, PerceptionModule],
    framesets: Mapping[str, FrameSet],
    provider: FrameSource,
    cache: PerceptionCache | None = None,
) -> RunResult:
    _refuse_unknown_framesets(modules, framesets)
    arestas = _edges(modules, framesets)
    ordem = topological_order(arestas)

    agora = datetime.now(timezone.utc)
    comecou = agora.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    saidas: dict[str, ModuleOutput] = {}
    etapas: list[StageRun] = []
    for kind in ordem:
        etapa, saida = _run_stage(
            modules[kind], arestas[kind], clip, framesets, provider, saidas, cache
        )
        etapas.append(etapa)
        if saida is not None:
            saidas[kind] = saida

    percepcao, falha_montagem = _assemble_or_report(clip, saidas)
    if falha_montagem is not None:
        etapas.append(falha_montagem)

    return RunResult(
        perception=percepcao,
        pipeline=PipelineMetadata(
            versions={
                etapa.kind: f"{etapa.name}@{etapa.version}"
                for etapa in etapas
                if etapa.kind in MODULE_KINDS and etapa.status == "ok"
            },
            timings_ms={
                etapa.kind: etapa.duration_ms for etapa in etapas if etapa.kind in MODULE_KINDS
            },
            host=platform.node(),
            started_at=comecou,
        ),
        outputs=saidas,
        stages=tuple(etapas),
    )


def _edges(
    modules: Mapping[str, PerceptionModule], framesets: Mapping[str, FrameSet]
) -> dict[str, tuple[str, ...]]:
    arestas: dict[str, tuple[str, ...]] = {}
    for kind, modulo in modules.items():
        conjunto = framesets[modulo.spec.frames]
        ancora = (conjunto.anchor,) if isinstance(conjunto, AroundFrames) else ()
        arestas[kind] = tuple(dict.fromkeys(modulo.spec.needs + ancora))
    return arestas


def _refuse_unknown_framesets(
    modules: Mapping[str, PerceptionModule], framesets: Mapping[str, FrameSet]
) -> None:
    for kind, modulo in modules.items():
        if modulo.spec.frames not in framesets:
            raise FrameSetError(
                f"modules.{kind} roda em {modulo.spec.frames!r}, que nao esta declarado em [frames]"
            )


def _run_stage(
    modulo: PerceptionModule,
    depende_de: tuple[str, ...],
    clip: Clip,
    framesets: Mapping[str, FrameSet],
    provider: FrameSource,
    saidas: Mapping[str, ModuleOutput],
    cache: PerceptionCache | None,
) -> tuple[StageRun, ModuleOutput | None]:
    spec = modulo.spec
    faltando = tuple(kind for kind in depende_de if kind not in saidas)
    if faltando:
        return _stage(spec, "skipped", f"{', '.join(faltando)} nao produziu saida", 0.0), None

    inicio = time.perf_counter()
    falha: str | None = None
    saida: ModuleOutput | None = None
    try:
        modulo.load()
        indices = resolve(
            framesets[spec.frames], clip, _anchors(spec, framesets, saidas, clip.frame_count)
        )
        saida = _infer(modulo, provider, indices, saidas, cache, clip.clip_id)
        check_records(saida)
    except Exception as erro:
        falha = f"{type(erro).__name__}: {erro}"
    try:
        modulo.close()
    except Exception as erro:
        falha = falha or f"{type(erro).__name__}: {erro}"
        saida = None
    decorrido = (time.perf_counter() - inicio) * 1000
    if falha is not None:
        return _stage(spec, "failed", falha, decorrido), None
    return _stage(spec, "ok", None, decorrido), saida


def _infer(
    modulo: PerceptionModule,
    provider: FrameSource,
    indices: tuple[int, ...],
    saidas: Mapping[str, ModuleOutput],
    cache: PerceptionCache | None,
    clip_id: str,
) -> ModuleOutput:
    """Com cache, o lote ja computado volta do disco em vez de passar pelo modulo de novo.

    Sem o `cache` aqui, `edge.cache` inteiro seria infraestrutura sem consumidor de producao: a
    chave, o digest do peso e o `PerceptionCache` existiam e nada os chamava fora dos testes,
    entao `var/cache/perception` ficava vazio depois de uma execucao completa.
    """
    spec = modulo.spec
    ctx = {kind: saidas[kind] for kind in spec.needs}
    lotes = [provider.frames(lote, spec.preprocess) for lote in _batches(indices, spec.batching)]
    if cache is None:
        return _merge(spec.kind, [modulo.infer(lote, ctx) for lote in lotes])
    return _merge(
        spec.kind,
        [run_module(modulo, lote, ctx, cache=cache, clip_id=clip_id) for lote in lotes],
    )


def _batches(
    indices: tuple[int, ...], batching: PerFrame | Temporal
) -> tuple[tuple[int, ...], ...]:
    if isinstance(batching, PerFrame):
        return (indices,)
    if not indices:
        return ()
    primeiro, ultimo = indices[0], indices[-1]
    tamanho = min(batching.size, ultimo - primeiro + 1)
    inicios = {
        min(max(primeiro, indice), ultimo - tamanho + 1) for indice in indices[:: batching.stride]
    }
    return tuple(tuple(range(inicio, inicio + tamanho)) for inicio in sorted(inicios))


def _merge(kind: str, partes: Sequence[ModuleOutput]) -> ModuleOutput:
    quadros: list[int] = []
    registros: list[tuple[Mapping[str, object], ...]] = []
    for parte in partes:
        for indice, registro in zip(parte.frames, parte.records, strict=True):
            if indice not in quadros:
                quadros.append(indice)
                registros.append(registro)
    return ModuleOutput(kind=kind, frames=tuple(quadros), records=tuple(registros))


def _anchors(
    spec: ModuleSpec,
    framesets: Mapping[str, FrameSet],
    saidas: Mapping[str, ModuleOutput],
    frame_count: int,
) -> dict[str, int]:
    conjunto = framesets[spec.frames]
    if not isinstance(conjunto, AroundFrames):
        return {}
    saida = saidas[conjunto.anchor]
    for indice, registros in zip(saida.frames, saida.records, strict=True):
        if registros:
            ancora = _anchor_frame(conjunto.anchor, registros[0], indice, frame_count)
            return {conjunto.anchor: ancora}
    return {}


def _anchor_frame(kind: str, registro: Mapping[str, object], indice: int, frame_count: int) -> int:
    valor = registro.get("frame_index", indice)
    if not isinstance(valor, int) or isinstance(valor, bool):
        raise FrameSetError(
            f"o modulo {kind} devolveu frame_index {valor!r}, e a ancora precisa de inteiro"
        )
    if not 0 <= valor < frame_count:
        raise FrameSetError(
            f"o modulo {kind} devolveu frame_index {valor}, fora dos {frame_count} quadros do "
            "clipe. A janela sairia deslocada e a etapa passaria como boa"
        )
    return valor


def _stage(spec: ModuleSpec, status: StageStatus, reason: str | None, ms: float) -> StageRun:
    return StageRun(
        kind=spec.kind,
        name=spec.name,
        version=spec.version,
        frames=spec.frames,
        status=status,
        reason=reason,
        duration_ms=round(ms, 3),
    )


def _assemble_or_report(
    clip: Clip, saidas: Mapping[str, ModuleOutput]
) -> tuple[Perception, StageRun | None]:
    try:
        return assemble(clip, saidas), None
    except Exception as erro:
        etapa = StageRun(
            kind="assemble",
            name="assemble",
            version="0",
            frames="",
            status="failed",
            reason=f"{type(erro).__name__}: {erro}",
            duration_ms=0.0,
        )
        return Perception(clip_id=clip.clip_id, frames=()), etapa


def assemble(clip: Clip, saidas: Mapping[str, ModuleOutput]) -> Perception:
    por_quadro: dict[str, dict[int, tuple[Mapping[str, object], ...]]] = {
        kind: dict(zip(saida.frames, saida.records, strict=True)) for kind, saida in saidas.items()
    }
    indices = sorted({indice for mapa in por_quadro.values() for indice in mapa})
    return Perception(
        clip_id=clip.clip_id,
        frames=tuple(_frame(clip, indice, por_quadro) for indice in indices),
    )


def _frame(
    clip: Clip,
    indice: int,
    por_quadro: Mapping[str, Mapping[int, tuple[Mapping[str, object], ...]]],
) -> FramePerception:
    campo = _field(por_quadro.get("field", {}).get(indice, ()))
    calibrado = campo is not None and campo.homography is not None
    trajetorias = {
        int(cast("Any", r["detection_index"])): int(cast("Any", r["track_id"]))
        for r in por_quadro.get("tracking", {}).get(indice, ())
        if "detection_index" in r
    }
    camisas = {
        int(cast("Any", r["track_id"])): int(cast("Any", r["jersey_number"]))
        for r in por_quadro.get("identity", {}).get(indice, ())
        if "jersey_number" in r
    }
    jogadores = tuple(
        _player(registro, trajetorias.get(posicao), camisas, calibrado)
        for posicao, registro in enumerate(por_quadro.get("players", {}).get(indice, ()))
    )
    return FramePerception(
        frame_index=indice,
        timestamp_s=indice / clip.fps,
        players=jogadores,
        ball=_ball(por_quadro.get("ball", {}).get(indice, ()), calibrado),
        field=campo,
    )


def _player(
    registro: Mapping[str, object],
    track_id: int | None,
    camisas: Mapping[int, int],
    calibrado: bool,
) -> PlayerDetection:
    return PlayerDetection(
        box=Box(*cast("Any", registro["box"])),
        confidence=float(cast("Any", registro["confidence"])),
        track_id=track_id,
        team=cast("Any", registro.get("team")),
        role=cast("Any", registro.get("role")),
        jersey_number=camisas.get(track_id) if track_id is not None else None,
        pitch_xy=_pitch(registro, calibrado),
    )


def _ball(registros: tuple[Mapping[str, object], ...], calibrado: bool) -> BallDetection | None:
    if not registros:
        return None
    registro = registros[0]
    return BallDetection(
        box=Box(*cast("Any", registro["box"])),
        confidence=float(cast("Any", registro["confidence"])),
        pitch_xy=_pitch(registro, calibrado),
    )


def _field(registros: tuple[Mapping[str, object], ...]) -> FieldGeometry | None:
    if not registros:
        return None
    registro = registros[0]
    return FieldGeometry(
        homography=_tuple_of_floats(registro.get("homography")),
        reprojection_error_px=_optional_float(registro.get("reprojection_error_px")),
        keypoints=tuple(
            (str(nome), float(u), float(v))
            for nome, u, v in cast("Any", registro.get("keypoints", ()))
        ),
    )


def _pitch(registro: Mapping[str, object], calibrado: bool) -> tuple[float, float] | None:
    valor = registro.get("pitch_xy")
    if not calibrado or valor is None:
        return None
    x, y = cast("Any", valor)
    return (float(x), float(y))


def _tuple_of_floats(valor: object) -> tuple[float, ...] | None:
    return None if valor is None else tuple(float(v) for v in cast("Any", valor))


def _optional_float(valor: object) -> float | None:
    return None if valor is None else float(cast("Any", valor))
