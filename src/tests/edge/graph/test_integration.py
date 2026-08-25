"""O fluxo de ponta a ponta com os sete stubs, que e o que sustenta a demonstracao da sprint.

Este teste esperava o runner existir: sem o grafo montado nao havia como ir do
clipe ao evento. Ele nao usa duplo nenhum: abre a fixture canonica, decodifica com o
`FrameProvider`, roda os sete stubs pela configuracao de `config/pipeline.toml` e valida o
evento contra o `shot_event.v1.json`.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest

from contracts import (
    SCHEMA_VERSION,
    Clip,
    ShotEvent,
    make_shot_id,
    validate_shot_event,
)
from edge.features import run_features
from edge.frames.provider import FrameProvider, VideoSource
from edge.graph.framesets import read_framesets
from edge.graph.runner import RunResult, run_pipeline
from edge.modules.base import (
    MODULE_KINDS,
    FrameBatch,
    ModuleError,
    ModuleInputs,
    ModuleOutput,
    ModuleSpec,
)
from edge.modules.registry import load_modules, read_config
from edge.testing.make_clip import make_synthetic_clip
from edge.testing.pitch import SHOOTER_TRACK_ID, SHOT_FRAME_INDEX, clip_truth

CONFIG = Path("config/pipeline.toml")


@pytest.fixture(scope="module")
def execucao(tmp_path_factory: pytest.TempPathFactory) -> RunResult:
    gabarito = clip_truth()
    caminho = make_synthetic_clip(tmp_path_factory.mktemp("clipe") / "synthetic.mp4")
    clip = Clip(
        clip_id="clipe-integracao",
        path=str(caminho),
        fps=float(gabarito.fps),
        frame_count=gabarito.frame_count,
        width=gabarito.width,
        height=gabarito.height,
    )
    provider = FrameProvider(VideoSource(caminho))
    try:
        return run_pipeline(
            clip=clip,
            modules=load_modules(read_config(CONFIG)),
            framesets=read_framesets(CONFIG),
            provider=provider,
        )
    finally:
        provider.close()


def test_the_default_configuration_runs_the_seven_kinds_without_a_single_model(
    execucao: RunResult,
) -> None:
    assert {etapa.kind for etapa in execucao.stages} == MODULE_KINDS
    assert all(etapa.status == "ok" for etapa in execucao.stages)


def test_the_declared_cascade_runs_the_window_after_the_shot(execucao: RunResult) -> None:
    ordem = [etapa.kind for etapa in execucao.stages]
    grossos = [etapa.kind for etapa in execucao.stages if etapa.frames == "coarse"]

    assert ordem.index("shot") < min(ordem.index(kind) for kind in ("pose", "identity", "tracking"))
    assert set(grossos) == {"players", "ball", "field", "shot"}


def test_every_stage_declares_itself_as_stub_in_pipeline_versions(execucao: RunResult) -> None:
    assert set(execucao.pipeline.versions) == MODULE_KINDS
    assert all(versao.startswith("stub-") for versao in execucao.pipeline.versions.values())
    assert set(execucao.pipeline.timings_ms) == MODULE_KINDS


def test_full_pipeline_with_all_stubs_produces_a_schema_valid_event(execucao: RunResult) -> None:
    gabarito = clip_truth()
    quadro = next(q for q in execucao.perception.frames if q.frame_index == SHOT_FRAME_INDEX)
    contato = execucao.outputs["shot"]
    (registro,) = next(registros for registros in contato.records if registros)

    evento = ShotEvent(
        schema_version=SCHEMA_VERSION,
        shot_id=make_shot_id("clipe-integracao", SHOT_FRAME_INDEX),
        clip_id="clipe-integracao",
        match_id=None,
        shot_timestamp_s=quadro.timestamp_s,
        features=run_features((), execucao.perception, _shot(registro), execucao.outputs),
        pipeline=execucao.pipeline,
    )

    validate_shot_event(evento.to_dict())
    assert registro["shooter_track_id"] == SHOOTER_TRACK_ID
    assert quadro.field is not None
    assert quadro.field.homography == gabarito.homography
    assert quadro.ball is not None
    assert quadro.ball.pitch_xy == gabarito.ball_pitch_xy[SHOT_FRAME_INDEX]


def test_the_platform_runs_end_to_end_without_a_single_feature_installed(
    execucao: RunResult,
) -> None:
    """Sem card de modelo, o fluxo do clipe ao evento tem que continuar inteiro.

    O `shot` que roda aqui e um stub, e a selecao de features esta vazia porque nenhuma foi
    escrita. O que existe e o caminho: quem instalar o primeiro extrator nao muda nenhuma linha
    destas.
    """
    (registro,) = next(registros for registros in execucao.outputs["shot"].records if registros)

    valores = run_features((), execucao.perception, _shot(registro), execucao.outputs)

    assert valores == ()
    assert [etapa.status for etapa in execucao.stages] == ["ok"] * len(execucao.stages)


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


def _shot(registro: Mapping[str, object]) -> _Shot:
    return _Shot(
        int(cast("Any", registro["frame_index"])),
        int(cast("Any", registro["shooter_track_id"])),
    )


def test_a_broken_module_still_produces_an_event_that_validates(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    gabarito = clip_truth()
    caminho = make_synthetic_clip(tmp_path_factory.mktemp("quebrado") / "synthetic.mp4")
    clip = Clip(
        clip_id="clipe-quebrado",
        path=str(caminho),
        fps=float(gabarito.fps),
        frame_count=gabarito.frame_count,
        width=gabarito.width,
        height=gabarito.height,
    )
    modules = load_modules(read_config(CONFIG))
    modules["field"] = _QuebraNoLoad(modules["field"].spec)

    provider = FrameProvider(VideoSource(caminho))
    try:
        execucao = run_pipeline(
            clip=clip,
            modules=modules,
            framesets=read_framesets(CONFIG),
            provider=provider,
        )
    finally:
        provider.close()

    campo = next(etapa for etapa in execucao.stages if etapa.kind == "field")
    assert campo.status == "failed"
    assert all(quadro.field is None for quadro in execucao.perception.frames)
    assert all(
        jogador.pitch_xy is None
        for quadro in execucao.perception.frames
        for jogador in quadro.players
    )

    evento = ShotEvent(
        schema_version=SCHEMA_VERSION,
        shot_id=make_shot_id("clipe-quebrado", SHOT_FRAME_INDEX),
        clip_id="clipe-quebrado",
        match_id=None,
        shot_timestamp_s=SHOT_FRAME_INDEX / gabarito.fps,
        features=(),
        pipeline=execucao.pipeline,
    )
    validate_shot_event(evento.to_dict())

    assert "field" not in execucao.pipeline.versions, (
        "`pipeline.versions` diz qual modulo rodou, e este levantou no `load` sem rodar nunca. "
        "Declarar procedencia dele faria o evento que cruza para a nuvem afirmar que ele "
        "produziu saida, e ninguem que lesse o evento depois conseguiria distinguir execucao "
        "completa de execucao em que metade dos modulos morreu"
    )
    assert "field" in execucao.pipeline.timings_ms, (
        "o tempo gasto antes de morrer e medida verdadeira e continua registrado"
    )


class _QuebraNoLoad:
    def __init__(self, spec: ModuleSpec) -> None:
        self.spec = spec

    def load(self) -> None:
        raise ModuleError("o peso do campo nao carregou")

    def infer(self, batch: FrameBatch, ctx: ModuleInputs) -> ModuleOutput:
        raise AssertionError("nao deveria ser chamado")

    def close(self) -> None: ...
