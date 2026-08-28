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
from edge.testing.pitch import SHOT_FRAME_INDEX, clip_truth

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


def test_the_shot_module_lands_on_the_frame_the_fixture_declares(execucao: RunResult) -> None:
    """A fixture sintetica e o unico lugar do projeto onde o instante tem gabarito exato.

    Ela nao substitui a validacao contra clipe real, que precisa de rotulo manual porque a marcacao
    do parceiro tem grao de 1 s e o corte acrescenta resíduo de ate 8 s. Mas ela prova que a
    geometria mede o que diz medir quando a trajetoria e conhecida.
    """
    (registro,) = next(registros for registros in execucao.outputs["shot"].records if registros)

    assert registro["source"] == "ball-break"
    assert registro["frame_index"] == SHOT_FRAME_INDEX


def test_the_window_frames_carry_track_id_keypoints_and_pitch_xy(execucao: RunResult) -> None:
    """A juncao por quadro so fecha porque `players`, `tracking`, `pose` e `field` dividem conjunto.

    Com `players` no `coarse` e o resto na janela, um quadro montado tinha jogador sem `track_id`,
    sem keypoint e sem `pitch_xy`, e nada no relatorio dizia isso.
    """
    janela = [q for q in execucao.perception.frames if q.players]

    assert janela
    for quadro in janela:
        assert all(jogador.track_id is not None for jogador in quadro.players)
        assert all(jogador.keypoints for jogador in quadro.players)
        assert all(jogador.pitch_xy is not None for jogador in quadro.players)


def test_no_stage_waits_on_an_anchor_it_cannot_influence(execucao: RunResult) -> None:
    """Os dois conjuntos sao uniformes, e por isso nao ha barreira no grafo.

    Enquanto quem mede sobre o jogador vivia num conjunto ancorado no `shot`, o `shot` nao podia
    olhar para `players`, e sem isso o instante saia da maior quebra da trajetoria mesmo com a bola
    sozinha no meio do campo. O conjunto denso e mais barato que a janela no clipe mediano: 105
    quadros contra 121.

    A bola tambem amostra no denso, e nao mais a 5 fps, porque a restricao de proximidade so tem
    sentido se algum quadro amostrado pegar a bola no pe: a 5 fps sao 12 quadros de passo, e o
    contato dura menos que isso.
    """
    ordem = [etapa.kind for etapa in execucao.stages]

    assert {e.frames for e in execucao.stages} == {"dense"}
    assert ordem.index("players") < ordem.index("shot")


def test_every_stage_declares_its_own_identity_in_pipeline_versions(execucao: RunResult) -> None:
    """A procedencia precisa distinguir modelo de stub, senao o evento mente sobre o que mediu."""
    versoes = execucao.pipeline.versions

    assert set(versoes) == MODULE_KINDS
    assert set(execucao.pipeline.timings_ms) == MODULE_KINDS
    assert versoes["shot"].startswith("ball-break@")
    assert all(versoes[kind].startswith("stub-") for kind in MODULE_KINDS if kind != "shot")


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
    assert quadro.field is not None
    assert quadro.field.homography == gabarito.homography
    assert quadro.ball is not None
    assert quadro.ball.pitch_xy == gabarito.ball_pitch_xy[SHOT_FRAME_INDEX]


def test_the_platform_runs_end_to_end_without_a_single_feature_installed(
    execucao: RunResult,
) -> None:
    """Sem card de modelo, o fluxo do clipe ao evento tem que continuar inteiro.

    O `shot` que roda aqui ja e modelo de verdade, e a selecao de features esta vazia porque
    nenhuma foi escrita. O que existe e o caminho: quem instalar o primeiro extrator nao muda
    nenhuma linha destas.
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
    """`shooter_track_id` e opcional no contrato, e exigi-lo aqui afirmava mais do que a especie
    promete. O modulo de instante nao consegue produzi-lo: `tracking` roda na janela ancorada no
    `shot`, entao depender dele fecharia ciclo no grafo.
    """
    atirador = registro.get("shooter_track_id")
    return _Shot(
        int(cast("Any", registro["frame_index"])),
        None if atirador is None else int(cast("Any", atirador)),
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
