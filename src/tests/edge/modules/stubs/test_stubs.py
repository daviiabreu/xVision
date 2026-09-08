"""Os sete stubs sao o que faz existir fluxo de ponta a ponta sem nenhum modelo pronto."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from edge.cache.perception import (
    PerceptionCache,
    implementation_id,
    perception_key,
    run_module,
)
from edge.modules.base import MODULE_KINDS, ModuleOutput, PerceptionModule
from edge.modules.conformance import ModuleConformance, make_batch
from edge.modules.stubs import STUBS
from edge.modules.stubs.base import UnexpectedClip
from edge.modules.stubs.perception import REPROJECTION_ERROR_PX
from edge.testing.pitch import (
    FPS,
    HEIGHT,
    SHOOTER_TRACK_ID,
    SHOT_FRAME_INDEX,
    WIDTH,
    clip_truth,
    homography_matrix,
    player_box,
)

POR_ESPECIE = {stub().spec.kind: stub for stub in STUBS}


@pytest.mark.parametrize("kind", sorted(MODULE_KINDS))
@pytest.mark.parametrize("size,fps", [((1920, 1080), float(FPS)), ((WIDTH, HEIGHT), 59.94)])
def test_stub_rejects_foreign_clip_geometry(kind: str, size: tuple[int, int], fps: float) -> None:
    stub = POR_ESPECIE[kind]()
    stub.load()
    batch = replace(make_batch(stub.spec.preprocess), source_size=size, source_fps=fps)

    with pytest.raises(UnexpectedClip):
        stub.infer(batch, {})


@pytest.mark.parametrize("kind", sorted(MODULE_KINDS))
def test_old_cache_cannot_bypass_stub_rejection(kind: str, tmp_path: Path) -> None:
    stub = POR_ESPECIE[kind]()
    stub.load()
    fixture = replace(
        make_batch(stub.spec.preprocess), source_size=(WIDTH, HEIGHT), source_fps=float(FPS)
    )
    batch = replace(fixture, source_size=(1920, 1080), source_fps=59.94)
    cache = PerceptionCache(tmp_path)
    key = perception_key(
        "outro-clipe",
        replace(stub.spec, version="0.1.0"),
        batch.indices,
        {},
        implementation_id(stub),
    )
    cache.put(key, stub.infer(fixture, {}))
    assert cache.get(key) is not None

    with pytest.raises(UnexpectedClip):
        run_module(stub, batch, {}, cache=cache, clip_id="outro-clipe")


def test_there_is_exactly_one_stub_for_each_of_the_seven_kinds() -> None:
    assert set(POR_ESPECIE) == MODULE_KINDS
    assert len(STUBS) == len(MODULE_KINDS)


@pytest.mark.parametrize("kind", sorted(MODULE_KINDS))
def test_stub_declares_itself_as_stub_in_produced_by(kind: str) -> None:
    spec = POR_ESPECIE[kind]().spec

    assert spec.name == f"stub-{kind}"
    assert spec.version
    assert spec.artifact is None


@pytest.mark.parametrize("kind", sorted(MODULE_KINDS))
def test_stub_answers_one_record_set_per_frame_of_the_batch(kind: str) -> None:
    saida = _infere(kind, (0, 1, 2))

    assert saida.frames == (0, 1, 2)
    assert len(saida.records) == 3


def test_stub_field_returns_a_valid_homography_for_the_synthetic_pitch() -> None:
    gabarito = clip_truth()

    registro = _infere("field", (0,)).records[0][0]
    assert registro["homography"] == gabarito.homography
    assert registro["reprojection_error_px"] == REPROJECTION_ERROR_PX
    assert REPROJECTION_ERROR_PX < 5.0
    np.linalg.inv(homography_matrix(gabarito.homography))


def test_stub_shot_marks_the_contact_frame_and_the_shooter() -> None:
    indices = (SHOT_FRAME_INDEX - 1, SHOT_FRAME_INDEX, SHOT_FRAME_INDEX + 1)

    saida = _infere("shot", indices)

    assert saida.records[0] == ()
    assert saida.records[2] == ()
    assert saida.records[1][0]["shooter_track_id"] == SHOOTER_TRACK_ID
    assert saida.records[1][0]["frame_index"] == SHOT_FRAME_INDEX


def test_stub_players_places_every_player_of_the_truth_on_the_pitch() -> None:
    gabarito = clip_truth()
    saida = _infere("players", (SHOT_FRAME_INDEX,))

    assert len(saida.records[0]) == len(gabarito.players)
    posicoes = [registro["pitch_xy"] for registro in saida.records[0]]
    assert posicoes == [jogador.pitch_xy[SHOT_FRAME_INDEX] for jogador in gabarito.players]


def test_the_identifier_of_a_trajectory_belongs_to_tracking_and_not_to_players() -> None:
    gabarito = clip_truth()

    deteccao = _infere("players", (SHOT_FRAME_INDEX,)).records[0]
    trajetorias = _infere("tracking", (SHOT_FRAME_INDEX,)).records[0]

    assert all("track_id" not in registro for registro in deteccao)
    assert [registro["detection_index"] for registro in trajetorias] == list(range(len(deteccao)))
    assert [registro["track_id"] for registro in trajetorias] == [
        jogador.track_id for jogador in gabarito.players
    ]


@pytest.mark.parametrize("kind", ["players", "ball", "tracking", "pose"])
def test_stub_reads_the_truth_at_the_frame_it_was_asked_for(kind: str) -> None:
    gabarito = clip_truth()
    indices = (0, SHOT_FRAME_INDEX, gabarito.frame_count - 1)

    saida = _infere(kind, indices)

    distintos = {repr(registros) for registros in saida.records}
    assert len(distintos) == len(indices), f"{kind} respondeu igual nos tres quadros"


def test_the_player_box_of_the_stub_is_the_one_the_truth_computes() -> None:
    gabarito = clip_truth()
    finalizador = next(j for j in gabarito.players if j.track_id == SHOOTER_TRACK_ID)

    deteccao = _infere("players", (SHOT_FRAME_INDEX,)).records[0][0]

    assert deteccao["box"] == player_box(gabarito, finalizador.pitch_xy[SHOT_FRAME_INDEX])
    assert deteccao["pitch_xy"] == finalizador.pitch_xy[SHOT_FRAME_INDEX]


def test_a_frame_outside_the_clip_gives_no_record_instead_of_raising() -> None:
    assert _infere("ball", (10_000,)).records == ((),)


def _infere(kind: str, indices: tuple[int, ...]) -> ModuleOutput:
    modulo = POR_ESPECIE[kind]()
    modulo.load()
    try:
        return modulo.infer(make_batch(modulo.spec.preprocess, indices=indices), {})
    finally:
        modulo.close()


@pytest.mark.parametrize("kind", sorted(MODULE_KINDS), ids=sorted(MODULE_KINDS))
class TestEachStubPassesTheConformanceSuite(ModuleConformance):
    @pytest.fixture
    def module(self, kind: str) -> PerceptionModule:
        return POR_ESPECIE[kind]()
