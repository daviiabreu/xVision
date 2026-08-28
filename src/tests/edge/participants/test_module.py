"""O adaptador conferido pela suite da propria borda, e nao por teste escrito aqui."""

from collections.abc import Mapping
from dataclasses import replace
from typing import cast

import numpy as np
import pytest

from edge.modules.base import ModuleNotLoaded, PreprocessSpec
from edge.modules.conformance import ModuleConformance, make_batch
from edge.participants.detector import Frame, RawDetection
from edge.participants.module import PREPROCESS, PlayersModule, TrackingModule, to_source_box
from edge.participants.types import Box, Role

SOURCE = (1920, 1080)
ROLES: Mapping[int, Role] = {1: "goalkeeper", 2: "player", 3: "referee"}


class FakeBackend:
    """Devolve sempre o mesmo par, com a bola no meio para provar que ela e descartada."""

    class_roles: Mapping[int, Role] = ROLES

    def __init__(self) -> None:
        self.starts = 0

    def start_clip(self) -> None:
        self.starts += 1

    def _raws(self) -> tuple[RawDetection, ...]:
        return (
            RawDetection(Box(100, 300, 200, 500), 0.9, class_id=2, track_id=7),
            RawDetection(Box(600, 600, 620, 620), 0.5, class_id=0, track_id=8),
            RawDetection(Box(300, 320, 380, 520), 0.7, class_id=3, track_id=9),
        )

    def infer(self, frame: Frame) -> tuple[RawDetection, ...]:
        return self._raws()

    def track(self, frame: Frame) -> tuple[RawDetection, ...]:
        return self._raws()


class TestPlayersModuleConformance(ModuleConformance):
    @pytest.fixture
    def module(self) -> PlayersModule:
        return PlayersModule(source_size=SOURCE, backend=FakeBackend())


class TestTrackingModuleConformance(ModuleConformance):
    @pytest.fixture
    def module(self) -> TrackingModule:
        return TrackingModule(source_size=SOURCE, backend=FakeBackend())


def test_the_registry_can_build_the_module_with_no_arguments() -> None:
    """`load_modules` constroi com `_resolve(...)()`, entao o entry point exige zero argumento."""
    assert PlayersModule().spec.kind == "players"
    assert TrackingModule().spec.kind == "tracking"


def test_without_the_source_resolution_the_module_refuses_to_infer() -> None:
    """A recusa continua, e mudou de lugar: sem o tamanho a caixa sairia errada e calada.

    Ela saiu do `load()` porque em producao o registro constroi com `_resolve(...)()`, sem
    argumento, e a medida chega no lote. Recusar antes de ver o lote recusaria sempre.
    """
    module = PlayersModule(backend=FakeBackend())
    module.load()

    with pytest.raises(ModuleNotLoaded, match="resolucao do clipe de origem"):
        module.infer(make_batch(PREPROCESS), {})


def test_the_module_takes_the_resolution_from_the_batch() -> None:
    """O caminho de producao: nada no construtor, e o `FrameBatch` carrega a medida.

    A caixa cobre o tensor inteiro, entao a de volta cobre o quadro inteiro. Se a conversao
    tivesse ficado sem tamanho, sairiam os pixels do tensor 1280x1280 em vez de 1920x1080.
    """
    module = PlayersModule(backend=FakeBackend())
    module.load()
    lote = replace(make_batch(PREPROCESS), source_size=SOURCE)

    registros = module.infer(lote, {}).records[0]

    assert [registro["role"] for registro in registros] == ["player", "referee"]
    x1, y1, x2, y2 = cast("tuple[float, float, float, float]", registros[0]["box"])
    assert (x1, y1) == (150.0, 30.0)
    assert (x2, y2) == (300.0, 330.0)


def test_the_batch_resolution_wins_over_the_constructor() -> None:
    """Divergindo as duas, vale a do lote: ela e a medida do quadro que de fato entrou."""
    module = PlayersModule(source_size=(640, 360), backend=FakeBackend())
    module.load()

    do_construtor = module.infer(make_batch(PREPROCESS), {}).records[0][0]["box"]
    do_lote = module.infer(replace(make_batch(PREPROCESS), source_size=SOURCE), {}).records[0][0][
        "box"
    ]

    assert do_construtor != do_lote


def test_players_drops_the_ball_and_reports_role_and_confidence() -> None:
    module = PlayersModule(source_size=SOURCE, backend=FakeBackend())
    module.load()

    registros = module.infer(make_batch(PREPROCESS), {}).records[0]

    assert [registro["role"] for registro in registros] == ["player", "referee"]
    assert [registro["confidence"] for registro in registros] == [0.9, 0.7]


def test_tracking_links_the_track_to_the_detection_by_position() -> None:
    """A bola sai antes de numerar, senao `detection_index` apontaria para o registro errado."""
    module = TrackingModule(source_size=SOURCE, backend=FakeBackend())
    module.load()

    registros = module.infer(make_batch(PREPROCESS), {}).records[0]

    assert [registro["detection_index"] for registro in registros] == [0, 1]
    assert [registro["track_id"] for registro in registros] == [7, 9]


def test_tracking_resets_the_tracker_when_the_clip_starts() -> None:
    backend = FakeBackend()
    TrackingModule(source_size=SOURCE, backend=backend).load()

    assert backend.starts == 1


def test_the_box_comes_back_in_the_pixels_of_the_original_frame() -> None:
    """1920x1080 em 1280x1280 com letterbox: escala 2/3 e 280 pixels de margem em cima."""
    inteiro = to_source_box(Box(0, 280, 1280, 1000), PREPROCESS, SOURCE)

    assert inteiro == Box(0.0, 0.0, 1920.0, 1080.0)


def test_the_box_conversion_inverts_the_edge_preprocessing_exactly() -> None:
    """Confere contra o `prepare` da borda, e nao contra a conta repetida aqui."""
    from edge.frames.preprocess import prepare

    quadro = np.zeros((SOURCE[1], SOURCE[0], 3), dtype=np.uint8)
    quadro[270:540, 480:960] = 255
    preparado = prepare(quadro, PREPROCESS)

    linhas = np.nonzero(preparado.any(axis=(1, 2)))[0]
    colunas = np.nonzero(preparado.any(axis=(0, 2)))[0]
    devolvida = to_source_box(
        Box(float(colunas[0]), float(linhas[0]), float(colunas[-1] + 1), float(linhas[-1] + 1)),
        PREPROCESS,
        SOURCE,
    )

    assert devolvida.x1 == pytest.approx(480, abs=2)
    assert devolvida.y1 == pytest.approx(270, abs=2)
    assert devolvida.x2 == pytest.approx(960, abs=2)
    assert devolvida.y2 == pytest.approx(540, abs=2)


def test_a_batch_from_another_spec_is_refused() -> None:
    module = PlayersModule(source_size=SOURCE, backend=FakeBackend())
    module.load()
    outro = PreprocessSpec(width=640, height=640, color="bgr", letterbox=True, normalize="none")

    with pytest.raises(Exception, match="declarou"):
        module.infer(make_batch(outro), {})
