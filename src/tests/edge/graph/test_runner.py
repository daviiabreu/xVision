"""O runner: executa o grafo, mede o tempo de cada etapa e nunca derruba a execucao inteira."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import pytest

from contracts import Clip, Perception, PipelineMetadata
from edge.frames.provider import MissingFrame
from edge.graph.framesets import read_framesets
from edge.graph.runner import RunResult, assemble, run_pipeline
from edge.modules.base import (
    FrameBatch,
    ModuleError,
    ModuleInputs,
    ModuleOutput,
    ModuleSpec,
    PerceptionModule,
    PreprocessSpec,
    Temporal,
)
from edge.modules.stubs import STUBS
from edge.testing.pitch import SHOT_FRAME_INDEX, clip_truth

COARSE = PreprocessSpec(width=320, height=240, color="rgb", letterbox=False, normalize="none")

CONFIG = """
[frames.coarse]
strategy = "uniform"
fps = 5

[frames.window]
strategy = "around"
anchor   = "shot"
before_s = 1.5
after_s  = 0.5
fps      = "native"

[frames.estreita]
strategy = "around"
anchor   = "shot"
before_s = 0.1
after_s  = 0.1
fps      = "native"
"""


class ProviderFalso:
    def __init__(self, frame_count: int = 30) -> None:
        self.pedidos: list[tuple[tuple[int, ...], str]] = []
        self.frame_count = frame_count

    def frames(self, indices: Sequence[int], spec: PreprocessSpec) -> FrameBatch:
        indices = tuple(indices)
        fora = [i for i in indices if not 0 <= i < self.frame_count]
        if fora:
            raise MissingFrame(f"o clipe de teste nao entrega os quadros {fora}")
        self.pedidos.append((indices, spec.fingerprint[:8]))
        tensor = np.zeros((len(indices), spec.height, spec.width, 3), dtype=np.uint8)
        return FrameBatch(indices=indices, tensor=tensor, preprocess=spec)


class ModuloEspia:
    def __init__(self, kind: str, needs: tuple[str, ...] = (), **extra: object) -> None:
        self.spec = ModuleSpec(
            kind=kind,
            name=f"espia-{kind}",
            version="0.1.0",
            frames="coarse",
            preprocess=COARSE,
            needs=needs,
            **extra,  # pyright: ignore[reportArgumentType]
        )
        self.viu: list[str] = []
        self.lotes: list[tuple[int, ...]] = []

    def load(self) -> None: ...

    def infer(self, batch: FrameBatch, ctx: ModuleInputs) -> ModuleOutput:
        self.viu = sorted(ctx)
        self.lotes.append(batch.indices)
        return ModuleOutput(
            kind=self.spec.kind, frames=batch.indices, records=((),) * len(batch.indices)
        )

    def close(self) -> None: ...


class ModuloQueExplode(ModuloEspia):
    def infer(self, batch: FrameBatch, ctx: ModuleInputs) -> ModuleOutput:
        raise ModuleError("o peso nao carregou")


def clipe() -> Clip:
    gabarito = clip_truth()
    return Clip(
        clip_id="clipe-1",
        path="x.mp4",
        fps=float(gabarito.fps),
        frame_count=gabarito.frame_count,
        width=gabarito.width,
        height=gabarito.height,
    )


def roda(modules: dict[str, PerceptionModule], config: str = CONFIG) -> RunResult:
    return run_pipeline(
        clip=clipe(),
        modules=modules,
        framesets=read_framesets(config),
        provider=ProviderFalso(),  # pyright: ignore[reportArgumentType]
    )


def test_topological_order_is_the_order_the_modules_run_in() -> None:
    jogadores, campo = ModuloEspia("players"), ModuloEspia("field", needs=("players",))

    resultado = roda({"players": jogadores, "field": campo})

    assert [etapa.kind for etapa in resultado.stages] == ["players", "field"]
    assert campo.viu == ["players"]
    assert jogadores.viu == []


def test_anchor_frameset_resolves_only_after_producer() -> None:
    provider = ProviderFalso()
    chute = _stub("shot")
    pose = _stub("pose")
    pose.spec = replace(pose.spec, frames="window")

    run_pipeline(
        clip=clipe(),
        modules={"shot": chute, "pose": pose},
        framesets=read_framesets(CONFIG),
        provider=provider,  # pyright: ignore[reportArgumentType]
    )

    coarse, janela = provider.pedidos[0][0], provider.pedidos[-1][0]
    assert coarse == tuple(range(0, 30, 2))
    assert janela == tuple(range(SHOT_FRAME_INDEX - 15, SHOT_FRAME_INDEX + 6))


def test_unknown_frameset_name_fails_before_execution() -> None:
    jogadores = ModuloEspia("players")
    jogadores.spec = replace(jogadores.spec, frames="inexistente")

    with pytest.raises(ModuleError, match="inexistente"):
        roda({"players": jogadores})

    assert jogadores.lotes == []


def test_module_exception_does_not_abort_run() -> None:
    quebrado, bola = ModuloQueExplode("players"), ModuloEspia("ball")

    resultado = roda({"players": quebrado, "ball": bola})

    assert {etapa.kind: etapa.status for etapa in resultado.stages} == {
        "players": "failed",
        "ball": "ok",
    }
    assert bola.lotes


def test_module_exception_is_recorded_with_type_and_message() -> None:
    resultado = roda({"players": ModuloQueExplode("players")})

    (etapa,) = resultado.stages
    assert etapa.reason == "ModuleError: o peso nao carregou"


def test_a_module_whose_anchor_failed_is_skipped_and_not_blamed_for_the_frameset() -> None:
    pose = _stub("pose")
    pose.spec = replace(pose.spec, frames="window")

    resultado = roda({"shot": ModuloQueExplode("shot"), "pose": pose})

    etapa = next(item for item in resultado.stages if item.kind == "pose")
    assert etapa.status == "skipped"
    assert etapa.reason == "shot nao produziu saida"


def test_the_merge_of_overlapping_batches_keeps_the_first_record_of_each_frame() -> None:

    class ContaLotes(ModuloEspia):
        def infer(self, batch: FrameBatch, ctx: ModuleInputs) -> ModuleOutput:
            self.lotes.append(batch.indices)
            marca = len(self.lotes)
            return ModuleOutput(
                kind=self.spec.kind,
                frames=batch.indices,
                records=tuple(({"lote": marca, "track_id": 1},) for _ in batch.indices),
            )

    modulo = ContaLotes("tracking", batching=Temporal(size=4, stride=1))

    saida = roda({"tracking": modulo}).outputs["tracking"]

    assert saida.records[0] == ({"lote": 1, "track_id": 1},)
    assert len(set(saida.frames)) == len(saida.frames)


def test_a_module_that_needs_a_failed_one_is_skipped_with_the_reason() -> None:
    resultado = roda(
        {"players": ModuloQueExplode("players"), "field": ModuloEspia("field", needs=("players",))}
    )

    campo = next(etapa for etapa in resultado.stages if etapa.kind == "field")
    assert campo.status == "skipped"
    assert "players" in (campo.reason or "")


def test_timings_recorded_for_every_stage() -> None:

    class DemoraEExplode(ModuloEspia):
        def infer(self, batch: FrameBatch, ctx: ModuleInputs) -> ModuleOutput:
            time.sleep(0.02)
            raise ModuleError("depois de gastar tempo")

    resultado = roda({"players": ModuloEspia("players"), "ball": DemoraEExplode("ball")})

    assert set(resultado.pipeline.timings_ms) == {"players", "ball"}
    assert resultado.pipeline.timings_ms["ball"] >= 20.0
    assert resultado.pipeline.timings_ms["players"] > 0.0


def test_versions_record_name_version_and_the_stub_prefix() -> None:
    resultado = roda({"players": _stub("players"), "ball": ModuloEspia("ball")})

    assert resultado.pipeline.versions["players"] == "stub-players@0.1.0"
    assert resultado.pipeline.versions["ball"] == "espia-ball@0.1.0"


def test_temporal_batching_delivers_consecutive_frames() -> None:
    temporal = ModuloEspia("tracking", batching=Temporal(size=4, stride=2))

    resultado = roda({"tracking": temporal})

    assert [etapa.status for etapa in resultado.stages] == ["ok"]
    assert temporal.lotes
    for lote in temporal.lotes:
        assert len(lote) == 4
        assert lote == tuple(range(lote[0], lote[0] + 4))
        assert lote[0] >= 0 and lote[-1] < clipe().frame_count


def test_a_temporal_window_at_the_end_of_the_frameset_slides_back_instead_of_shrinking() -> None:
    temporal = ModuloEspia("tracking", batching=Temporal(size=8, stride=1))

    resultado = roda({"tracking": temporal})

    ultimo = temporal.lotes[-1]
    assert [etapa.status for etapa in resultado.stages] == ["ok"]
    assert len(ultimo) == 8
    assert ultimo[-1] == 28


def test_an_exception_inside_close_degrades_the_stage_instead_of_killing_the_run() -> None:

    class FechaMal(ModuloEspia):
        def close(self) -> None:
            raise ModuleError("contexto de cuda ja liberado")

    resultado = roda({"players": FechaMal("players"), "ball": ModuloEspia("ball")})

    estados = {etapa.kind: etapa.status for etapa in resultado.stages}
    assert estados == {"players": "failed", "ball": "ok"}
    jogadores = next(etapa for etapa in resultado.stages if etapa.kind == "players")
    assert jogadores.reason == "ModuleError: contexto de cuda ja liberado"


def test_the_run_produces_a_perception_and_a_pipeline_metadata() -> None:
    resultado = roda({kind: _stub(kind) for kind in ("players", "ball", "field", "shot")})

    assert isinstance(resultado.perception, Perception)
    assert isinstance(resultado.pipeline, PipelineMetadata)
    assert resultado.perception.clip_id == "clipe-1"


def test_the_perception_carries_what_the_stubs_saw_at_the_shot_frame() -> None:
    gabarito = clip_truth()
    resultado = roda({kind: _stub(kind) for kind in ("players", "ball", "field", "tracking")})

    quadro = next(q for q in resultado.perception.frames if q.frame_index == SHOT_FRAME_INDEX)

    assert len(quadro.players) == len(gabarito.players)
    assert quadro.ball is not None
    assert quadro.field is not None
    assert quadro.field.homography == gabarito.homography
    assert {jogador.track_id for jogador in quadro.players} == {
        jogador.track_id for jogador in gabarito.players
    }


def test_pitch_coordinates_only_survive_where_the_field_module_ran() -> None:
    resultado = roda({kind: _stub(kind) for kind in ("players", "ball")})

    quadro = resultado.perception.frames[0]
    assert quadro.field is None
    assert all(jogador.pitch_xy is None for jogador in quadro.players)
    assert quadro.ball is not None and quadro.ball.pitch_xy is None


def test_assembler_projects_player_feet_with_detected_field_geometry() -> None:
    homography = (10.0, 0.0, 960.0, 0.0, -10.0, 540.0, 0.0, 0.0, 1.0)
    field = {
        "homography": homography,
        "reprojection_error_px": 0.2,
        "reprojection_error_p95_px": 0.4,
        "keypoints": (),
        "observed_keypoint_count": 8,
        "inlier_count": 8,
        "source": "detected",
        "age_frames": 0,
        "template_id": "pitch-105x68-center-v1",
        "reason": None,
    }
    outputs = {
        "field": ModuleOutput("field", (5,), (((field),),)),
        "players": ModuleOutput(
            "players",
            (5,),
            ((({"box": (1060.0, 500.0, 1100.0, 620.0), "confidence": 0.9}),),),
        ),
    }

    perception = assemble(clipe(), outputs)

    assert perception.frames[0].players[0].pitch_xy == pytest.approx((12.0, -8.0))


def test_assembler_does_not_publish_player_coordinates_outside_the_template() -> None:
    field = {
        "homography": (10.0, 0.0, 960.0, 0.0, -10.0, 540.0, 0.0, 0.0, 1.0),
        "reprojection_error_px": 0.2,
        "reprojection_error_p95_px": 0.4,
        "keypoints": (),
        "observed_keypoint_count": 8,
        "inlier_count": 8,
        "source": "detected",
        "age_frames": 0,
        "template_id": "pitch-105x68-center-v1",
        "reason": None,
    }
    outputs = {
        "field": ModuleOutput("field", (5,), (((field),),)),
        "players": ModuleOutput(
            "players",
            (5,),
            ((({"box": (1940.0, 100.0, 1980.0, 120.0), "confidence": 0.9}),),),
        ),
    }

    perception = assemble(clipe(), outputs)

    assert perception.frames[0].players[0].pitch_xy is None


def test_assembler_never_projects_an_airborne_ball_on_the_pitch_plane() -> None:
    field = {
        "homography": (10.0, 0.0, 960.0, 0.0, -10.0, 540.0, 0.0, 0.0, 1.0),
        "reprojection_error_px": 0.2,
        "reprojection_error_p95_px": 0.4,
        "keypoints": (),
        "observed_keypoint_count": 8,
        "inlier_count": 8,
        "source": "detected",
        "age_frames": 0,
        "template_id": "pitch-105x68-center-v1",
        "reason": None,
        "refinement": "accepted",
        "line_support_count": 6,
    }
    outputs = {
        "field": ModuleOutput("field", (5,), (((field),),)),
        "ball": ModuleOutput(
            "ball",
            (5,),
            (
                (
                    (
                        {
                            "box": (1075.0, 615.0, 1085.0, 625.0),
                            "confidence": 0.9,
                            "pitch_xy": (12.0, -8.0),
                        }
                    ),
                ),
            ),
        ),
    }

    perception = assemble(clipe(), outputs)

    assert perception.frames[0].ball is not None
    assert perception.frames[0].ball.pitch_xy is None
    assert perception.frames[0].field is not None
    assert perception.frames[0].field.refinement == "accepted"
    assert perception.frames[0].field.line_support_count == 6


def _stub(kind: str) -> PerceptionModule:
    return next(classe for classe in STUBS if classe.spec.kind == kind)()


class ModuloComAncora(ModuloEspia):
    def __init__(self, kind: str, indice: int) -> None:
        super().__init__(kind)
        self._indice = indice

    def infer(self, batch: FrameBatch, ctx: ModuleInputs) -> ModuleOutput:
        registros = tuple(
            ({"frame_index": self._indice},) if i == batch.indices[0] else () for i in batch.indices
        )
        return ModuleOutput(kind=self.spec.kind, frames=batch.indices, records=registros)


@pytest.mark.parametrize("indice", [-1, 30, 10_000])
def test_an_anchor_outside_the_clip_fails_the_stage_instead_of_shifting_the_window(
    indice: int,
) -> None:
    contato = ModuloComAncora("shot", indice)
    pose = ModuloEspia("pose", needs=("shot",))
    pose.spec = replace(pose.spec, frames="window")

    resultado = roda({"shot": contato, "pose": pose})

    etapa = next(e for e in resultado.stages if e.kind == "pose")
    assert etapa.status == "failed"
    assert etapa.reason is not None
    assert str(indice) in etapa.reason
    assert pose.lotes == []


def test_a_temporal_window_never_reads_past_the_frameset_it_declared() -> None:
    contato = ModuloComAncora("shot", SHOT_FRAME_INDEX)
    pose = ModuloEspia("pose", needs=("shot",))
    pose.spec = replace(pose.spec, frames="window", batching=Temporal(size=8, stride=8))

    resultado = roda({"shot": contato, "pose": pose})

    etapa = next(e for e in resultado.stages if e.kind == "pose")
    assert etapa.status == "ok", etapa.reason
    janela = set(range(3, 24))
    fora = sorted({i for lote in pose.lotes for i in lote} - janela)
    assert fora == [], f"leu {fora}, que estao fora da janela declarada 3..23"


def test_a_frameset_narrower_than_the_declared_batch_yields_one_batch_of_what_exists() -> None:
    contato = ModuloComAncora("shot", SHOT_FRAME_INDEX)
    pose = ModuloEspia("pose", needs=("shot",))
    pose.spec = replace(pose.spec, frames="estreita", batching=Temporal(size=8, stride=1))

    resultado = roda({"shot": contato, "pose": pose})

    assert [e.status for e in resultado.stages if e.kind == "pose"] == ["ok"]
    assert pose.lotes == [(17, 18, 19)]


def test_the_synthetic_assembly_stage_never_reaches_the_event_metadata() -> None:
    """`versions` e `timings_ms` documentam modulo que rodou, e `assemble` nao e modulo."""
    from edge.modules.base import MODULE_KINDS

    contato = ModuloComAncora("shot", SHOT_FRAME_INDEX)

    resultado = roda({"shot": contato})

    assert set(resultado.pipeline.versions) <= MODULE_KINDS
    assert set(resultado.pipeline.timings_ms) <= MODULE_KINDS


def test_a_failed_assembly_stage_stays_out_of_the_event_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`versions` documenta modulo que rodou, e `assemble` nao e modulo nenhum."""
    from edge.graph import runner as runner_mod
    from edge.modules.base import MODULE_KINDS

    def explode(*_: object, **__: object) -> object:
        raise TypeError("montagem quebrada")

    monkeypatch.setattr(runner_mod, "assemble", explode)

    resultado = roda({"players": ModuloEspia("players")})

    assert any(e.kind == "assemble" and e.status == "failed" for e in resultado.stages)
    assert set(resultado.pipeline.versions) <= MODULE_KINDS
    assert "assemble" not in resultado.pipeline.timings_ms
