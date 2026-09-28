"""Registro sem a chave que a especie promete vira etapa falha, e nao execucao morta.

O bug e reprodutivel e a vitima e o exemplo oficial do repositorio: `ExampleModule` devolve
`box` sem `confidence`, passa nos seis testes de conformidade, e `assemble` estoura com
`KeyError` depois de todas as etapas terem reportado `ok`. Erro que passa como acerto e some do
relatorio e o pior modo de falha do runner.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pytest

from contracts import Clip, Perception
from edge.graph.framesets import read_framesets
from edge.graph.runner import run_pipeline
from edge.modules.base import (
    FrameBatch,
    ModuleInputs,
    ModuleOutput,
    ModuleSpec,
    PreprocessSpec,
)
from edge.modules.records import REQUIRED, RecordError, check_records
from edge.testing.pitch import clip_truth

COARSE = PreprocessSpec(width=320, height=240, color="rgb", letterbox=False, normalize="none")

CONFIG = """
[frames.coarse]
strategy = "uniform"
fps = 5
"""


class ModuloIncompleto:
    def __init__(self, kind: str, registro: dict[str, object]) -> None:
        self.spec = ModuleSpec(
            kind=kind,
            name=f"incompleto-{kind}",
            version="0.1.0",
            frames="coarse",
            preprocess=COARSE,
        )
        self._registro = registro

    def load(self) -> None: ...

    def infer(self, batch: FrameBatch, ctx: ModuleInputs) -> ModuleOutput:
        return ModuleOutput(
            kind=self.spec.kind,
            frames=batch.indices,
            records=((dict(self._registro),),) * len(batch.indices),
        )

    def close(self) -> None: ...


class ProviderFalso:
    def frames(self, indices: Sequence[int], spec: PreprocessSpec) -> FrameBatch:
        pedidos = tuple(indices)
        return FrameBatch(
            indices=pedidos,
            tensor=np.zeros((len(pedidos), spec.height, spec.width, 3), dtype=np.uint8),
            preprocess=spec,
        )


def clipe() -> Clip:
    g = clip_truth()
    return Clip(
        clip_id="c",
        path="x.mp4",
        fps=float(g.fps),
        frame_count=g.frame_count,
        width=g.width,
        height=g.height,
    )


def test_every_species_the_runner_reads_declares_what_it_needs() -> None:
    from edge.modules.base import MODULE_KINDS

    assert set(REQUIRED) == set(MODULE_KINDS)


def test_a_record_without_the_promised_key_names_species_frame_record_and_key() -> None:
    saida = ModuleOutput(kind="players", frames=(7,), records=(({"box": (0.0, 0.0, 1.0, 1.0)},),))

    with pytest.raises(RecordError) as erro:
        check_records(saida)

    mensagem = str(erro.value)
    assert "players" in mensagem and "confidence" in mensagem
    assert "7" in mensagem


def test_marking_promises_both_players_and_their_distance() -> None:
    saida = ModuleOutput(
        kind="marking",
        frames=(7,),
        records=(({"team_a_track_id": 7, "team_b_track_id": 4},),),
    )

    with pytest.raises(RecordError) as erro:
        check_records(saida)

    mensagem = str(erro.value)
    assert "marking" in mensagem and "distance" in mensagem
    assert "7" in mensagem


def test_an_extra_key_is_allowed_because_it_is_not_our_business() -> None:
    """`players` que devolve `velocity` continua valido: a especie diz o minimo, nao o maximo."""
    saida = ModuleOutput(
        kind="players",
        frames=(0,),
        records=(({"box": (0.0, 0.0, 1.0, 1.0), "confidence": 0.9, "velocity": 3.2},),),
    )

    check_records(saida)


def test_a_confidence_outside_zero_and_one_is_not_our_business_either() -> None:
    """Score calibrado e exigencia que nem todo detector tem, e para `field` nao significa nada."""
    saida = ModuleOutput(
        kind="players", frames=(0,), records=(({"box": (0, 0, 1, 1), "confidence": 42.0},),)
    )

    check_records(saida)


def test_a_frame_with_no_detection_is_valid() -> None:
    check_records(ModuleOutput(kind="players", frames=(0, 1), records=((), ())))


def test_the_incomplete_module_fails_its_own_stage_instead_of_killing_the_run() -> None:
    """Antes, a execucao morria com KeyError depois de todas as etapas dizerem `ok`."""
    incompleto = ModuloIncompleto("players", {"box": (0.0, 0.0, 1.0, 1.0)})

    resultado = run_pipeline(
        clip=clipe(),
        modules={"players": incompleto},
        framesets=read_framesets(CONFIG),
        provider=ProviderFalso(),  # pyright: ignore[reportArgumentType]
    )

    etapa = next(e for e in resultado.stages if e.kind == "players")
    assert etapa.status == "failed"
    assert etapa.reason is not None and "confidence" in etapa.reason
    assert resultado.perception.frames == ()


def test_identity_without_a_jersey_number_still_assembles() -> None:
    """A tabela nao exige camisa de proposito, para nao fechar reidentificacao por aparencia.

    Entao `assemble` tem que ler o campo com guarda, como ja faz com `detection_index`. Sem isso
    a etapa reportava `ok`, `assemble` estourava com KeyError e saia evento com cara de bom.
    """
    from edge.graph.runner import assemble

    percepcao = assemble(
        clipe(),
        {
            "players": ModuleOutput(
                kind="players",
                frames=(0,),
                records=(({"box": (0.0, 0.0, 1.0, 1.0), "confidence": 0.9},),),
            ),
            "identity": ModuleOutput(kind="identity", frames=(0,), records=(({"track_id": 7},),)),
        },
    )

    (quadro,) = percepcao.frames
    assert quadro.players[0].jersey_number is None


def _percepcao_com_identity(identity: Mapping[str, object]) -> Perception:
    """Monta um quadro de um jogador cuja deteccao ja traz `team`, rastreado como track 7."""
    from edge.graph.runner import assemble

    return assemble(
        clipe(),
        {
            "players": ModuleOutput(
                kind="players",
                frames=(0,),
                records=(({"box": (0.0, 0.0, 1.0, 1.0), "confidence": 0.9, "team": "detector"},),),
            ),
            "tracking": ModuleOutput(
                kind="tracking",
                frames=(0,),
                records=(({"detection_index": 0, "track_id": 7},),),
            ),
            "identity": ModuleOutput(kind="identity", frames=(0,), records=((identity,),)),
        },
    )


def test_o_time_de_identity_prevalece_sobre_o_time_do_detector() -> None:
    """A identidade temporal ve o clipe inteiro; o detector so ve o quadro, entao perde o empate."""
    percepcao = _percepcao_com_identity({"track_id": 7, "team": "team_b"})

    assert percepcao.frames[0].players[0].team == "team_b"


def test_o_time_do_detector_sobrevive_quando_identity_ainda_nao_classificou() -> None:
    """Enquanto faltam amostras, `identity` emite so o track: o fallback nao pode virar None."""
    percepcao = _percepcao_com_identity({"track_id": 7})

    assert percepcao.frames[0].players[0].team == "detector"


def test_o_papel_de_identity_prevalece_sobre_o_papel_estavel_do_tracking() -> None:
    """A identidade corrige pela cor o árbitro que o voto do tracking deixou como jogador."""
    from edge.graph.runner import assemble

    percepcao = assemble(
        clipe(),
        {
            "players": ModuleOutput(
                kind="players",
                frames=(0,),
                records=(({"box": (0.0, 0.0, 1.0, 1.0), "confidence": 0.9, "role": "player"},),),
            ),
            "tracking": ModuleOutput(
                kind="tracking",
                frames=(0,),
                records=(({"detection_index": 0, "track_id": 7, "role": "player"},),),
            ),
            "identity": ModuleOutput(
                kind="identity", frames=(0,), records=(({"track_id": 7, "role": "referee"},),)
            ),
        },
    )

    assert percepcao.frames[0].players[0].role == "referee"
