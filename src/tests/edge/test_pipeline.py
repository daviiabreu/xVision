"""Do clipe ao evento, num lugar que o pacote entrega.

Esta cola existia so dentro de `src/tests/edge/graph/test_integration.py`, o que tinha duas
consequencias: mudanca de assinatura no runner nao quebrava nada visivel, e a demonstracao da
sprint dependia de codigo de teste.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from contracts import Clip, FeatureValue, Perception
from contracts.validation import validate_shot_event
from edge.features.base import FeatureExtractor, FeatureSpec, ShotOutput, shot_provenance
from edge.frames.provider import FrameProvider, VideoSource
from edge.graph.framesets import read_framesets
from edge.modules.base import FrameBatch, ModuleInputs, ModuleOutput, PerceptionModule
from edge.modules.registry import load_modules, read_config
from edge.pipeline import (
    AmbiguousContact,
    AssemblyFailed,
    ClipResult,
    ContactOutOfRange,
    NoShotDetected,
    process_clip,
)
from edge.testing.make_clip import make_synthetic_clip
from edge.testing.pitch import SHOT_FRAME_INDEX, clip_truth

CONFIG = Path("config/pipeline.toml")


class _ShotMudo:
    """Roda de verdade e nao aponta contato em quadro nenhum, o caso real de clipe sem chute."""

    def __init__(self, real: PerceptionModule) -> None:
        self.spec = real.spec
        self._real = real

    def load(self) -> None:
        self._real.load()

    def infer(self, batch: FrameBatch, ctx: ModuleInputs) -> ModuleOutput:
        saida = self._real.infer(batch, ctx)
        return replace(saida, records=((),) * len(saida.frames))

    def close(self) -> None:
        self._real.close()


@pytest.fixture
def clipe(tmp_path: Path) -> Clip:
    caminho = make_synthetic_clip(tmp_path / "s.mp4")
    g = clip_truth()
    return Clip(
        clip_id="clipe-1",
        path=str(caminho),
        fps=float(g.fps),
        frame_count=g.frame_count,
        width=g.width,
        height=g.height,
    )


def _roda(clipe: Clip, extractors: Sequence[FeatureExtractor] = ()) -> ClipResult:
    provider = FrameProvider(VideoSource(clipe.path))
    try:
        return process_clip(
            clip=clipe,
            modules=load_modules(read_config(CONFIG)),
            framesets=read_framesets(CONFIG),
            provider=provider,
            extractors=extractors,
        )
    finally:
        provider.close()


def test_a_clip_becomes_an_event_that_validates_against_the_schema(clipe: Clip) -> None:
    resultado = _roda(clipe)

    validate_shot_event(resultado.event.to_dict())
    assert [e.status for e in resultado.stages] == ["ok"] * len(resultado.stages)


def test_the_shot_id_and_the_timestamp_come_from_the_frame_the_shot_module_found(
    clipe: Clip,
) -> None:
    """O instante nao e escolhido aqui: sai do quadro que a especie `shot` apontou."""
    evento = _roda(clipe).event

    assert evento.shot_id == f"{clipe.clip_id}:{SHOT_FRAME_INDEX}"
    assert evento.shot_timestamp_s == pytest.approx(SHOT_FRAME_INDEX / clipe.fps)


def test_an_empty_selection_yields_an_event_without_features(clipe: Clip) -> None:
    assert _roda(clipe).event.features == ()


def test_a_run_where_the_shot_module_finds_nothing_says_so_instead_of_inventing(
    clipe: Clip,
) -> None:
    """Sem instante de contato nao existe finalizacao, e inventar um seria pior que falhar."""
    modulos = load_modules(read_config(CONFIG))
    modulos["shot"] = _ShotMudo(modulos["shot"])
    provider = FrameProvider(VideoSource(clipe.path))
    try:
        with pytest.raises(NoShotDetected, match="shot"):
            process_clip(
                clip=clipe,
                modules=modulos,
                framesets=read_framesets(CONFIG),
                provider=provider,
                extractors=(),
            )
    finally:
        provider.close()


def test_a_failed_assembly_refuses_to_produce_an_event(
    clipe: Clip, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Percepcao vazia com as saidas presentes nao marca feature nenhuma como indisponivel.

    Sem esta recusa, um erro de programacao dentro de `assemble` vira evento schema-valido com
    feature `ok` medindo sobre nada, e quem consome nao tem como distinguir de dado ruim.
    """
    from edge.graph import runner

    def explode(*_: object, **__: object) -> object:
        raise TypeError("bug meu, e nao dado ruim")

    monkeypatch.setattr(runner, "assemble", explode)

    with pytest.raises(AssemblyFailed, match="bug meu"):
        _roda(clipe)


def test_a_contact_frame_outside_the_clip_is_refused(clipe: Clip) -> None:
    """`_anchor_frame` ja recusa fora de faixa; sem a mesma regra aqui saia timestamp absurdo."""
    modulos = load_modules(read_config(CONFIG))
    modulos["shot"] = _ShotEm(modulos["shot"], 100_000)
    provider = FrameProvider(VideoSource(clipe.path))
    try:
        with pytest.raises(ContactOutOfRange, match="100000"):
            process_clip(
                clip=clipe,
                modules=modulos,
                framesets=read_framesets(CONFIG),
                provider=provider,
                extractors=(),
            )
    finally:
        provider.close()


def test_two_contacts_in_one_clip_are_refused_instead_of_silently_taking_the_first(
    clipe: Clip,
) -> None:
    """Dois contatos significam ou deteccao por quadro sem fusao, ou duas finalizacoes no corte.

    Nos dois casos, pegar o primeiro produz evento errado calado, e um evento por clipe deixa de
    ser a modelagem certa. Falhar nomeia o problema para quem for consertar.
    """
    modulos = load_modules(read_config(CONFIG))
    modulos["shot"] = _ShotEm(modulos["shot"], SHOT_FRAME_INDEX, tambem=SHOT_FRAME_INDEX + 4)
    provider = FrameProvider(VideoSource(clipe.path))
    try:
        with pytest.raises(AmbiguousContact, match="2"):
            process_clip(
                clip=clipe,
                modules=modulos,
                framesets=read_framesets(CONFIG),
                provider=provider,
                extractors=(),
            )
    finally:
        provider.close()


class _ShotEm:
    def __init__(self, real: PerceptionModule, indice: int, tambem: int | None = None) -> None:
        self.spec = real.spec
        self._real = real
        self._indices = (indice,) if tambem is None else (indice, tambem)

    def load(self) -> None:
        self._real.load()

    def infer(self, batch: FrameBatch, ctx: ModuleInputs) -> ModuleOutput:
        saida = self._real.infer(batch, ctx)
        registros = tuple(
            ({"frame_index": self._indices[0], "shooter_track_id": 9},)
            if posicao == 0
            else (
                ({"frame_index": self._indices[1], "shooter_track_id": 9},)
                if posicao == 1 and len(self._indices) > 1
                else ()
            )
            for posicao in range(len(saida.frames))
        )
        return replace(saida, records=registros)

    def close(self) -> None:
        self._real.close()


def test_two_contacts_inside_one_frame_are_refused_too(clipe: Clip) -> None:
    """Olhar so o primeiro registro de cada quadro esconderia o segundo contato do mesmo quadro."""
    modulos = load_modules(read_config(CONFIG))
    modulos["shot"] = _ShotNoMesmoQuadro(modulos["shot"], SHOT_FRAME_INDEX, SHOT_FRAME_INDEX + 3)
    provider = FrameProvider(VideoSource(clipe.path))
    try:
        with pytest.raises(AmbiguousContact):
            process_clip(
                clip=clipe,
                modules=modulos,
                framesets=read_framesets(CONFIG),
                provider=provider,
                extractors=(),
            )
    finally:
        provider.close()


class _ShotNoMesmoQuadro:
    def __init__(self, real: PerceptionModule, um: int, outro: int) -> None:
        self.spec = real.spec
        self._real = real
        self._dois = (um, outro)

    def load(self) -> None:
        self._real.load()

    def infer(self, batch: FrameBatch, ctx: ModuleInputs) -> ModuleOutput:
        saida = self._real.infer(batch, ctx)
        registros = tuple(
            tuple({"frame_index": i, "shooter_track_id": 9} for i in self._dois)
            if posicao == 0
            else ()
            for posicao in range(len(saida.frames))
        )
        return replace(saida, records=registros)

    def close(self) -> None:
        self._real.close()


class _ShotComContato:
    """Aponta o contato com o valor que o teste mandar, para exercitar a leitura do instante."""

    def __init__(self, real: PerceptionModule, registro: dict[str, object]) -> None:
        self.spec = real.spec
        self._real = real
        self._registro = registro

    def load(self) -> None:
        self._real.load()

    def infer(self, batch: FrameBatch, ctx: ModuleInputs) -> ModuleOutput:
        saida = self._real.infer(batch, ctx)
        registros = ((self._registro,),) + ((),) * (len(saida.frames) - 1)
        return replace(saida, records=registros)

    def close(self) -> None:
        self._real.close()


def _roda_com_shot(clipe: Clip, registro: dict[str, object]) -> ClipResult:
    modulos = dict(load_modules(read_config(CONFIG)))
    modulos["shot"] = _ShotComContato(modulos["shot"], registro)  # pyright: ignore[reportArgumentType]
    provider = FrameProvider(VideoSource(clipe.path))
    try:
        return process_clip(
            clip=clipe,
            modules=modulos,
            framesets=read_framesets(CONFIG),
            provider=provider,
            extractors=(),
        )
    finally:
        provider.close()


def test_a_boolean_is_not_accepted_as_the_frame_of_the_contact(clipe: Clip) -> None:
    """`bool` e `int` em Python, entao `True` passaria como quadro 1 sem a guarda explicita.

    A linha que recusa booleano existe exatamente para isto e nenhum teste a exercitava: um
    modulo que devolvesse `frame_index: True` produziria evento apontando para o quadro 1, com
    `shot_timestamp_s` calculado em cima disso, e nada no relatorio diria que houve troca.
    """
    with pytest.raises(ContactOutOfRange, match="frame_index"):
        _roda_com_shot(clipe, {"frame_index": True})


def test_the_frame_right_after_the_last_one_is_out_of_range(clipe: Clip) -> None:
    """A fronteira exata: `frame_count` e um alem do ultimo indice valido.

    Aceito, ele vira `shot_timestamp_s` maior que a duracao do clipe e `shot_id` apontando para
    quadro que nao existe.
    """
    with pytest.raises(ContactOutOfRange, match=str(clipe.frame_count)):
        _roda_com_shot(clipe, {"frame_index": clipe.frame_count})


def test_the_track_of_whoever_shot_survives_until_the_features(clipe: Clip) -> None:
    """O identificador de quem chutou e o unico elo entre a percepcao e toda feature de atirador.

    Descartado, ele chega `None` a cada extractor e a feature degrada em silencio, o que fica
    indistinguivel de nao ter havido deteccao. Uma mutacao que o jogava fora nao derrubava teste
    nenhum.
    """
    vistos: list[int | None] = []

    class Espia:
        spec = FeatureSpec(name="espia", version="0.1.0", requires=())

        def extract(self, perception: Perception, shot: ShotOutput) -> FeatureValue:
            vistos.append(shot.shooter_track_id)
            return FeatureValue(
                name=self.spec.name,
                version=self.spec.version,
                status="ok",
                value=1.0,
                confidence=None,
                reason=None,
                evidence={},
            )

    modulos = dict(load_modules(read_config(CONFIG)))
    modulos["shot"] = _ShotComContato(  # pyright: ignore[reportArgumentType]
        modulos["shot"], {"frame_index": SHOT_FRAME_INDEX, "shooter_track_id": 77}
    )
    provider = FrameProvider(VideoSource(clipe.path))
    try:
        process_clip(
            clip=clipe,
            modules=modulos,
            framesets=read_framesets(CONFIG),
            provider=provider,
            extractors=(Espia(),),
        )
    finally:
        provider.close()

    assert vistos == [77], "o identificador de quem chutou nao chegou a feature"


def test_the_provenance_of_the_instant_survives_until_the_features(clipe: Clip) -> None:
    """Sem isto, o gate de ponto medio nunca dispara e ninguem ve.

    Uma feature que le a trajetoria antes do contato precisa saber se a janela esta ancorada em
    contato medido ou em ponto medio estimado, porque a segunda nao tem um antes. O `ClipResult` ja
    carregava a procedencia, mas ela parava ali: `run_features` passa so `perception` e `shot`.
    """
    vistos: list[tuple[str | None, float | None, str | None]] = []

    class Espia:
        spec = FeatureSpec(name="espia", version="0.1.0", requires=())

        def extract(self, perception: Perception, shot: ShotOutput) -> FeatureValue:
            p = shot_provenance(shot)
            vistos.append((p.source, p.confidence, p.ball_support))
            return FeatureValue(
                name=self.spec.name,
                version=self.spec.version,
                status="ok",
                value=1.0,
                confidence=None,
                reason=None,
                evidence={},
            )

    modulos = dict(load_modules(read_config(CONFIG)))
    modulos["shot"] = _ShotComContato(  # pyright: ignore[reportArgumentType]
        modulos["shot"],
        {
            "frame_index": SHOT_FRAME_INDEX,
            "source": "midpoint",
            "confidence": 0.05,
            "ball_support": "absent",
        },
    )
    provider = FrameProvider(VideoSource(clipe.path))
    try:
        process_clip(
            clip=clipe,
            modules=modulos,
            framesets=read_framesets(CONFIG),
            provider=provider,
            extractors=(Espia(),),
        )
    finally:
        provider.close()

    assert vistos == [("midpoint", 0.05, "absent")], (
        "a procedencia do instante nao chegou a feature"
    )


def test_the_run_can_tell_a_measured_instant_from_an_estimated_one(clipe: Clip) -> None:
    """Sem isto, cem clipes no ponto medio e cem medidos chegam ao relatorio com a mesma cara."""
    medido = _roda_com_shot(clipe, {"frame_index": SHOT_FRAME_INDEX, "source": "ball-break"})
    estimado = _roda_com_shot(clipe, {"frame_index": SHOT_FRAME_INDEX, "source": "midpoint"})

    assert medido.shot_source == "ball-break"
    assert estimado.shot_source == "midpoint"


def test_a_shot_record_without_source_does_not_invent_one(clipe: Clip) -> None:
    assert _roda_com_shot(clipe, {"frame_index": SHOT_FRAME_INDEX}).shot_source is None


def test_a_lista_de_candidatos_do_instante_chega_ao_resultado(clipe: Clip) -> None:
    """A execucao escreve a lista por clipe, e ela nasce no registro da especie `shot`."""
    resultado = _roda_com_shot(
        clipe,
        {
            "frame_index": SHOT_FRAME_INDEX,
            "source": "ball-break",
            "candidates": [{"frame_index": 10, "sample_index": 12, "score": 0.5}],
        },
    )

    assert resultado.shot_candidates == ({"frame_index": 10, "sample_index": 12, "score": 0.5},)


def test_candidatos_fora_da_forma_declarada_saem_como_ausentes(clipe: Clip) -> None:
    """Lista inventada aqui viraria diagnostico inventado la na frente.

    Ausencia significa "esta execucao nao declarou", e a taxonomia responde `nao-declarado`. Um
    `str` convertido para lista de caracteres, ou um numero embrulhado em lista, produziria
    candidato com quadro plausivel que nenhum modulo chegou a calcular.
    """
    resultado = _roda_com_shot(
        clipe,
        {"frame_index": SHOT_FRAME_INDEX, "source": "ball-break", "candidates": "duas"},
    )

    assert resultado.shot_candidates == ()


def test_registro_de_instante_sem_candidato_nao_inventa_lista(clipe: Clip) -> None:
    resultado = _roda_com_shot(clipe, {"frame_index": SHOT_FRAME_INDEX, "source": "ball-break"})

    assert resultado.shot_candidates == ()


def test_o_conjunto_de_quebras_chega_ao_resultado(clipe: Clip) -> None:
    """Sem ele, o teto de reordenar sai sem linha de base."""
    resultado = _roda_com_shot(
        clipe,
        {
            "frame_index": SHOT_FRAME_INDEX,
            "source": "ball-break",
            "candidate_pool": [10, 40, 70],
        },
    )

    assert resultado.shot_candidate_pool == (10, 40, 70)


def test_booleano_no_conjunto_de_quebras_nao_vira_quadro(clipe: Clip) -> None:
    """`bool` e subclasse de `int`, entao `True` entraria como quadro 1 e sujaria o acaso."""
    resultado = _roda_com_shot(
        clipe,
        {
            "frame_index": SHOT_FRAME_INDEX,
            "source": "ball-break",
            "candidate_pool": [10, True, "40", 70],
        },
    )

    assert resultado.shot_candidate_pool == (10, 70)
