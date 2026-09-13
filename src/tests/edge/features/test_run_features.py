"""A semantica de degradacao: uma feature quebrada nunca derruba a execucao."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from contracts import (
    SCHEMA_VERSION,
    ContractError,
    FeatureValue,
    Perception,
    PipelineMetadata,
    ShotEvent,
    make_shot_id,
    validate_shot_event,
)
from edge.features.base import FeatureError, FeatureSpec, ShotOutput, run_features

PERCEPTION = Perception(clip_id="clipe-1", frames=())

CHAMADAS: list[str] = []


@dataclass(frozen=True)
class Shot:
    frame_index: int = 187
    shooter_track_id: int | None = 9


@dataclass(frozen=True)
class Constante:
    spec: FeatureSpec = field(
        default_factory=lambda: FeatureSpec("constante", "0.1.0", ("players",))
    )
    valor: float = 1.0

    def extract(self, perception: Perception, shot: ShotOutput) -> FeatureValue:
        CHAMADAS.append(self.spec.name)
        return FeatureValue(
            name=self.spec.name,
            version=self.spec.version,
            status="ok",
            value=self.valor,
            confidence=0.9,
            reason=None,
            evidence={"frame_index": shot.frame_index},
        )


@dataclass(frozen=True)
class NaoDeveSerChamada:
    spec: FeatureSpec = field(
        default_factory=lambda: FeatureSpec("shot_distance_m", "0.1.0", ("field",))
    )

    def extract(self, perception: Perception, shot: ShotOutput) -> FeatureValue:
        CHAMADAS.append(self.spec.name)
        raise AssertionError("sem `field` produzido a plataforma nao chega a chamar o extractor")


@dataclass(frozen=True)
class Quebrada:
    spec: FeatureSpec = field(
        default_factory=lambda: FeatureSpec("shot_angle_deg", "0.1.0", ("players",))
    )

    def extract(self, perception: Perception, shot: ShotOutput) -> FeatureValue:
        return 1 / 0  # pyright: ignore[reportReturnType]


@dataclass(frozen=True)
class ComNumpy:
    spec: FeatureSpec = field(default_factory=lambda: FeatureSpec("numpy", "0.1.0", ()))

    def extract(self, perception: Perception, shot: ShotOutput) -> FeatureValue:
        return FeatureValue(
            name=self.spec.name,
            version=self.spec.version,
            status="ok",
            value=0.5,
            confidence=None,
            reason=None,
            evidence={"intermediario": np.float32(2.5)},
        )


@dataclass(frozen=True)
class MenteONome:
    spec: FeatureSpec = field(default_factory=lambda: FeatureSpec("declarado", "0.1.0", ()))

    def extract(self, perception: Perception, shot: ShotOutput) -> FeatureValue:
        return FeatureValue(
            name="outro",
            version="9.9.9",
            status="ok",
            value=1.0,
            confidence=None,
            reason=None,
            evidence={},
        )


@pytest.fixture(autouse=True)
def registro_limpo() -> None:
    CHAMADAS.clear()


def roda(extractors: list[object], produced: frozenset[str]) -> tuple[FeatureValue, ...]:
    return run_features(
        extractors,  # pyright: ignore[reportArgumentType]
        PERCEPTION,
        Shot(),
        produced,
    )


def test_missing_requirement_skips_extractor_with_reason() -> None:
    (valor,) = roda([NaoDeveSerChamada()], frozenset({"players"}))

    assert valor.status == "unavailable"
    assert valor.name == "shot_distance_m"
    assert "field" in (valor.reason or "")


def test_missing_requirement_does_not_call_the_extractor() -> None:
    (valor,) = roda([NaoDeveSerChamada()], frozenset())

    assert CHAMADAS == []
    assert valor.status == "unavailable"


def test_an_extractor_that_forgets_to_return_degrades_instead_of_killing_the_run() -> None:

    @dataclass(frozen=True)
    class SemReturn:
        spec: FeatureSpec = field(default_factory=lambda: FeatureSpec("esquecida", "0.1.0", ()))

        def extract(self, perception: Perception, shot: ShotOutput) -> FeatureValue:
            FeatureValue(
                name=self.spec.name,
                version=self.spec.version,
                status="ok",
                value=1.0,
                confidence=None,
                reason=None,
                evidence={},
            )
            return None  # pyright: ignore[reportReturnType]

    esquecida, constante = roda([SemReturn(), Constante()], frozenset({"players"}))

    assert esquecida.status == "unavailable"
    assert esquecida.reason is not None and esquecida.reason.startswith("AttributeError:")
    assert constante.status == "ok"


def test_extractor_exception_becomes_unavailable_with_type_and_message() -> None:
    (valor,) = roda([Quebrada()], frozenset({"players"}))

    assert valor.status == "unavailable"
    assert valor.reason == "ZeroDivisionError: division by zero"


def test_one_broken_extractor_does_not_stop_the_others() -> None:
    quebrada, constante = roda([Quebrada(), Constante()], frozenset({"players"}))

    assert quebrada.status == "unavailable"
    assert constante.status == "ok"
    assert constante.value == 1.0


def test_evidence_that_json_cannot_serialize_becomes_unavailable() -> None:
    (valor,) = roda([ComNumpy()], frozenset())

    assert valor.status == "unavailable"
    assert valor.reason is not None
    assert valor.reason.startswith("ContractError:")
    assert "float32" in valor.reason


def test_returned_value_must_match_the_declared_name_and_version() -> None:
    (valor,) = roda([MenteONome()], frozenset())

    assert valor.status == "unavailable"
    assert valor.name == "declarado"
    assert valor.version == "0.1.0"
    assert "outro" in (valor.reason or "")


def test_duplicate_feature_name_is_refused_before_any_extractor_runs() -> None:
    with pytest.raises(FeatureError, match="constante"):
        roda([Constante(), Constante(valor=2.0)], frozenset({"players"}))

    assert CHAMADAS == []


def test_frozen_shot_double_is_accepted() -> None:
    shot: ShotOutput = Shot()

    assert (shot.frame_index, shot.shooter_track_id) == (187, 9)


def test_result_order_follows_the_declared_selection() -> None:
    valores = roda([Constante(), NaoDeveSerChamada(), Quebrada()], frozenset({"players"}))

    assert [valor.name for valor in valores] == ["constante", "shot_distance_m", "shot_angle_deg"]


def test_degraded_values_pass_the_event_schema() -> None:
    valores = roda([NaoDeveSerChamada(), Quebrada()], frozenset({"players", "ball"}))

    evento = ShotEvent(
        schema_version=SCHEMA_VERSION,
        shot_id=make_shot_id("clipe-1", 187),
        clip_id="clipe-1",
        match_id=None,
        shot_timestamp_s=7.48,
        features=valores,
        pipeline=PipelineMetadata(
            versions={"features": "0.1.0"},
            timings_ms={"features": 0.4},
            host="prova",
            started_at="2026-08-21T00:00:00Z",
        ),
    )

    validate_shot_event(evento.to_dict())
    assert all(valor.status == "unavailable" for valor in evento.features)


def test_an_empty_selection_yields_an_event_without_features() -> None:
    """Plataforma sem feature instalada entrega evento sem feature, e nao evento inventado.

    A configuracao vem escrita aqui e nao de `config/pipeline.toml`. Ate a entrega da primeira
    feature as duas coincidiam, e o teste passava por acidente: hoje a producao declara
    `[features.body_part]`, e ler de la faria este teste medir o estado do arquivo em vez do
    comportamento do carregador.
    """
    from edge.features.registry import load_extractors, read_feature_config

    selecao = load_extractors(read_feature_config('[modules.shot]\nimpl = "x:Y"\n'))

    assert selecao == ()
    assert roda(list(selecao), frozenset({"players", "ball", "field"})) == ()


def test_the_production_configuration_loads_every_feature_it_declares() -> None:
    """O outro lado: o que `config/pipeline.toml` declara precisa existir e carregar."""
    from pathlib import Path

    from edge.features.registry import load_extractors, read_feature_config

    selecao = load_extractors(read_feature_config(Path("config/pipeline.toml")))

    assert {e.spec.name for e in selecao} == {
        "assist",
        "body_part",
        "crowded",
        "marked",
        "type_of_play",
    }


def test_contract_error_is_what_the_platform_catches_for_bad_evidence() -> None:
    with pytest.raises(ContractError):
        FeatureValue(
            name="x",
            version="0.1.0",
            status="ok",
            value=1.0,
            confidence=None,
            reason=None,
            evidence={"n": np.float32(1.0)},
        )
