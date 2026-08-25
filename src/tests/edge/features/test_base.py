"""O contrato do extractor: o que uma feature declara e como ela e recusada na construcao."""

from __future__ import annotations

import pytest

from edge.features.base import FeatureError, FeatureSpec, unavailable


def test_unknown_requirement_is_refused_at_construction() -> None:
    with pytest.raises(FeatureError, match="fieldd"):
        FeatureSpec(name="shot_distance_m", version="0.1.0", requires=("fieldd",))


def test_every_module_kind_is_accepted_as_a_requirement() -> None:
    spec = FeatureSpec(name="tudo", version="0.1.0", requires=("field", "players", "shot"))

    assert spec.requires == ("field", "players", "shot")


def test_a_feature_can_require_nothing() -> None:
    assert FeatureSpec(name="constante", version="0.1.0", requires=()).requires == ()


def test_the_spec_carries_only_what_the_feature_knows_about_itself() -> None:
    """`unit` entrou porque a fronteira HTTP precisa dela e adivinhar seria pior.

    Derivar metro de um sufixo `_m` no nome quebra na primeira feature que nao seguir a
    convencao, e uma tabela na fronteira ficaria longe de quem sabe a resposta.
    """
    assert set(FeatureSpec.__dataclass_fields__) == {"name", "version", "requires", "unit"}


def test_a_feature_without_a_declared_unit_is_dimensionless() -> None:
    """Booleano e classe nao tem unidade, e a maioria das metricas do projeto e booleana."""
    assert FeatureSpec(name="marcado", version="0.1.0", requires=()).unit == ""


def test_unavailable_carries_the_declared_name_version_and_the_reason() -> None:
    spec = FeatureSpec(name="shot_angle_deg", version="0.2.0", requires=("field",))

    valor = unavailable(spec, "modulo field nao rodou")

    assert (valor.name, valor.version, valor.status) == ("shot_angle_deg", "0.2.0", "unavailable")
    assert valor.value is None
    assert valor.confidence is None
    assert valor.reason == "modulo field nao rodou"
    assert dict(valor.evidence) == {}
