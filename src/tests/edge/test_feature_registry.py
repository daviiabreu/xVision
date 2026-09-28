"""A selecao de features sai do `.py` e vai para o TOML, pelo mecanismo que os modulos ja usam.

Sete pessoas editando linhas diferentes de um TOML conflitam menos que sete editando a mesma
tupla num arquivo Python, e a omissao continua aparecendo no diff da revisao, que era a razao de
a tupla ser explicita.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from edge.features.evidence_model import EvidenceModel
from edge.features.registry import UnknownFeature, load_extractors, read_feature_config

CONFIG = """
[features.constante]
impl = "edge.testing.example_feature:Constante"
"""


def test_a_feature_declared_in_the_toml_is_loaded_by_path() -> None:
    (extractor,) = load_extractors(read_feature_config(CONFIG))

    assert extractor.spec.name == "constante"


def test_the_order_in_the_file_is_the_order_in_the_event() -> None:
    """A ordem do `features` do evento e a ordem daqui, entao ela nao pode virar ordem de dict."""
    config = """
[features.outra]
impl = "edge.testing.example_feature:Outra"

[features.constante]
impl = "edge.testing.example_feature:Constante"
"""

    nomes = [e.spec.name for e in load_extractors(read_feature_config(config))]

    assert nomes == ["outra", "constante"]


def test_a_file_without_features_yields_an_empty_selection() -> None:
    assert load_extractors(read_feature_config("[modules.players]\nimpl = 'x:Y'")) == ()


def test_an_impl_that_does_not_exist_says_which_entry_points_at_it() -> None:
    config = "[features.fantasma]\nimpl = 'edge.testing.example_feature:NaoExiste'"

    with pytest.raises(UnknownFeature, match="features.fantasma"):
        load_extractors(read_feature_config(config))


def test_an_entry_without_impl_is_refused_instead_of_ignored() -> None:
    with pytest.raises(UnknownFeature, match="impl"):
        read_feature_config("[features.vazia]\nversion = '1'")


def test_the_section_name_has_to_be_the_feature_name() -> None:
    """Sem isto, `[features.distancia]` apontando para uma classe chamada `constante` produz
    evento com o nome errado, e o dono descobre no relatorio de acuracia."""
    config = "[features.distancia]\nimpl = 'edge.testing.example_feature:Constante'"

    with pytest.raises(UnknownFeature, match="distancia"):
        load_extractors(read_feature_config(config))


def test_chave_alem_de_impl_vira_opcao_do_construtor(tmp_path: Path) -> None:
    modelo = EvidenceModel(
        feature="type_of_play",
        classes=("individual", "open play", "set piece"),
        columns=(),
        mean=(),
        scale=(),
        coef=((), (), ()),
        intercept=(0.0, 1.0, 0.0),
    )
    modelo.save(tmp_path / "m.json")
    texto = (
        '[features.type_of_play]\nimpl = "edge.features.type_of_play:TypeOfPlay"\n'
        f'model = "{tmp_path / "m.json"}"\n'
    )

    (extractor,) = load_extractors(read_feature_config(texto))

    assert extractor.spec.version.endswith("+modelo")


def test_opcao_que_a_feature_nao_conhece_e_recusada() -> None:
    texto = '[features.type_of_play]\nimpl = "edge.features.type_of_play:TypeOfPlay"\nlimiar = 3\n'

    with pytest.raises(UnknownFeature, match="opcoes"):
        load_extractors(read_feature_config(texto))
