"""A tabela que diz qual coluna do parceiro e o gabarito de cada metrica."""

from __future__ import annotations

import pytest

from mlops.labels import LabelError, Labels, labels_config, read_labels

TABELA = """
schema_version = "1.0"

[[label]]
name = "marked"
table = "match_shots_xg"
column = "marked"
type = "bool"
domain = ["true", "false"]
missing = [""]

[[label]]
name = "body_part_3"
table = "match_shots"
column = "body_part"
type = "enum"
domain = ["CABECA", "PERNA_ESQUERDA", "PERNA_DIREITA"]
missing = [""]

[[label]]
name = "shot_distance_m"
table = "match_shots"
column = "meters"
type = "numeric"
grain = 5.0
missing = [""]

[[metric]]
feature = "atacante_marcado"
label = "marked"

[[metric]]
feature = "marcado_por_regra"
label = "marked"
"""


def _lidas() -> Labels:
    return read_labels(TABELA)


def test_a_label_declares_where_the_ground_truth_lives() -> None:
    marked = _lidas().label("marked")

    assert (marked.table, marked.column, marked.type) == ("match_shots_xg", "marked", "bool")


def test_two_features_may_answer_the_same_label() -> None:
    """A planning da "olhou pra bola" para duas pessoas, entao a estrutura nao pode proibir."""
    assert _lidas().features_for("marked") == ("atacante_marcado", "marcado_por_regra")


def test_a_bool_label_reads_the_partner_text_and_not_python_truthiness() -> None:
    """As colunas booleanas sao o texto 'true' e 'false', e `bool('false')` e True."""
    marked = _lidas().label("marked")

    assert marked.parse("true") is True
    assert marked.parse("false") is False
    assert marked.parse("") is None


def test_an_enum_label_refuses_a_value_outside_the_declared_domain() -> None:
    parte = _lidas().label("body_part_3")

    assert parte.parse("CABECA") == "CABECA"
    with pytest.raises(LabelError, match="PERNA_TORTA"):
        parte.parse("PERNA_TORTA")


def test_a_numeric_label_carries_the_grain_the_partner_measured_at() -> None:
    """MAE abaixo de metade do grao mede o rotulo, e nao o modelo, e o relatorio precisa saber."""
    distancia = _lidas().label("shot_distance_m")

    assert distancia.grain == 5.0
    assert distancia.parse("15") == 15.0


def test_a_metric_pointing_at_a_label_that_does_not_exist_is_refused() -> None:
    with pytest.raises(LabelError, match="fantasma"):
        read_labels(TABELA + '\n[[metric]]\nfeature = "x"\nlabel = "fantasma"\n')


def test_two_labels_with_the_same_name_are_refused() -> None:
    duplicada = '\n[[label]]\nname = "marked"\ntable = "t"\ncolumn = "c"\ntype = "bool"\n'

    with pytest.raises(LabelError, match="marked"):
        read_labels(TABELA + duplicada)


def test_the_versioned_table_matches_the_columns_the_partner_actually_ships() -> None:
    """O arquivo do repositorio precisa apontar para coluna que existe, senao o eval mente."""
    tabela = read_labels(labels_config())

    assert tabela.label("body_part_3").column == "body_part"
    assert tabela.label("body_part_2").column == "used"
    assert tabela.label("shot_distance_m").grain == 5.0


def test_the_required_labels_come_from_the_table_and_not_from_code() -> None:
    """Sete nomes cravados no CLI duplicavam a tabela, e tirar um la mudava a contagem calado."""
    obrigatorios = {rotulo.name for rotulo in read_labels(labels_config()).required()}

    assert obrigatorios == {
        "goalkeeper_position",
        "occupied_area",
        "marked",
        "looked_before",
        "assist",
        "body_part_3",
        "type_of_play",
    }


def test_every_declared_domain_is_non_empty_for_enum_labels() -> None:
    """Encolher um dominio passava calado e faria 994 linhas reais levantarem LabelError."""
    for rotulo in read_labels(labels_config()).labels.values():
        if rotulo.type == "enum":
            assert rotulo.domain, f"{rotulo.name} e enum e nao declara dominio"
        if rotulo.type == "bool":
            assert set(rotulo.domain) == {"true", "false"}, rotulo.name
        assert rotulo.missing, f"{rotulo.name} sem `missing` conta ausente como valor"


def test_a_bool_label_refuses_text_that_is_not_the_partner_vocabulary() -> None:
    marked = _lidas().label("marked")

    with pytest.raises(LabelError, match="1"):
        marked.parse("1")


def test_the_versioned_table_declares_the_domains_the_partner_actually_ships() -> None:
    """Encolher um dominio passava calado, e faria 994 linhas reais levantarem LabelError."""
    tabela = read_labels(labels_config())

    assert set(tabela.label("type_of_play").domain) == {"open play", "set piece", "individual"}
    assert set(tabela.label("body_part_3").domain) == {
        "CABECA",
        "PERNA_ESQUERDA",
        "PERNA_DIREITA",
    }
    assert set(tabela.label("body_part_2").domain) == {"foot", "head"}


def test_the_versioned_table_carries_every_label_the_project_promised() -> None:
    """Renomear um rotulo passava calado, e quebra a juncao de quem ja escreveu contra ele."""
    assert set(read_labels(labels_config()).labels) == {
        "goalkeeper_position",
        "occupied_area",
        "marked",
        "looked_before",
        "assist",
        "body_part_3",
        "body_part_2",
        "type_of_play",
        "shot_distance_m",
        "expected_goals",
        "is_goal",
    }
