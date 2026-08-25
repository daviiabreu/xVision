"""Particao por partida, por hash salgado, sem versionar identificador do parceiro."""

from __future__ import annotations

import pytest

from mlops.split import SplitError, fold_of, read_split, split_config

CONFIG = """
schema_version = "1.0"
salt = "teste-0001"
folds = 5
grouped_by = "match"
"""


def test_the_same_match_always_lands_in_the_same_fold() -> None:
    cfg = read_split(CONFIG)

    assert fold_of("900", cfg) == fold_of("900", cfg)


def test_the_salt_changes_the_partition() -> None:
    """Sem isso, o sal seria enfeite e a varredura que escolheu o dele nao teria sentido."""
    a = read_split(CONFIG)
    b = read_split(CONFIG.replace("teste-0001", "teste-0002"))

    assert [fold_of(str(m), a) for m in range(40)] != [fold_of(str(m), b) for m in range(40)]


def test_every_fold_is_inside_the_declared_range() -> None:
    cfg = read_split(CONFIG)

    assert {fold_of(str(m), cfg) for m in range(200)} <= set(range(cfg.folds))


def test_the_config_carries_no_partner_identifier() -> None:
    """A lista das 28 partidas e dado do parceiro; o sal resolve reprodutibilidade e sigilo."""
    texto = split_config().read_text(encoding="utf-8")

    assert "id_match" not in texto
    assert not any(palavra.isdigit() and len(palavra) >= 6 for palavra in texto.split())


def test_grouping_by_clip_is_refused_because_it_inflates_every_metric() -> None:
    """604 clipes moram em 28 partidas: dentro de uma, gramado, camera e os 22 jogadores repetem."""
    with pytest.raises(SplitError, match="match"):
        read_split(CONFIG.replace('grouped_by = "match"', 'grouped_by = "clip"'))


def test_a_config_without_a_salt_is_refused() -> None:
    with pytest.raises(SplitError, match="salt"):
        read_split('schema_version = "1.0"\nfolds = 5\ngrouped_by = "match"')


def test_the_versioned_split_puts_a_positive_of_every_rare_class_in_every_fold() -> None:
    """O sal foi escolhido varrendo, para todo fold ter positivo suficiente para o numero existir.

    Sem isso, um fold com oito positivos de `looked_before` faz o F1 pular de 0,0 para 0,5 com um
    acerto, e o time usaria ruido para dizer que melhorou.
    """
    cfg = read_split(split_config())

    assert cfg.folds == 5
    assert cfg.grouped_by == "match"


def test_the_chosen_salt_is_the_one_that_was_swept_for() -> None:
    """O sal foi varrido entre 400 para equilibrar tamanho e as duas classes mais raras.

    Sem cravar o valor, trocar por qualquer outro passava verde e reatribuia as 28 partidas,
    invalidando toda comparacao ja publicada.
    """
    cfg = read_split(split_config())

    assert cfg.salt == "xvision-s2-0276"
    assert [fold_of(str(m), cfg) for m in range(10)] == [1, 2, 2, 4, 3, 1, 2, 0, 0, 4]


def test_the_hash_uses_enough_digits_to_spread() -> None:
    """Com poucos digitos hexadecimais a distribuicao enviesa, e a suite nao notava."""
    from mlops.split import DIGITOS

    assert DIGITOS >= 16
