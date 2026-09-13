from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from cli import (
    _valor_para_avaliacao as _valor_para_avaliacao,  # pyright: ignore[reportPrivateUsage]
)
from mlops.labels import read_labels


def test_avaliacao_preserva_escore_da_feature_booleana() -> None:
    labels = read_labels(Path("config/labels.toml"))
    feature: Mapping[str, object] = {
        "name": "looked_before",
        "status": "ok",
        "value": True,
        "evidence": {"score": 0.60143},
    }

    assert _valor_para_avaliacao(feature, labels) == 0.60143


def test_avaliacao_mantem_valor_de_feature_sem_escore() -> None:
    labels = read_labels(Path("config/labels.toml"))
    feature: Mapping[str, object] = {
        "name": "type_of_play",
        "status": "ok",
        "value": "open play",
        "evidence": {},
    }

    assert _valor_para_avaliacao(feature, labels) == "open play"


def test_avaliacao_mantem_indisponibilidade_acima_do_escore() -> None:
    labels = read_labels(Path("config/labels.toml"))
    feature: Mapping[str, object] = {
        "name": "goalkeeper_position",
        "status": "unavailable",
        "value": None,
        "evidence": {"score": 0.9},
    }

    assert _valor_para_avaliacao(feature, labels) is None
