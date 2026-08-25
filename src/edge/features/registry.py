"""Selecao de features declarada em `config/pipeline.toml`, e nao numa tupla de codigo.

Acrescentar uma feature e uma secao no arquivo, e feature que nao esta la nao roda. A diferenca
para um registro por decorador nao e o modo de falha, que e o mesmo sumico silencioso: e que a
omissao aparece no diff da revisao em vez de depender de um import ter acontecido. A ordem das
secoes e a ordem do `features` do evento.

O mecanismo e o mesmo que `edge.modules.registry` ja usa para escolher implementacao de modulo,
de proposito: sete pessoas editando linhas diferentes de um TOML conflitam menos que sete
editando a mesma tupla num arquivo Python.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import cast

from edge.features.base import FeatureError, FeatureExtractor

if sys.version_info >= (3, 11):  # pragma: no cover - o piso e 3.10, e e nele que a suite roda
    import tomllib
else:
    import tomli as tomllib


class UnknownFeature(FeatureError): ...


@dataclass(frozen=True)
class FeatureConfig:
    entry: str
    impl: str


def read_feature_config(source: str | Path) -> tuple[FeatureConfig, ...]:
    texto = source.read_text(encoding="utf-8") if isinstance(source, Path) else source
    documento = cast("dict[str, object]", tomllib.loads(texto))
    entradas = cast("dict[str, dict[str, object]]", documento.get("features", {}))
    configuracoes: list[FeatureConfig] = []
    for entry, corpo in entradas.items():
        impl = corpo.get("impl")
        if not isinstance(impl, str):
            raise UnknownFeature(f"features.{entry} precisa declarar impl")
        configuracoes.append(FeatureConfig(entry=entry, impl=impl))
    return tuple(configuracoes)


def load_extractors(configs: tuple[FeatureConfig, ...]) -> tuple[FeatureExtractor, ...]:
    extractors: list[FeatureExtractor] = []
    for config in configs:
        extractor = _resolve(config)()
        if extractor.spec.name != config.entry:
            raise UnknownFeature(
                f"features.{config.entry} carrega uma feature chamada {extractor.spec.name!r}. "
                "O nome da secao e o nome que sai no evento, entao os dois precisam bater"
            )
        extractors.append(extractor)
    return tuple(extractors)


def _resolve(config: FeatureConfig) -> Callable[[], FeatureExtractor]:
    caminho, _, atributo = config.impl.partition(":")
    if not atributo:
        raise UnknownFeature(
            f"features.{config.entry} aponta para {config.impl!r}, e falta o nome da classe "
            "depois do dois-pontos"
        )
    try:
        return cast("Callable[[], FeatureExtractor]", getattr(import_module(caminho), atributo))
    except (ImportError, AttributeError) as erro:
        raise UnknownFeature(
            f"features.{config.entry} aponta para {config.impl!r}: {erro}"
        ) from erro
