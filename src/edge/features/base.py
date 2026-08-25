"""O contrato do extractor e a semantica de degradacao, sem nenhuma feature dentro.

Duas regras viram codigo aqui. Feature que estoura nao derruba a execucao, porque uma medida
quebrada nao pode custar as outras do mesmo clipe. E feature que nao pode medir sai
`unavailable` com motivo, nunca zero e nunca silencio, porque numero inventado que passa no
schema e pior que ausencia declarada: ele chega ao consumidor com a mesma cara de um numero
medido.

Nao existe registro por decorador. Registry por efeito colateral de import faz uma feature nao
importada sumir em silencio, que e exatamente o modo de falha que a degradacao existe para
evitar. A selecao e a tupla explicita de `edge.features`.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Protocol

from contracts import FeatureValue, Perception
from edge.modules.base import MODULE_KINDS


class FeatureError(Exception): ...


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    version: str
    requires: tuple[str, ...]
    unit: str = ""

    def __post_init__(self) -> None:
        desconhecidas = tuple(kind for kind in self.requires if kind not in MODULE_KINDS)
        if desconhecidas:
            raise FeatureError(
                f"feature {self.name!r} declara requires {desconhecidas}, "
                f"e as especies conhecidas sao {', '.join(sorted(MODULE_KINDS))}"
            )


class ShotOutput(Protocol):
    @property
    def frame_index(self) -> int: ...

    @property
    def shooter_track_id(self) -> int | None: ...


class FeatureExtractor(Protocol):
    @property
    def spec(self) -> FeatureSpec: ...

    def extract(self, perception: Perception, shot: ShotOutput) -> FeatureValue: ...


def unavailable(spec: FeatureSpec, reason: str) -> FeatureValue:
    return FeatureValue(
        name=spec.name,
        version=spec.version,
        status="unavailable",
        value=None,
        confidence=None,
        reason=reason,
        evidence={},
    )


def run_features(
    extractors: Sequence[FeatureExtractor],
    perception: Perception,
    shot: ShotOutput,
    produced: Collection[str],
) -> tuple[FeatureValue, ...]:
    _refuse_duplicates(extractors)
    return tuple(_run_one(extractor, perception, shot, produced) for extractor in extractors)


def _refuse_duplicates(extractors: Sequence[FeatureExtractor]) -> None:
    vistos: set[str] = set()
    for extractor in extractors:
        nome = extractor.spec.name
        if nome in vistos:
            raise FeatureError(f"a selecao repete a feature {nome!r}, e o evento nao aceita duas")
        vistos.add(nome)


def _run_one(
    extractor: FeatureExtractor,
    perception: Perception,
    shot: ShotOutput,
    produced: Collection[str],
) -> FeatureValue:
    spec = extractor.spec
    faltando = tuple(kind for kind in spec.requires if kind not in produced)
    if faltando:
        return unavailable(spec, f"modulo {', '.join(faltando)} nao produziu saida nesta execucao")
    try:
        valor = extractor.extract(perception, shot)
        declarado = (valor.name, valor.version)
    except Exception as erro:
        logging.getLogger(__name__).exception("feature %s falhou", spec.name)
        return unavailable(spec, f"{type(erro).__name__}: {erro}")
    if declarado != (spec.name, spec.version):
        return unavailable(
            spec,
            f"o extractor devolveu {valor.name}@{valor.version} e declarou "
            f"{spec.name}@{spec.version}",
        )
    return valor
