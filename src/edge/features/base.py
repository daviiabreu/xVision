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
import math
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


@dataclass(frozen=True)
class ShotProvenance:
    """A procedencia do instante, lida do registro de `shot` sem exigi-la do protocolo.

    Ausente e ausente: `None` em cada campo significa que o detector nao declarou, e nao que ele
    declarou o pior caso nem o melhor. Uma feature que trate `None` como `midpoint` recusaria clipe
    por falta de declaracao, e uma que trate como `detected` publicaria confianca que ninguem mediu.
    """

    source: str | None
    confidence: float | None
    ball_support: str | None


def _confianca(valor: object) -> float | None:
    """Confianca so vale se for numero real e finito, e fora disso e ausencia declarada."""
    if isinstance(valor, bool) or not isinstance(valor, int | float):
        return None
    if not math.isfinite(valor):
        return None
    return float(valor)


def shot_provenance(shot: ShotOutput) -> ShotProvenance:
    """Le `source`, `confidence` e `ball_support` de um `ShotOutput` que talvez nao os tenha.

    Por `getattr`, e nao por membro do `Protocol`. Membro novo obrigatorio no `ShotOutput` quebraria
    na checagem de tipo os onze dubles que o proprio projeto usa nos testes e no pipeline, e o custo
    disso nao se paga para um campo que so uma feature consome. O tipo e conferido em vez de
    convertido: `str()` de um objeto qualquer produziria procedencia plausivel e inventada.

    Booleano nao e confianca. `bool` e subclasse de `int`, entao um `true` no registro passaria
    por confianca maxima, que e o valor mais forte que existe saindo do erro de tipo mais bobo.

    Nem `nan` nem infinito. `json.loads` aceita o token `NaN` por padrao, entao um `nan` num
    registro ou num arquivo de cache chega aqui inteiro, e `valor is None` nao pega `nan`: ele
    atravessa para `PlayEvidence.shot_confidence` e, na fatia 5, um `nan` num logit zera o softmax
    inteiro. O modulo vizinho `edge/features/touch.py` ja tem exatamente esta guarda, com o mesmo
    motivo, e nao tê-la aqui era assimetria e nao decisão.
    """
    origem = getattr(shot, "source", None)
    confianca = getattr(shot, "confidence", None)
    apoio = getattr(shot, "ball_support", None)
    return ShotProvenance(
        source=origem if isinstance(origem, str) else None,
        confidence=_confianca(confianca),
        ball_support=apoio if isinstance(apoio, str) else None,
    )


class FeatureExtractor(Protocol):
    spec: FeatureSpec

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
