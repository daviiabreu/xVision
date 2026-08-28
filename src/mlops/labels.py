"""Qual coluna do parceiro e o gabarito de cada metrica, declarado onde o dono edita.

Sem este registro, sete pessoas nomeiam sete coisas diferentes e a juncao com o gabarito vira
trabalho manual no fim da sprint.

Duas regras existem para nao decidir pelo dono. Pe ou cabeca tem duas colunas de gabarito, uma
com tres classes e outra com duas, e escolher uma delas seria escolher o espaco de saida da
metrica: as duas entram, e o dono aponta a feature para a que usar. E o mapeamento de feature para
rotulo e muitos para um, porque a planning da a mesma metrica para duas pessoas: implementacoes
concorrentes tem nomes diferentes, rodam na mesma execucao e aparecem lado a lado no relatorio.

Nao calcula metrica, nao implementa feature, nao escolhe limiar e nao traduz vocabulario. O `type`
descreve o gabarito, e nao a saida do extractor.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from edge.paths import config_file

if sys.version_info >= (3, 11):  # pragma: no cover - o piso e 3.10, e e nele que a suite roda
    import tomllib
else:
    import tomli as tomllib

LABELS_NOME = "labels.toml"


def labels_config() -> Path:
    """Funcao, e nao constante: congelar no import faz variavel de ambiente nao ter efeito."""
    return config_file(LABELS_NOME)


TIPOS = frozenset({"bool", "enum", "numeric"})
VERDADEIRO = "true"
FALSO = "false"


class LabelError(ValueError): ...


@dataclass(frozen=True)
class Label:
    name: str
    table: str
    column: str
    type: str
    domain: tuple[str, ...] = ()
    missing: tuple[str, ...] = ("",)
    grain: float | None = None
    required: bool = False

    def __post_init__(self) -> None:
        if self.type not in TIPOS:
            raise LabelError(
                f"rotulo {self.name!r} com type {self.type!r}, e os conhecidos sao "
                f"{', '.join(sorted(TIPOS))}"
            )

    def parse(self, texto: str) -> bool | str | float | None:
        if texto in self.missing:
            return None
        if self.type == "bool":
            if texto not in (VERDADEIRO, FALSO):
                raise LabelError(
                    f"rotulo {self.name!r} recebeu {texto!r}, e a coluna booleana do parceiro "
                    f"traz {VERDADEIRO!r} ou {FALSO!r}"
                )
            return texto == VERDADEIRO
        if self.type == "enum":
            if self.domain and texto not in self.domain:
                raise LabelError(
                    f"rotulo {self.name!r} recebeu {texto!r}, fora do dominio declarado "
                    f"{', '.join(self.domain)}"
                )
            return texto
        try:
            return float(texto)
        except ValueError as erro:
            raise LabelError(f"rotulo {self.name!r} recebeu {texto!r}, que nao e numero") from erro


@dataclass(frozen=True)
class Metric:
    """Uma metrica: a feature, a coluna que a mede, e a barra contra a qual ela e julgada.

    `baseline_metadata` e a fracao que o metadado do proprio corte ja acerta sozinho, sem tocar em
    video. Ela existe porque tres dos rotulos declarados ja estao resolvidos por tabela, e comparar
    uma feature de video so contra a maioria publicaria como ganho o que ninguem comprou: medido,
    `cortes_metadados.foot` reproduz `body_part` em 633 de 633, e `origin_of_move` acerta
    `type_of_play` em 485 de 609 com o melhor mapa possivel.

    `None` quando nao ha coluna tabular que resolva o rotulo, que e o caso das cinco restantes.
    """

    feature: str
    label: str
    baseline_metadata: float | None = None

    def __post_init__(self) -> None:
        if self.baseline_metadata is not None and not 0.0 <= self.baseline_metadata <= 1.0:
            raise LabelError(
                f"a metrica {self.feature!r} declara baseline_metadata "
                f"{self.baseline_metadata!r}, e ela e uma fracao entre 0 e 1"
            )


@dataclass(frozen=True)
class Labels:
    labels: Mapping[str, Label]
    metrics: Sequence[Metric] = field(default_factory=tuple)

    def label(self, name: str) -> Label:
        try:
            return self.labels[name]
        except KeyError as erro:
            raise LabelError(
                f"rotulo {name!r} nao esta na tabela: {', '.join(sorted(self.labels))}"
            ) from erro

    def features_for(self, label: str) -> tuple[str, ...]:
        return tuple(m.feature for m in self.metrics if m.label == label)

    def metric_for(self, feature: str) -> Metric | None:
        return next((m for m in self.metrics if m.feature == feature), None)

    def required(self) -> tuple[Label, ...]:
        return tuple(rotulo for rotulo in self.labels.values() if rotulo.required)

    def label_for(self, feature: str) -> Label | None:
        metrica = self.metric_for(feature)
        return None if metrica is None else self.label(metrica.label)


def read_labels(source: str | Path) -> Labels:
    texto = source.read_text(encoding="utf-8") if isinstance(source, Path) else source
    documento = cast("dict[str, object]", tomllib.loads(texto))
    rotulos: dict[str, Label] = {}
    for entrada in cast("list[dict[str, object]]", documento.get("label", [])):
        rotulo = _label(entrada)
        if rotulo.name in rotulos:
            raise LabelError(f"rotulo {rotulo.name!r} declarado duas vezes")
        rotulos[rotulo.name] = rotulo
    metricas: list[Metric] = []
    for entrada in cast("list[dict[str, object]]", documento.get("metric", [])):
        feature, alvo = entrada.get("feature"), entrada.get("label")
        if not isinstance(feature, str) or not isinstance(alvo, str):
            raise LabelError("cada metric precisa declarar feature e label, os dois texto")
        if alvo not in rotulos:
            raise LabelError(f"a metrica {feature!r} aponta para o rotulo {alvo!r}, que nao existe")
        base = entrada.get("baseline_metadata")
        if base is not None and not isinstance(base, (int, float)):
            raise LabelError(
                f"a metrica {feature!r} declara baseline_metadata {base!r}, e ele e um numero"
            )
        metricas.append(
            Metric(
                feature=feature, label=alvo, baseline_metadata=None if base is None else float(base)
            )
        )
    return Labels(labels=rotulos, metrics=tuple(metricas))


def _label(entrada: dict[str, object]) -> Label:
    faltando = {"name", "table", "column", "type"} - set(entrada)
    if faltando:
        raise LabelError(f"um label sem {', '.join(sorted(faltando))}")
    grao = entrada.get("grain")
    return Label(
        name=cast("str", entrada["name"]),
        table=cast("str", entrada["table"]),
        column=cast("str", entrada["column"]),
        type=cast("str", entrada["type"]),
        domain=tuple(cast("list[str]", entrada.get("domain", []))),
        missing=tuple(cast("list[str]", entrada.get("missing", [""]))),
        grain=None if grao is None else float(cast("float", grao)),
        required=bool(entrada.get("required", False)),
    )
