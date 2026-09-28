"""Modelo linear sobre a evidencia de uma feature: colunas declaradas, coeficientes e softmax."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

VALOR = "__value__"
"""Caminho que le o valor publicado pela feature, e nao uma chave da evidencia."""

Tipo = Literal["number", "missing", "equals"]
Amostra = tuple[object, Mapping[str, object]]


@dataclass(frozen=True)
class Column:
    path: tuple[str, ...]
    kind: Tipo
    value: str | None = None

    @property
    def name(self) -> str:
        base = ".".join(self.path)
        return base if self.kind == "number" else f"{base}:{self.value or 'ausente'}"

    def read(self, valor: object, evidencia: Mapping[str, object]) -> float:
        bruto = _caminho(valor, evidencia, self.path)
        if self.kind == "missing":
            return 1.0 if bruto is None else 0.0
        if self.kind == "equals":
            return 1.0 if bruto is not None and str(bruto) == self.value else 0.0
        if isinstance(bruto, bool):
            return float(bruto)
        if isinstance(bruto, int | float) and math.isfinite(bruto):
            return float(bruto)
        return 0.0


@dataclass(frozen=True)
class EvidenceModel:
    feature: str
    classes: tuple[str, ...]
    columns: tuple[Column, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    coef: tuple[tuple[float, ...], ...]
    intercept: tuple[float, ...]

    def matrix(self, amostras: Sequence[Amostra]) -> np.ndarray[Any, np.dtype[np.float64]]:
        """As colunas cruas, uma linha por amostra, antes de padronizar."""
        return np.array(
            [
                [coluna.read(valor, evidencia) for coluna in self.columns]
                for valor, evidencia in amostras
            ],
            dtype=np.float64,
        ).reshape(len(amostras), len(self.columns))

    def probabilities(self, amostras: Sequence[Amostra]) -> np.ndarray[Any, np.dtype[np.float64]]:
        padronizado = (self.matrix(amostras) - np.asarray(self.mean)) / np.asarray(self.scale)
        return softmax(padronizado @ np.asarray(self.coef).T + np.asarray(self.intercept))

    def predict(self, amostras: Sequence[Amostra]) -> list[str]:
        return [self.classes[int(i)] for i in self.probabilities(amostras).argmax(axis=1)]

    def to_dict(self) -> dict[str, object]:
        return {
            "feature": self.feature,
            "classes": list(self.classes),
            "columns": [
                {"path": list(c.path), "kind": c.kind, "value": c.value} for c in self.columns
            ],
            "mean": list(self.mean),
            "scale": list(self.scale),
            "coef": [list(linha) for linha in self.coef],
            "intercept": list(self.intercept),
        }

    @classmethod
    def from_dict(cls, dados: Mapping[str, Any]) -> EvidenceModel:
        return cls(
            feature=str(dados["feature"]),
            classes=tuple(str(c) for c in dados["classes"]),
            columns=tuple(
                Column(tuple(str(p) for p in c["path"]), cast("Tipo", c["kind"]), c.get("value"))
                for c in dados["columns"]
            ),
            mean=tuple(float(v) for v in dados["mean"]),
            scale=tuple(float(v) for v in dados["scale"]),
            coef=tuple(tuple(float(v) for v in linha) for linha in dados["coef"]),
            intercept=tuple(float(v) for v in dados["intercept"]),
        )

    def save(self, destino: Path) -> None:
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(json.dumps(self.to_dict(), indent=1) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, origem: Path) -> EvidenceModel:
        return cls.from_dict(json.loads(origem.read_text(encoding="utf-8")))


def softmax(logits: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    deslocado = np.exp(logits - logits.max(axis=1, keepdims=True))
    return deslocado / deslocado.sum(axis=1, keepdims=True)


def _caminho(valor: object, evidencia: Mapping[str, object], caminho: tuple[str, ...]) -> object:
    if caminho == (VALOR,):
        return valor
    atual: object = evidencia
    for chave in caminho:
        mapa = _mapa(atual)
        if mapa is None:
            return None
        atual = mapa.get(chave)
    return atual


def _mapa(valor: object) -> Mapping[str, object] | None:
    return cast("Mapping[str, object]", valor) if isinstance(valor, Mapping) else None
