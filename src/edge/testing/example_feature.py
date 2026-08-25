"""Duas features de exemplo, que existem para a selecao por TOML ter o que carregar nos testes."""

from __future__ import annotations

from contracts import FeatureValue, Perception
from edge.features.base import FeatureSpec, ShotOutput


class Constante:
    spec = FeatureSpec(name="constante", version="0.1.0", requires=())

    def extract(self, perception: Perception, shot: ShotOutput) -> FeatureValue:
        return FeatureValue(
            name=self.spec.name,
            version=self.spec.version,
            status="ok",
            value=1.0,
            confidence=None,
            reason=None,
            evidence={},
        )


class Outra(Constante):
    spec = FeatureSpec(name="outra", version="0.1.0", requires=())
