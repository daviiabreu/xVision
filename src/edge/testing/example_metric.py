"""Uma feature com unidade declarada, para a fronteira HTTP ter o que traduzir.

`config/pipeline.toml` nao declara feature nenhuma, entao em toda execucao real `metricas` sai
vazia. Os testes da API iteravam sobre essa lista vazia e nao afirmavam nada: dava para cravar o
status da etapa em verdadeiro e apagar a unidade de toda metrica sem derrubar um teste.

Ela mora aqui e nao em `config/pipeline.toml` porque a producao continua sem feature ate os
modelos entrarem, e declarar uma de mentira no arquivo de verdade faria o `eval` medir contra
nada e reportar numero.
"""

from __future__ import annotations

from contracts import FeatureValue, Perception
from edge.features.base import FeatureSpec, ShotOutput

UNIDADE = "m"
VALOR = 11.5
CATEGORIA = "PERNA_ESQUERDA"


class DistanciaFalsa:
    """Devolve sempre o mesmo numero, porque o que se mede aqui e a traducao e nao a feature."""

    spec = FeatureSpec(name="distancia_falsa", version="0.1.0", requires=(), unit=UNIDADE)

    def extract(self, perception: Perception, shot: ShotOutput) -> FeatureValue:
        return FeatureValue(
            name=self.spec.name,
            version=self.spec.version,
            status="ok",
            value=VALOR,
            confidence=None,
            reason=None,
            evidence={},
        )


class SemLeitura(DistanciaFalsa):
    """A feature que nao conseguiu medir, que e o caso que o frontend nao pode ler como medida."""

    spec = FeatureSpec(name="sem_leitura", version="0.1.0", requires=(), unit=UNIDADE)

    def extract(self, perception: Perception, shot: ShotOutput) -> FeatureValue:
        return FeatureValue(
            name=self.spec.name,
            version=self.spec.version,
            status="unavailable",
            value=None,
            confidence=None,
            reason="o modulo de campo nao produziu homografia neste quadro",
            evidence={},
        )


class ParteDoCorpo(DistanciaFalsa):
    """A feature de valor textual, que nenhuma implementacao produz ainda e o contrato ja aceita.

    Tres dos onze blocos `[[label]]` de `config/labels.toml` declaram `type = "enum"`,
    `body_part_2`, `body_part_3` e `type_of_play`, entao categoria e o proximo tipo de valor a
    aparecer e nao uma hipotese. Sem unidade de proposito: categoria nao tem unidade.
    """

    spec = FeatureSpec(name="parte_do_corpo", version="0.1.0", requires=())

    def extract(self, perception: Perception, shot: ShotOutput) -> FeatureValue:
        return FeatureValue(
            name=self.spec.name,
            version=self.spec.version,
            status="ok",
            value=CATEGORIA,
            confidence=None,
            reason=None,
            evidence={},
        )
