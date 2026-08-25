"""A plataforma de extracao de feature, e nenhuma feature dentro.

Quais features rodam nao se decide aqui: sai de `[features.<nome>]` em `config/pipeline.toml`,
que `edge.features.registry` resolve. Sete pessoas editando linhas diferentes de um arquivo de
configuracao conflitam menos que sete editando a mesma tupla num modulo Python, e a omissao
continua aparecendo no diff da revisao.
"""

from edge.features.base import (
    FeatureError,
    FeatureExtractor,
    FeatureSpec,
    ShotOutput,
    run_features,
    unavailable,
)

__all__ = [
    "FeatureError",
    "FeatureExtractor",
    "FeatureSpec",
    "ShotOutput",
    "run_features",
    "unavailable",
]
