"""Os sete stubs, que sao o que faz existir fluxo de ponta a ponta sem nenhum modelo pronto.

O nome de cada um comeca em `stub-`, e e por ele que `pipeline.versions` separa o que foi
medido do que foi simulado. Selecionar um e apontar `impl = "edge.modules.stubs:StubPlayers"`
numa entrada `[modules.<especie>]`, que e o que `edge.modules.registry` resolve em
`config/pipeline.toml`.
"""

from edge.modules.stubs.perception import (
    STUBS,
    StubBall,
    StubField,
    StubIdentity,
    StubPlayers,
    StubPose,
    StubShot,
    StubTracking,
)

__all__ = [
    "STUBS",
    "StubBall",
    "StubField",
    "StubIdentity",
    "StubPlayers",
    "StubPose",
    "StubShot",
    "StubTracking",
]
