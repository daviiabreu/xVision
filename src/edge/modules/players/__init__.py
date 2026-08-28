"""A especie `players`: uma caixa por jogador, por quadro, e a base das outras tres."""

from edge.modules.players.detector import (
    COCO_PERSON,
    FUTEBOL,
    PREPROCESS,
    PlayerBackend,
    PlayersConfig,
    YoloPlayers,
)

__all__ = [
    "COCO_PERSON",
    "FUTEBOL",
    "PREPROCESS",
    "PlayerBackend",
    "PlayersConfig",
    "YoloPlayers",
]
