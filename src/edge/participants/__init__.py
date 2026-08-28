"""Tudo o que o sistema sabe sobre os participantes de uma partida.

Detectar, rastrear, dar identidade temporal, atribuir equipe e dizer quem marca quem. O nome
anterior era `xvision_players_yolo`, e ele mentia: metade do que vive aqui, a identificacao de
equipe e a marcacao, nao tem nada a ver com YOLO nem com deteccao, e trabalha em cima de tracking
que ja existe. O YOLO e um adaptador la dentro, em `detector.py`, e nao o assunto do pacote.
"""

from edge.participants.detector import PlayerDetector
from edge.participants.module import PlayersModule, TrackingModule
from edge.participants.tracking import PlayerTracker
from edge.participants.types import Box, FramePerception, PlayerDetection, Role

__all__ = [
    "Box",
    "FramePerception",
    "PlayerDetection",
    "PlayerDetector",
    "PlayerTracker",
    "PlayersModule",
    "Role",
    "TrackingModule",
]
