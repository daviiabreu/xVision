"""Deteccao e tracking de participantes de partidas de futebol."""

from xvision_players_yolo.detector import PlayerDetector
from xvision_players_yolo.module import PlayersModule, TrackingModule
from xvision_players_yolo.tracking import PlayerTracker
from xvision_players_yolo.types import Box, FramePerception, PlayerDetection, Role

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
