"""Tipos congelados da percepcao, produzidos na borda e nunca serializados no evento."""

from __future__ import annotations

from dataclasses import dataclass

from contracts.events import ContractError


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class Clip:
    clip_id: str
    path: str
    fps: float
    frame_count: int
    width: int
    height: int
    match_id: str | None = None

    def __post_init__(self) -> None:
        if not self.clip_id:
            raise ContractError("clip_id vazio")
        for campo, valor in (
            ("fps", self.fps),
            ("frame_count", self.frame_count),
            ("width", self.width),
            ("height", self.height),
        ):
            if valor <= 0:
                raise ContractError(
                    f"clipe {self.clip_id!r} com {campo}={valor!r}, e a medida precisa ser positiva"
                )


@dataclass(frozen=True)
class PlayerDetection:
    box: Box
    confidence: float
    track_id: int | None = None
    team: str | None = None
    role: str | None = None
    jersey_number: int | None = None
    pitch_xy: tuple[float, float] | None = None


@dataclass(frozen=True)
class BallDetection:
    box: Box
    confidence: float
    pitch_xy: tuple[float, float] | None = None


@dataclass(frozen=True)
class FieldGeometry:
    homography: tuple[float, ...] | None
    reprojection_error_px: float | None
    keypoints: tuple[tuple[str, float, float], ...] = ()


@dataclass(frozen=True)
class FramePerception:
    frame_index: int
    timestamp_s: float
    players: tuple[PlayerDetection, ...] = ()
    ball: BallDetection | None = None
    field: FieldGeometry | None = None

    def __post_init__(self) -> None:
        if self.field is not None and self.field.homography is not None:
            return
        if any(player.pitch_xy is not None for player in self.players) or (
            self.ball is not None and self.ball.pitch_xy is not None
        ):
            raise ContractError(
                f"quadro {self.frame_index} sem homografia nao pode carregar pitch_xy"
            )


@dataclass(frozen=True)
class Perception:
    clip_id: str
    frames: tuple[FramePerception, ...]
