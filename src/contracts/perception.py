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
class Keypoint:
    """Um ponto do corpo em pixel do quadro original, com a confianca de quem o estimou.

    A confianca e por ponto e nao por pessoa porque e assim que ela decide: tornozelo ocluso sai
    com confianca baixa enquanto a cabeca do mesmo jogador sai alta, e a feature que mede parte do
    corpo precisa distinguir os dois para sair `unavailable` no ponto certo em vez de medir sobre
    ruido. `None` e para estimador que nao reporta, e obriga quem consome a tratar a ausencia.
    """

    name: str
    x: float
    y: float
    confidence: float | None = None


@dataclass(frozen=True)
class PlayerDetection:
    box: Box
    confidence: float
    track_id: int | None = None
    team: str | None = None
    role: str | None = None
    jersey_number: int | None = None
    pitch_xy: tuple[float, float] | None = None
    keypoints: tuple[Keypoint, ...] = ()


@dataclass(frozen=True)
class BallDetection:
    box: Box
    confidence: float
    pitch_xy: tuple[float, float] | None = None
    source: str | None = None
    """`detected` quando um detector viu a bola neste quadro, `interpolated` quando o rastreador a
    inventou preenchendo lacuna, `None` quando quem montou nao sabe.

    Sem este campo o consumidor nao distingue medida de extrapolacao, e a diferenca e grande: a
    interpolacao e linear e preenche ate 0,33 s, enquanto o instante do chute e exatamente onde a
    trajetoria vira. Uma reta tracada por cima da virada passa longe do pe. Medido: na linha de base
    de 38 clipes anotados, 28 tinham bola no quadro do contato e cerca de metade era posicao
    inventada, entao toda taxa publicada sobre `ball` estava inflada por extrapolacao.
    """


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
