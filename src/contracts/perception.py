"""Tipos congelados da percepcao, produzidos na borda e nunca serializados no evento."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from contracts.events import ContractError

FieldUnavailableReason = Literal[
    "no-field-instance",
    "insufficient-keypoints",
    "collinear-keypoints",
    "insufficient-inliers",
    "mirrored-correspondence",
    "degenerate-homography",
    "high-reprojection-error",
    "non-finite-projection",
    "outside-template",
]
FIELD_UNAVAILABLE_REASONS = frozenset(
    {
        "no-field-instance",
        "insufficient-keypoints",
        "collinear-keypoints",
        "insufficient-inliers",
        "mirrored-correspondence",
        "degenerate-homography",
        "high-reprojection-error",
        "non-finite-projection",
        "outside-template",
    }
)


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

    trajectory_segment: int | None = None
    gap_frames: int | None = None
    velocity_xy_px_s: tuple[float, float] | None = None
    velocity_source: Literal["finite-difference", "trajectory-model"] | None = None

    def __post_init__(self) -> None:
        values = (self.box.x1, self.box.y1, self.box.x2, self.box.y2, self.confidence)
        if not all(math.isfinite(value) for value in values):
            raise ContractError("ball exige caixa e confianca finitas")
        if self.box.x2 <= self.box.x1 or self.box.y2 <= self.box.y1:
            raise ContractError("ball exige caixa com area positiva")
        if self.source not in (None, "detected", "interpolated"):
            raise ContractError(f"ball source {self.source!r} desconhecido")
        if self.trajectory_segment is not None and self.trajectory_segment < 0:
            raise ContractError("ball exige trajectory_segment nao negativo")
        if self.gap_frames is not None and self.gap_frames < 0:
            raise ContractError("ball exige gap_frames nao negativo")
        if self.source == "detected" and self.gap_frames not in (None, 0):
            raise ContractError("ball detected exige gap_frames=0")
        if self.source == "interpolated":
            if self.confidence != 0.0:
                raise ContractError("ball interpolated exige confidence=0.0")
            if self.gap_frames is None or self.gap_frames <= 0:
                raise ContractError("ball interpolated exige gap_frames positivo")
        if self.velocity_xy_px_s is not None and not all(
            math.isfinite(value) for value in self.velocity_xy_px_s
        ):
            raise ContractError("ball exige velocidade finita")
        if (self.velocity_xy_px_s is None) != (self.velocity_source is None):
            raise ContractError("ball exige velocidade e procedencia juntas")
        if self.velocity_source not in (None, "finite-difference", "trajectory-model"):
            raise ContractError(f"ball velocity_source {self.velocity_source!r} desconhecido")


@dataclass(frozen=True)
class FieldGeometry:
    homography: tuple[float, ...] | None
    reprojection_error_px: float | None
    reprojection_error_p95_px: float | None
    keypoints: tuple[tuple[str, float, float], ...]
    observed_keypoint_count: int
    inlier_count: int
    source: Literal["detected", "unavailable"]
    age_frames: Literal[0]
    template_id: str
    reason: FieldUnavailableReason | None
    refinement: Literal[
        "disabled", "unavailable", "insufficient-lines", "unsafe-fit", "keypoint-error", "accepted"
    ] = "disabled"
    line_support_count: int = 0

    def __post_init__(self) -> None:
        if self.refinement not in {
            "disabled",
            "unavailable",
            "insufficient-lines",
            "unsafe-fit",
            "keypoint-error",
            "accepted",
        }:
            raise ContractError("field recebeu refinement desconhecido")
        if self.line_support_count < 0 or (
            self.refinement == "accepted"
            and (self.line_support_count < 4 or self.source != "detected")
        ):
            raise ContractError("field recebeu apoio de retas invalido")
        if not self.template_id:
            raise ContractError("field sem template_id")
        if self.age_frames != 0:
            raise ContractError("field por quadro exige age_frames=0")
        if (
            self.observed_keypoint_count < 0
            or not 0 <= self.inlier_count <= self.observed_keypoint_count
        ):
            raise ContractError("contagens de keypoints do field sao invalidas")
        if self.source == "detected":
            if self.homography is None or len(self.homography) != 9:
                raise ContractError("field detected exige homography com 9 valores")
            if not all(math.isfinite(value) for value in self.homography):
                raise ContractError("field detected exige homography finita")
            if abs(_determinant3(self.homography)) < 1e-12:
                raise ContractError("field detected recebeu homography degenerada")
            if self.reprojection_error_px is None or self.reprojection_error_p95_px is None:
                raise ContractError("field detected exige erros de reprojecao")
            if (
                not math.isfinite(self.reprojection_error_px)
                or not math.isfinite(self.reprojection_error_p95_px)
                or self.reprojection_error_px < 0
                or self.reprojection_error_p95_px < self.reprojection_error_px
            ):
                raise ContractError("field detected recebeu erros de reprojecao invalidos")
            if self.reason is not None:
                raise ContractError("field detected nao aceita reason")
            return
        if self.source != "unavailable":
            raise ContractError(f"field source {self.source!r} desconhecido")
        if self.homography is not None:
            raise ContractError("field unavailable nao aceita homography")
        if self.reprojection_error_px is not None or self.reprojection_error_p95_px is not None:
            raise ContractError("field unavailable nao aceita erro de reprojecao")
        if not self.reason:
            raise ContractError("field unavailable exige reason")
        if self.reason not in FIELD_UNAVAILABLE_REASONS:
            raise ContractError(f"field unavailable recebeu reason {self.reason!r} desconhecido")


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


def _determinant3(matrix: tuple[float, ...]) -> float:
    a, b, c, d, e, f, g, h, i = matrix
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
