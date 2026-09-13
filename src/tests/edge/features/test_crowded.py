"""A feature `crowded`: conta jogadores num raio real da bola, e degrada sem inventar numero."""

from __future__ import annotations

from dataclasses import dataclass

from contracts import (
    BallDetection,
    Box,
    FieldGeometry,
    FramePerception,
    Perception,
    PlayerDetection,
)
from edge.features.crowded import Crowded


@dataclass(frozen=True)
class Shot:
    frame_index: int
    shooter_track_id: int | None = None


def _box() -> Box:
    return Box(x1=0.0, y1=0.0, x2=1.0, y2=1.0)


def _field() -> FieldGeometry:
    return FieldGeometry(
        homography=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        reprojection_error_px=0.5,
        reprojection_error_p95_px=0.6,
        keypoints=(),
        observed_keypoint_count=6,
        inlier_count=6,
        source="detected",
        age_frames=0,
        template_id="pitch-105x68-center-v1",
        reason=None,
    )


def _jogador(pitch_xy: tuple[float, float] | None) -> PlayerDetection:
    return PlayerDetection(box=_box(), confidence=0.9, pitch_xy=pitch_xy)


def test_conta_so_jogadores_dentro_do_raio() -> None:
    perto = _jogador((1.0, 0.0))  # 1m da bola
    longe = _jogador((10.0, 0.0))  # 10m da bola
    frame = FramePerception(
        frame_index=5,
        timestamp_s=0.2,
        players=(perto, longe),
        ball=BallDetection(box=_box(), confidence=0.9, pitch_xy=(0.0, 0.0)),
        field=_field(),
    )
    perception = Perception(clip_id="clipe-1", frames=(frame,))

    valor = Crowded(radius_m=3.0).extract(perception, Shot(frame_index=5))

    assert valor.status == "ok"
    assert valor.value == 1
    assert dict(valor.evidence) == {"radius_m": 3.0, "player_count": 2}


def test_unavailable_quando_bola_sem_pitch_xy() -> None:
    frame = FramePerception(
        frame_index=5,
        timestamp_s=0.2,
        players=(_jogador((1.0, 0.0)),),
        ball=BallDetection(box=_box(), confidence=0.9, pitch_xy=None),
        field=_field(),
    )
    perception = Perception(clip_id="clipe-1", frames=(frame,))

    valor = Crowded().extract(perception, Shot(frame_index=5))

    assert valor.status == "unavailable"
    assert valor.value is None
    assert "pitch_xy" in (valor.reason or "")


def test_unavailable_quando_bola_nao_detectada() -> None:
    frame = FramePerception(
        frame_index=5,
        timestamp_s=0.2,
        players=(_jogador((1.0, 0.0)),),
        ball=None,
        field=_field(),
    )
    perception = Perception(clip_id="clipe-1", frames=(frame,))

    valor = Crowded().extract(perception, Shot(frame_index=5))

    assert valor.status == "unavailable"


def test_unavailable_quando_nenhum_jogador_tem_pitch_xy() -> None:
    frame = FramePerception(
        frame_index=5,
        timestamp_s=0.2,
        players=(_jogador(None),),
        ball=BallDetection(box=_box(), confidence=0.9, pitch_xy=(0.0, 0.0)),
        field=_field(),
    )
    perception = Perception(clip_id="clipe-1", frames=(frame,))

    valor = Crowded().extract(perception, Shot(frame_index=5))

    assert valor.status == "unavailable"


def test_unavailable_quando_frame_do_chute_nao_esta_na_percepcao() -> None:
    frame = FramePerception(frame_index=5, timestamp_s=0.2)
    perception = Perception(clip_id="clipe-1", frames=(frame,))

    valor = Crowded().extract(perception, Shot(frame_index=99))

    assert valor.status == "unavailable"
    assert "99" in (valor.reason or "")


def test_spec_declara_players_e_ball_como_dependencias() -> None:
    assert Crowded.spec.requires == ("players", "ball")
    assert Crowded.spec.name == "crowded"
