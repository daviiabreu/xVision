from __future__ import annotations

from dataclasses import dataclass

import pytest

from contracts import (
    BallDetection,
    Box,
    FieldGeometry,
    FramePerception,
    Keypoint,
    Perception,
    PlayerDetection,
)
from edge.features.goalkeeper_position import GoalkeeperPosition

SHOT_FRAME = 60
HOMOGRAPHY = (10.0, 0.0, 600.0, 0.0, -10.0, 400.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class _Shot:
    frame_index: int = SHOT_FRAME
    shooter_track_id: int | None = 7


def _field() -> FieldGeometry:
    return FieldGeometry(
        homography=HOMOGRAPHY,
        reprojection_error_px=2.0,
        reprojection_error_p95_px=3.0,
        keypoints=(),
        observed_keypoint_count=6,
        inlier_count=6,
        source="detected",
        age_frames=0,
        template_id="pitch-105x68-center-v1",
        reason=None,
    )


def _player(
    track_id: int,
    pitch_xy: tuple[float, float],
    *,
    role: str = "player",
    confidence: float = 0.9,
    pose: bool = False,
    box: Box | None = None,
) -> PlayerDetection:
    return PlayerDetection(
        box=box or Box(100.0, 100.0, 140.0, 220.0),
        confidence=confidence,
        track_id=track_id,
        role=role,
        pitch_xy=pitch_xy,
        keypoints=(Keypoint("nose", 120.0, 110.0, 0.9),) if pose else (),
    )


def _perception(
    *,
    shooter: tuple[float, float] = (40.0, 0.0),
    goalkeeper: tuple[float, float] | None = (51.0, 0.0),
    shooter_pose: bool = False,
    ball: bool = False,
) -> Perception:
    players = [_player(7, shooter, confidence=0.8, pose=shooter_pose)]
    if goalkeeper is not None:
        players.append(_player(1, goalkeeper, role="goalkeeper", confidence=0.7))
    detected_ball = (
        BallDetection(box=Box(110.0, 210.0, 120.0, 220.0), confidence=0.9, source="detected")
        if ball
        else None
    )
    return Perception(
        clip_id="c",
        frames=(
            FramePerception(
                frame_index=SHOT_FRAME,
                timestamp_s=1.0,
                players=tuple(players),
                ball=detected_ball,
                field=_field(),
            ),
        ),
    )


def test_goleiro_sobre_a_linha_do_chute_recebe_escore_maximo() -> None:
    value = GoalkeeperPosition().extract(_perception(), _Shot())

    assert value.status == "ok"
    assert value.value is True
    assert value.evidence["score"] == 1.0
    assert value.evidence["threshold"] == 0.5
    assert value.confidence == 0.7
    assert value.evidence["goalkeeper_track_id"] == 1


def test_desvio_de_uma_trave_recebe_escore_zero() -> None:
    value = GoalkeeperPosition().extract(_perception(goalkeeper=(51.0, 3.66)), _Shot())

    assert value.status == "ok"
    assert value.value is False
    assert value.evidence["score"] == pytest.approx(0.0, abs=0.01)
    assert value.evidence["desvio_lateral_m"] == pytest.approx(3.66, abs=0.01)


def test_a_distancia_e_perpendicular_tambem_em_chute_diagonal() -> None:
    shooter = (40.0, -10.0)
    goalkeeper = (51.0, -10.0 / 12.5 * 1.5)

    value = GoalkeeperPosition().extract(
        _perception(shooter=shooter, goalkeeper=goalkeeper), _Shot()
    )

    assert value.value is True
    assert value.evidence["score"] == pytest.approx(1.0, abs=0.01)


def test_o_gol_mais_proximo_do_finalizador_e_a_referencia() -> None:
    value = GoalkeeperPosition().extract(
        _perception(shooter=(-40.0, 0.0), goalkeeper=(-51.0, 0.0)), _Shot()
    )

    assert value.status == "ok"
    assert value.evidence["gol"] == "esquerdo"


def test_sem_goleiro_a_feature_declara_indisponibilidade() -> None:
    value = GoalkeeperPosition().extract(_perception(goalkeeper=None), _Shot())

    assert value.status == "unavailable"
    assert "goleiro" in (value.reason or "")


def test_o_quadro_mais_proximo_com_o_finalizador_declarado_e_usado() -> None:
    sem_finalizador = FramePerception(
        frame_index=SHOT_FRAME,
        timestamp_s=1.0,
        players=(_player(1, (51.0, 0.0), role="goalkeeper"),),
        field=_field(),
    )
    com_finalizador = _perception().frames[0]
    deslocado = FramePerception(
        frame_index=SHOT_FRAME - 2,
        timestamp_s=0.967,
        players=com_finalizador.players,
        field=com_finalizador.field,
    )

    value = GoalkeeperPosition().extract(Perception("c", (sem_finalizador, deslocado)), _Shot())

    assert value.status == "ok"
    assert value.evidence["frame_utilizado"] == SHOT_FRAME - 2


def test_sem_homografia_a_feature_nao_mede_em_pixels_como_se_fossem_metros() -> None:
    frame = FramePerception(
        frame_index=SHOT_FRAME,
        timestamp_s=1.0,
        players=(PlayerDetection(Box(100.0, 100.0, 140.0, 220.0), 0.8, track_id=7),),
        field=FieldGeometry(
            homography=None,
            reprojection_error_px=None,
            reprojection_error_p95_px=None,
            keypoints=(),
            observed_keypoint_count=0,
            inlier_count=0,
            source="unavailable",
            age_frames=0,
            template_id="pitch-105x68-center-v1",
            reason="insufficient-keypoints",
        ),
    )

    value = GoalkeeperPosition().extract(Perception("c", (frame,)), _Shot())

    assert value.status == "unavailable"
    assert "pitch_xy" in (value.reason or "")


def test_pose_unica_resolve_finalizador_quando_shot_nao_traz_id() -> None:
    value = GoalkeeperPosition().extract(
        _perception(shooter_pose=True), _Shot(shooter_track_id=None)
    )

    assert value.status == "ok"
    assert value.evidence["shooter_track_id"] == 7


def test_bola_detectada_e_fallback_quando_pose_e_id_faltam() -> None:
    value = GoalkeeperPosition().extract(_perception(ball=True), _Shot(shooter_track_id=None))

    assert value.status == "ok"
    assert value.evidence["shooter_track_id"] == 7
