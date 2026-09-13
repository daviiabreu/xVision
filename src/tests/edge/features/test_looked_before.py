from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from contracts import Box, FieldGeometry, FramePerception, Keypoint, Perception, PlayerDetection
from edge.features.looked_before import LookedBefore

FPS = 60.0
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


def _pose(*, direita: bool = True, confidence: float = 0.9) -> tuple[Keypoint, ...]:
    direcao = 1.0 if direita else -1.0
    return (
        Keypoint("nose", 100.0 + 20.0 * direcao, 100.0, confidence),
        Keypoint("eye_left", 100.0 + 16.0 * direcao, 96.0, confidence),
        Keypoint("eye_right", 100.0 + 16.0 * direcao, 104.0, confidence),
        Keypoint("shoulder_left", 100.0, 92.0, confidence),
        Keypoint("shoulder_right", 100.0, 108.0, confidence),
    )


def _player(
    *,
    track_id: int = 7,
    pose: tuple[Keypoint, ...] = (),
    pitch_xy: tuple[float, float] | None = (40.0, 0.0),
) -> PlayerDetection:
    return PlayerDetection(
        box=Box(80.0, 80.0, 120.0, 180.0),
        confidence=0.9,
        track_id=track_id,
        role="player",
        pitch_xy=pitch_xy,
        keypoints=pose,
    )


def _perception(
    *, direita: bool = True, confidence: float = 0.9, second_track: bool = False
) -> Perception:
    frames: list[FramePerception] = []
    for index in (30, 36, 42, 48, 54, SHOT_FRAME):
        keypoints = _pose(direita=direita, confidence=confidence) if index < SHOT_FRAME else ()
        players = [_player(pose=keypoints)]
        if second_track and index < SHOT_FRAME:
            players.append(_player(track_id=8, pose=keypoints, pitch_xy=(38.0, 4.0)))
        frames.append(
            FramePerception(
                frame_index=index,
                timestamp_s=index / FPS,
                players=tuple(players),
                field=_field(),
            )
        )
    return Perception(clip_id="c", frames=tuple(frames))


def test_orientado_para_o_gol_recebe_escore_alto() -> None:
    value = LookedBefore().extract(_perception(), _Shot())

    assert value.status == "ok"
    assert value.value is True
    assert cast("float", value.evidence["score"]) > 0.95
    assert value.evidence["threshold"] == 0.5
    assert value.evidence["gol"] == "direito"
    assert value.evidence["quadros_validos"] == 5


def test_orientado_para_o_lado_oposto_recebe_escore_baixo() -> None:
    value = LookedBefore().extract(_perception(direita=False), _Shot())

    assert value.status == "ok"
    assert value.value is False
    assert cast("float", value.evidence["score"]) < 0.05


def test_keypoints_de_baixa_confianca_nao_viram_orientacao() -> None:
    value = LookedBefore().extract(_perception(confidence=0.2), _Shot())

    assert value.status == "unavailable"
    assert "quadros validos" in (value.reason or "")


def test_sem_homografia_a_feature_nao_compara_pontos_em_espacos_diferentes() -> None:
    field = FieldGeometry(
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
    )
    frames = tuple(
        FramePerception(
            frame_index=index,
            timestamp_s=index / FPS,
            players=(_player(pose=_pose(), pitch_xy=None),),
            field=field,
        )
        for index in (30, 36, 42, 48, 54, SHOT_FRAME)
    )

    value = LookedBefore().extract(Perception("c", frames), _Shot())

    assert value.status == "unavailable"
    assert "homografia" in (value.reason or "")


def test_sem_finalizador_unico_a_feature_nao_escolhe_um_por_conveniencia() -> None:
    value = LookedBefore().extract(_perception(second_track=True), _Shot(shooter_track_id=None))

    assert value.status == "unavailable"
    assert "finalizador" in (value.reason or "")


def test_finalizador_declarado_resolve_ambiguidade_entre_corpos() -> None:
    value = LookedBefore().extract(_perception(second_track=True), _Shot(shooter_track_id=7))

    assert value.status == "ok"
    assert value.evidence["track_id"] == 7
