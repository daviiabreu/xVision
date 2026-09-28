"""Comportamento publico da geometria de campo."""

from __future__ import annotations

import numpy as np
import pytest

from edge.modules.field.geometry import (
    PITCH_105X68,
    CalibrationConfig,
    FieldProjectionError,
    ImageKeypoint,
    calibrate,
    has_safe_pitch_projection,
    has_safe_visible_projection,
    image_to_pitch,
    pitch_to_image,
)
from models.line_tracking.dataset import KEYPOINT_NAMES


def test_pitch_to_image_round_trips_in_metres() -> None:
    homography = (10.0, 0.0, 960.0, 0.0, -10.0, 540.0, 0.0, 0.0, 1.0)

    image_point = pitch_to_image(homography, (12.0, -8.0))

    assert image_point == pytest.approx((1080.0, 620.0))
    assert image_to_pitch(homography, image_point) == pytest.approx((12.0, -8.0))
    assert PITCH_105X68.template_id == "pitch-105x68-center-v1"


def test_template_names_the_29_weight_outputs_in_order() -> None:
    keypoints = PITCH_105X68.keypoints

    assert len(keypoints) == 29
    assert tuple(point.name for point in keypoints) == tuple(KEYPOINT_NAMES)
    assert len({point.name for point in keypoints}) == 29
    assert len({point.pitch_xy for point in keypoints}) == 29
    assert keypoints[0].name == "sideline_top_left"
    assert keypoints[0].pitch_xy == (-52.5, -34.0)
    assert keypoints[15].name == "field_center"
    assert keypoints[15].pitch_xy == (0.0, 0.0)
    assert keypoints[28].name == "center_circle_right"
    assert keypoints[28].pitch_xy == (9.15, 0.0)


def test_ransac_excludes_an_outlier_and_recovers_pitch_to_image() -> None:
    source = PITCH_105X68.keypoints[:7]
    observations = tuple(
        ImageKeypoint(
            name=point.name,
            image_xy=(point.pitch_xy[0] * 10.0 + 960.0, point.pitch_xy[1] * -10.0 + 540.0),
            confidence=0.9,
        )
        for point in source
    )
    observations = (*observations, ImageKeypoint("field_center", (15.0, 15.0), 0.9))

    result = calibrate(
        observations,
        frame_size=(1920, 1080),
        config=CalibrationConfig(method="ransac", min_inliers=6),
    )

    assert result.reason is None
    assert result.homography is not None
    assert result.inlier_count == 7
    assert pitch_to_image(result.homography, (12.0, -8.0)) == pytest.approx(
        (1080.0, 620.0), abs=0.01
    )


def test_high_reprojection_error_is_unavailable() -> None:
    selected = (0, 2, 9, 11, 12, 16, 18, 25)
    observations = [
        ImageKeypoint(
            name=PITCH_105X68.keypoints[index].name,
            image_xy=(
                PITCH_105X68.keypoints[index].pitch_xy[0] * 10.0 + 960.0,
                PITCH_105X68.keypoints[index].pitch_xy[1] * -10.0 + 540.0,
            ),
            confidence=0.9,
        )
        for index in selected
    ]
    observations[-1] = ImageKeypoint(observations[-1].name, (1500.0, 200.0), 0.9)

    result = calibrate(
        tuple(observations),
        frame_size=(1920, 1080),
        config=CalibrationConfig(
            method="least-squares", min_inliers=6, max_reprojection_p95_px=2.0
        ),
    )

    assert result.homography is None
    assert result.reason == "high-reprojection-error"


def test_duplicate_semantic_keypoints_do_not_count_as_independent() -> None:
    observations = tuple(
        ImageKeypoint(name, (float(index), float(index)), 0.9)
        for index, name in enumerate(
            (
                "sideline_top_left",
                "sideline_top_left",
                "field_center",
                "field_center",
                "sideline_bottom_right",
                "sideline_bottom_right",
            )
        )
    )

    result = calibrate(observations, frame_size=(1920, 1080))

    assert result.observed_keypoint_count == 3
    assert result.reason == "insufficient-keypoints"


def test_non_finite_observation_is_rejected_without_calling_it_geometry() -> None:
    selected = (0, 2, 9, 11, 12, 16)
    observations = tuple(
        ImageKeypoint(
            PITCH_105X68.keypoints[index].name,
            (float("inf"), 10.0)
            if position == 0
            else (
                PITCH_105X68.keypoints[index].pitch_xy[0] * 10.0 + 960.0,
                PITCH_105X68.keypoints[index].pitch_xy[1] * -10.0 + 540.0,
            ),
            0.9,
        )
        for position, index in enumerate(selected)
    )

    result = calibrate(observations, frame_size=(1920, 1080))

    assert result.homography is None
    assert result.reason == "non-finite-projection"


def test_projection_rejects_a_point_with_zero_projective_denominator() -> None:
    homography = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)

    with pytest.raises(FieldProjectionError, match="denominador"):
        image_to_pitch(homography, (1.0, 4.0))


def test_calibration_rejects_horizon_crossing_unobserved_half_of_pitch() -> None:
    homography = (10.0, 0.0, 960.0, 0.0, -10.0, 540.0, 0.1, 0.0, 1.0)
    observations = tuple(
        ImageKeypoint(point.name, pitch_to_image(homography, point.pitch_xy), 0.9)
        for point in PITCH_105X68.keypoints
        if point.pitch_xy[0] >= 0
    )

    result = calibrate(observations, frame_size=(1920, 1080))

    assert len(observations) == 17
    assert result.homography is None
    assert result.reason == "degenerate-homography"


def test_a_checagem_visivel_aceita_canto_distante_atras_da_camera() -> None:
    """Camera virada para um gol: o canto do outro lado fica atras dela sem aparecer no quadro."""
    virada = np.array([[10.0, 0.0, 960.0], [0.0, -10.0, 540.0], [-0.03, 0.0, 1.0]])

    assert not has_safe_pitch_projection(virada)
    assert has_safe_visible_projection(virada, (1920, 1080))


def test_a_checagem_visivel_recusa_campo_espelhado_e_campo_fora_do_quadro() -> None:
    espelhada = -np.array([[10.0, 0.0, 960.0], [0.0, -10.0, 540.0], [0.0, 0.0, -1.0]])
    longe = np.array([[10.0, 0.0, 90000.0], [0.0, -10.0, 540.0], [0.0, 0.0, 1.0]])

    assert not has_safe_visible_projection(espelhada, (1920, 1080))
    assert not has_safe_visible_projection(longe, (1920, 1080))


def test_camera_virada_para_um_gol_so_calibra_com_a_checagem_visivel() -> None:
    virada = (10.0, 0.0, 960.0, 0.0, -10.0, 540.0, -0.025, 0.0, 1.0)
    pontos = tuple(
        ImageKeypoint(p.name, pitch_to_image(virada, p.pitch_xy), 0.9)
        for p in PITCH_105X68.keypoints
        if p.pitch_xy[0] <= 10.0
    )

    padrao = calibrate(pontos, frame_size=(1920, 1080))
    visivel = calibrate(
        pontos, frame_size=(1920, 1080), config=CalibrationConfig(safe_projection="visible")
    )

    assert padrao.reason == "degenerate-homography"
    assert visivel.homography is not None
    assert pitch_to_image(visivel.homography, (-20.0, 5.0)) == pytest.approx(
        pitch_to_image(virada, (-20.0, 5.0)), abs=0.01
    )
