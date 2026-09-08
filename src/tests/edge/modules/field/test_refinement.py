"""Apoio sintetico independente, sem frames ou anotacoes do parceiro."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pytest

from edge.modules.field.geometry import pitch_to_image
from edge.modules.field.refinement import refine_white_lines

SEGMENTS = (
    ((-52.5, -34), (52.5, -34)),
    ((-52.5, 34), (52.5, 34)),
    ((-52.5, -34), (-52.5, 34)),
    ((52.5, -34), (52.5, 34)),
    ((0, -34), (0, 34)),
    ((-52.5, -20.16), (-36, -20.16)),
    ((-36, -20.16), (-36, 20.16)),
    ((-36, 20.16), (-52.5, 20.16)),
)


def _pitch(homography: tuple[float, ...]) -> np.ndarray[Any, Any]:
    frame = np.full((544, 960, 3), (30, 120, 30), dtype=np.uint8)
    for a, b in SEGMENTS:
        start, end = (tuple(round(v) for v in pitch_to_image(homography, p)) for p in (a, b))
        cv2.line(frame, start, end, (245, 245, 245), 3)
    return frame


@pytest.mark.parametrize("projective", [0.0, 0.0008])
def test_single_global_fit_recovers_translation_and_perspective(projective: float) -> None:
    truth = (5.0, 0.2, 483.0, 0.1, -5.0, 274.0, projective, 0.0, 1.0)
    initial = (5.0, 0.2, 480.0, 0.1, -5.0, 272.0, projective, 0.0, 1.0)

    result = refine_white_lines(_pitch(truth), initial)

    assert result.status == "accepted"
    assert result.error_after_px < result.error_before_px  # type: ignore[operator]
    for point in ((12.0, -8.0), (-40.0, 25.0), (30.0, 15.0)):
        assert pitch_to_image(result.homography, point) == pytest.approx(
            pitch_to_image(truth, point), abs=1.2
        )


@pytest.mark.parametrize("background", [(0, 0, 0), (245, 245, 245)])
def test_white_without_nearby_grass_preserves_initial_matrix(
    background: tuple[int, int, int],
) -> None:
    initial = (5.0, 0.0, 480.0, 0.0, -5.0, 272.0, 0.0, 0.0, 1.0)
    frame = np.full((544, 960, 3), background, dtype=np.uint8)
    cv2.rectangle(frame, (200, 100), (700, 440), (245, 245, 245), 3)

    result = refine_white_lines(frame, initial)

    assert result.homography == initial
    assert result.status == "insufficient-lines"


def test_parallel_lines_and_small_white_object_cannot_determine_a_pitch() -> None:
    initial = (5.0, 0.0, 480.0, 0.0, -5.0, 272.0, 0.0, 0.0, 1.0)
    frame = np.full((544, 960, 3), (30, 120, 30), dtype=np.uint8)
    for y in (105, 445):
        cv2.line(frame, (210, y), (750, y), (245, 245, 245), 3)
    cv2.rectangle(frame, (480, 200), (486, 220), (245, 245, 245), -1)

    result = refine_white_lines(frame, initial)

    assert result.homography == initial
    assert result.status == "insufficient-lines"


def test_partial_occlusion_does_not_bend_the_canonical_pitch() -> None:
    truth = (5.0, 0.0, 483.0, 0.0, -5.0, 272.0, 0.0, 0.0, 1.0)
    initial = (5.0, 0.0, 480.0, 0.0, -5.0, 272.0, 0.0, 0.0, 1.0)
    frame = _pitch(truth)
    cv2.rectangle(frame, (440, 140), (510, 190), (0, 0, 100), -1)

    result = refine_white_lines(frame, initial)

    assert result.status == "accepted"
    assert pitch_to_image(result.homography, (0, 0)) == pytest.approx((483, 272), abs=1)


@pytest.mark.parametrize("matrix", [(0.0,) * 9, (float("nan"),) * 9])
def test_degenerate_initial_matrix_is_rejected(matrix: tuple[float, ...]) -> None:
    frame = np.zeros((544, 960, 3), dtype=np.uint8)

    result = refine_white_lines(frame, matrix)

    assert result.status == "unsafe-fit"


def test_no_previous_frame_geometry_is_reused() -> None:
    truth = (5.0, 0.0, 483.0, 0.0, -5.0, 272.0, 0.0, 0.0, 1.0)
    initial = (5.0, 0.0, 480.0, 0.0, -5.0, 272.0, 0.0, 0.0, 1.0)
    assert refine_white_lines(_pitch(truth), initial).status == "accepted"

    result = refine_white_lines(np.zeros((544, 960, 3), dtype=np.uint8), initial)

    assert result.homography == initial
    assert result.status == "insufficient-lines"


def test_backtracking_improves_white_fit_without_relaxing_anchor_limit() -> None:
    truth = (5.0, 0.0, 486.0, 0.0, -5.0, 272.0, 0.0, 0.0, 1.0)
    initial = (5.0, 0.0, 480.0, 0.0, -5.0, 272.0, 0.0, 0.0, 1.0)

    result = refine_white_lines(_pitch(truth), initial, accepts=lambda h: h[2] <= 484)

    assert result.status == "accepted"
    assert 480 < result.homography[2] <= 484
    assert result.error_after_px < result.error_before_px  # type: ignore[operator]


def test_rejected_anchor_constraint_keeps_original_homography() -> None:
    truth = (5.0, 0.0, 486.0, 0.0, -5.0, 272.0, 0.0, 0.0, 1.0)
    initial = (5.0, 0.0, 480.0, 0.0, -5.0, 272.0, 0.0, 0.0, 1.0)

    result = refine_white_lines(_pitch(truth), initial, accepts=lambda h: False)

    assert result.status == "keypoint-error"
    assert result.homography == initial
