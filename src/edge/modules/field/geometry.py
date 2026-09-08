"""Geometria canonica do campo, em metros, separada da inferencia de keypoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import cv2
import numpy as np

from contracts import FieldUnavailableReason


class FieldProjectionError(ValueError): ...


@dataclass(frozen=True)
class FieldKeypoint:
    name: str
    pitch_xy: tuple[float, float]


@dataclass(frozen=True)
class FieldTemplate:
    template_id: str
    length_m: float
    width_m: float
    keypoints: tuple[FieldKeypoint, ...]


CalibrationMethod = Literal["least-squares", "ransac", "magsac"]


@dataclass(frozen=True)
class ImageKeypoint:
    name: str
    image_xy: tuple[float, float]
    confidence: float


@dataclass(frozen=True)
class CalibrationConfig:
    method: CalibrationMethod = "ransac"
    keypoint_confidence: float = 0.25
    threshold_diagonal_fraction: float = 0.0025
    min_inliers: int = 6
    max_reprojection_p95_px: float = 10.0


@dataclass(frozen=True)
class CalibrationResult:
    homography: tuple[float, ...] | None
    reprojection_error_px: float | None
    reprojection_error_p95_px: float | None
    observed_keypoint_count: int
    inlier_count: int
    reason: FieldUnavailableReason | None
    inlier_keypoint_names: tuple[str, ...] = ()


FIELD_KEYPOINTS = (
    FieldKeypoint("sideline_top_left", (-52.5, -34.0)),
    FieldKeypoint("big_rect_left_top_pt1", (-52.5, -20.16)),
    FieldKeypoint("big_rect_left_top_pt2", (-36.0, -20.16)),
    FieldKeypoint("big_rect_left_bottom_pt1", (-52.5, 20.16)),
    FieldKeypoint("big_rect_left_bottom_pt2", (-36.0, 20.16)),
    FieldKeypoint("small_rect_left_top_pt1", (-52.5, -9.16)),
    FieldKeypoint("small_rect_left_top_pt2", (-47.0, -9.16)),
    FieldKeypoint("small_rect_left_bottom_pt1", (-52.5, 9.16)),
    FieldKeypoint("small_rect_left_bottom_pt2", (-47.0, 9.16)),
    FieldKeypoint("sideline_bottom_left", (-52.5, 34.0)),
    FieldKeypoint("left_semicircle_right", (-32.35, 0.0)),
    FieldKeypoint("center_line_top", (0.0, -34.0)),
    FieldKeypoint("center_line_bottom", (0.0, 34.0)),
    FieldKeypoint("center_circle_top", (0.0, -9.15)),
    FieldKeypoint("center_circle_bottom", (0.0, 9.15)),
    FieldKeypoint("field_center", (0.0, 0.0)),
    FieldKeypoint("sideline_top_right", (52.5, -34.0)),
    FieldKeypoint("big_rect_right_top_pt1", (52.5, -20.16)),
    FieldKeypoint("big_rect_right_top_pt2", (36.0, -20.16)),
    FieldKeypoint("big_rect_right_bottom_pt1", (52.5, 20.16)),
    FieldKeypoint("big_rect_right_bottom_pt2", (36.0, 20.16)),
    FieldKeypoint("small_rect_right_top_pt1", (52.5, -9.16)),
    FieldKeypoint("small_rect_right_top_pt2", (47.0, -9.16)),
    FieldKeypoint("small_rect_right_bottom_pt1", (52.5, 9.16)),
    FieldKeypoint("small_rect_right_bottom_pt2", (47.0, 9.16)),
    FieldKeypoint("sideline_bottom_right", (52.5, 34.0)),
    FieldKeypoint("right_semicircle_left", (32.35, 0.0)),
    FieldKeypoint("center_circle_left", (-9.15, 0.0)),
    FieldKeypoint("center_circle_right", (9.15, 0.0)),
)


PITCH_105X68 = FieldTemplate(
    template_id="pitch-105x68-center-v1",
    length_m=105.0,
    width_m=68.0,
    keypoints=FIELD_KEYPOINTS,
)


def pitch_to_image(
    homography: tuple[float, ...], point: tuple[float, float]
) -> tuple[float, float]:
    return _transform(homography, point)


def image_to_pitch(
    homography: tuple[float, ...], point: tuple[float, float]
) -> tuple[float, float]:
    try:
        inverse = np.linalg.inv(_matrix(homography))
    except (ValueError, np.linalg.LinAlgError) as error:
        raise FieldProjectionError("homography sem inversa finita") from error
    return _transform(tuple(float(value) for value in inverse.flat), point)


def calibrate(
    observations: tuple[ImageKeypoint, ...],
    *,
    frame_size: tuple[int, int],
    config: CalibrationConfig | None = None,
    template: FieldTemplate = PITCH_105X68,
) -> CalibrationResult:
    settings = config or CalibrationConfig()
    canonical = {point.name: point.pitch_xy for point in template.keypoints}
    eligible = tuple(
        point
        for point in observations
        if point.confidence >= settings.keypoint_confidence and point.name in canonical
    )
    by_name: dict[str, ImageKeypoint] = {}
    for point in eligible:
        current = by_name.get(point.name)
        if current is None or point.confidence > current.confidence:
            by_name[point.name] = point
    selected = tuple(by_name.values())
    if len(selected) < 4:
        return _unavailable(len(selected), "insufficient-keypoints")
    pitch = np.asarray([canonical[point.name] for point in selected], dtype=np.float64)
    image = np.asarray([point.image_xy for point in selected], dtype=np.float64)
    if not np.isfinite(pitch).all() or not np.isfinite(image).all():
        return _unavailable(len(selected), "non-finite-projection")
    if _collinear(pitch) or _collinear(image):
        return _unavailable(len(selected), "collinear-keypoints")
    method = {
        "least-squares": 0,
        "ransac": cv2.RANSAC,
        "magsac": cv2.USAC_MAGSAC,
    }[settings.method]
    width, height = frame_size
    threshold_px = float(np.hypot(width, height)) * settings.threshold_diagonal_fraction
    matrix, mask = cast(
        "tuple[np.ndarray[Any, Any] | None, np.ndarray[Any, Any] | None]",
        cv2.findHomography(pitch, image, method, threshold_px),
    )
    if matrix is None or not np.isfinite(matrix).all() or abs(float(matrix[2, 2])) < 1e-12:
        return _unavailable(len(selected), "degenerate-homography")
    matrix = matrix / matrix[2, 2]
    accepted = (
        np.ones(len(selected), dtype=bool)
        if settings.method == "least-squares" or mask is None
        else mask.ravel().astype(bool)
    )
    inliers = int(accepted.sum())
    if inliers < settings.min_inliers:
        return _unavailable(len(selected), "insufficient-inliers", inliers)
    projected = cv2.perspectiveTransform(pitch.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    errors = np.linalg.norm(projected[accepted] - image[accepted], axis=1)
    if not np.isfinite(projected).all() or not np.isfinite(errors).all():
        return _unavailable(len(selected), "non-finite-projection", inliers)
    error_p95 = float(cast("Any", np.percentile(errors, 95)))
    if error_p95 > settings.max_reprojection_p95_px:
        return _unavailable(len(selected), "high-reprojection-error", inliers)
    if not has_safe_pitch_projection(matrix, template):
        return _unavailable(len(selected), "degenerate-homography", inliers)
    return CalibrationResult(
        homography=tuple(float(value) for value in matrix.flat),
        reprojection_error_px=float(cast("Any", np.median(errors))),
        reprojection_error_p95_px=error_p95,
        observed_keypoint_count=len(selected),
        inlier_count=inliers,
        reason=None,
        inlier_keypoint_names=tuple(
            point.name for point, keep in zip(selected, accepted, strict=True) if keep
        ),
    )


def has_safe_pitch_projection(
    matrix: np.ndarray[Any, Any], template: FieldTemplate = PITCH_105X68
) -> bool:
    if not np.isfinite(matrix).all() or abs(float(matrix[2, 2])) < 1e-12:
        return False
    normalized = matrix / matrix[2, 2]
    if abs(float(np.linalg.det(normalized))) < 1e-12:
        return False
    try:
        inverse = cast("np.ndarray[Any, Any]", np.linalg.inv(normalized))
    except np.linalg.LinAlgError:
        return False
    corners = np.array(
        [
            (x, y, 1.0)
            for x in (-template.length_m / 2, template.length_m / 2)
            for y in (-template.width_m / 2, template.width_m / 2)
        ]
    )
    denominators = corners @ normalized[2]
    return bool(np.isfinite(inverse).all() and (denominators > 1e-6).all())


def _transform(homography: tuple[float, ...], point: tuple[float, float]) -> tuple[float, float]:
    matrix = _matrix(homography)
    source = np.asarray((point[0], point[1], 1.0), dtype=np.float64)
    projected = matrix @ source
    denominator = float(projected[2])
    if not np.isfinite(projected).all() or abs(denominator) < 1e-12:
        raise FieldProjectionError("projecao com denominador nulo ou nao finito")
    return (float(projected[0] / denominator), float(projected[1] / denominator))


def _matrix(homography: tuple[float, ...]) -> np.ndarray[Any, Any]:
    return np.asarray(homography, dtype=np.float64).reshape(3, 3)


def _collinear(points: np.ndarray[Any, Any]) -> bool:
    centered = points - points.mean(axis=0)
    return bool(np.linalg.matrix_rank(centered) < 2)


def _unavailable(
    observed: int, reason: FieldUnavailableReason, inliers: int = 0
) -> CalibrationResult:
    return CalibrationResult(
        homography=None,
        reprojection_error_px=None,
        reprojection_error_p95_px=None,
        observed_keypoint_count=observed,
        inlier_count=inliers,
        reason=reason,
    )
