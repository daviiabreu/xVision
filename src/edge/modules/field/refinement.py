"""Refinamento local de uma homografia semantica usando retas brancas no gramado."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from edge.modules.field.geometry import FIELD_KEYPOINTS, has_safe_pitch_projection

Array = np.ndarray[Any, Any]
STRAIGHT_SEGMENTS = (
    (0, 16),
    (9, 25),
    (0, 9),
    (16, 25),
    (11, 12),
    (1, 2),
    (2, 4),
    (4, 3),
    (17, 18),
    (18, 20),
    (20, 19),
    (5, 6),
    (6, 8),
    (8, 7),
    (21, 22),
    (22, 24),
    (24, 23),
)


@dataclass(frozen=True)
class LineRefinement:
    homography: tuple[float, ...]
    status: str
    supported_lines: int = 0
    error_before_px: float | None = None
    error_after_px: float | None = None


@dataclass(frozen=True)
class _Line:
    pitch: Array
    equation: Array
    family: int


def refine_white_lines(
    frame: Array,
    homography: tuple[float, ...],
    *,
    accepts: Callable[[tuple[float, ...]], bool] | None = None,
) -> LineRefinement:
    """Retorna a matriz original quando o apoio branco nao restringe um ajuste seguro.

    Imagem BGR uint8 e homografia devem compartilhar o mesmo espaco de pixels.
    O residuo e apoio interno no branco, nao erro de validacao contra anotacao humana.
    """
    original = np.asarray(homography, dtype=np.float64).reshape(3, 3)
    height, width = frame.shape[:2]
    if not np.isfinite(original).all() or abs(float(np.linalg.det(original))) < 1e-12:
        return LineRefinement(homography, "unsafe-fit")
    radius = max(9, min(32, round(width * 0.015)))
    mask = _paint_mask(frame)
    lines = _supported_lines(mask, original, radius)
    fallback = LineRefinement(homography, "insufficient-lines", len(lines))
    if any(sum(line.family == family for line in lines) < 2 for family in (0, 1)):
        return fallback
    candidate = _fit(lines, width, height)
    if candidate is None or not _safe(candidate, original, width, height, radius):
        return LineRefinement(homography, "unsafe-fit", len(lines))
    before = _error(lines, original)
    after = _error(lines, candidate)
    if not np.isfinite(after) or after > max(1.5, width / 960) or before - after < 0.25:
        return LineRefinement(homography, "unsafe-fit", len(lines), before, after)
    for fraction in (1.0, 0.5, 0.25, 0.125):
        trial = original + fraction * (candidate - original)
        values = tuple(float(value) for value in trial.flat)
        residual = _error(lines, trial)
        if before - residual < 0.25 or not _safe(trial, original, width, height, radius):
            continue
        if accepts is None or accepts(values):
            return LineRefinement(values, "accepted", len(lines), before, residual)
    return LineRefinement(homography, "keypoint-error", len(lines), before, after)


def _paint_mask(frame: Array) -> Array:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, np.array([25, 40, 25]), np.array([100, 255, 255]))
    radius = max(5, round(frame.shape[1] / 160))
    nearby = cv2.dilate(grass, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8))
    white = cv2.inRange(hsv, np.array([0, 0, 70]), np.array([179, 95, 255]))
    return cv2.bitwise_and(white, nearby)


def _project(points: Array, matrix: Array) -> Array:
    projected = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return projected[:, :2] / projected[:, 2:]


def _supported_lines(mask: Array, matrix: Array, radius: int) -> list[_Line]:
    lines: list[_Line] = []
    height, width = mask.shape
    for start, end in STRAIGHT_SEGMENTS:
        pitch = np.asarray(
            [FIELD_KEYPOINTS[start].pitch_xy, FIELD_KEYPOINTS[end].pitch_xy], dtype=np.float64
        )
        projected = _project(pitch, matrix)
        if not np.isfinite(projected).all() or np.abs(projected).max() > 1e7:
            continue
        visible, a, b = cv2.clipLine(
            (0, 0, width, height),
            (int(round(projected[0, 0])), int(round(projected[0, 1]))),
            (int(round(projected[1, 0])), int(round(projected[1, 1]))),
        )
        if not visible:
            continue
        samples = _support(mask, np.asarray(a), np.asarray(b), radius)
        if samples is None:
            continue
        fit = cv2.fitLine(samples, cv2.DIST_WELSCH, 0, 0.01, 0.01).ravel()
        normal = np.array([-float(fit[1]), float(fit[0])])
        offset = -float(normal @ fit[2:])
        residual = np.abs(samples @ normal + offset)
        direction = projected[1] - projected[0]
        angle = abs(float(direction @ fit[:2])) / float(np.linalg.norm(direction))
        if float(np.quantile(residual, 0.8)) > max(1.5, width / 960) or angle < 0.985:
            continue
        lines.append(_Line(pitch, np.array([*normal, offset]), int(pitch[0, 0] == pitch[1, 0])))
    return lines


def _support(mask: Array, start: Array, end: Array, radius: int) -> Array | None:
    length = float(np.linalg.norm(end - start))
    if length < 50:
        return None
    tangent = (end - start) / length
    normal = np.array([-tangent[1], tangent[0]])
    anchors = np.linspace(start, end, max(20, min(180, round(length / 5))))
    offsets = np.arange(-radius, radius + 1)
    samples: list[Array] = []
    bins: set[int] = set()
    for index, anchor in enumerate(anchors):
        probe = np.rint(anchor + offsets[:, None] * normal).astype(int)
        inside = (
            (probe[:, 0] >= 0)
            & (probe[:, 0] < mask.shape[1])
            & (probe[:, 1] >= 0)
            & (probe[:, 1] < mask.shape[0])
        )
        values = np.zeros(len(offsets), dtype=bool)
        values[inside] = mask[probe[inside, 1], probe[inside, 0]] > 0
        positive = np.flatnonzero(values)
        runs = np.split(positive, np.flatnonzero(np.diff(positive) > 1) + 1)
        stripes = [run for run in runs if 1 <= len(run) <= max(7, mask.shape[1] / 100)]
        if len(stripes) != 1:
            continue
        samples.append(anchor + float(np.mean(offsets[stripes[0]])) * normal)
        bins.add(min(4, index * 5 // len(anchors)))
    if len(samples) < 12 or len(samples) / len(anchors) < 0.55 or len(bins) < 4:
        return None
    return np.asarray(samples, dtype=np.float32)


def _fit(lines: list[_Line], width: int, height: int) -> Array | None:
    image_scale = np.array([[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]])
    pitch_scale = np.diag([52.5, 34.0, 1.0])
    rows: list[Array] = []
    for line in lines:
        equation: Array = image_scale.T @ line.equation
        equation = np.asarray(equation / float(np.linalg.norm(equation[:2])), dtype=np.float64)
        for x, y in line.pitch / np.array([52.5, 34.0]):
            rows.append(np.kron(equation, np.array([x, y, 1.0])))
    _, singular, vectors = np.linalg.svd(np.asarray(rows), full_matrices=True)
    if singular[7] < singular[0] * 1e-5:
        return None
    candidate = image_scale @ vectors[-1].reshape(3, 3) @ np.linalg.inv(pitch_scale)
    if not np.isfinite(candidate).all() or abs(float(candidate[2, 2])) < 1e-12:
        return None
    return candidate / candidate[2, 2]


def _safe(candidate: Array, original: Array, width: int, height: int, radius: int) -> bool:
    if not has_safe_pitch_projection(candidate) or not has_safe_pitch_projection(original):
        return False
    if float(np.linalg.det(candidate) * np.linalg.det(original)) <= 0:
        return False
    grid = np.array([(x, y) for x in np.linspace(-52.5, 52.5, 9) for y in np.linspace(-34, 34, 7)])
    before, after = _project(grid, original), _project(grid, candidate)
    visible = (
        (before[:, 0] >= 0) & (before[:, 0] < width) & (before[:, 1] >= 0) & (before[:, 1] < height)
    )
    return bool(
        visible.sum() >= 4
        and np.isfinite(after).all()
        and np.linalg.norm(after[visible] - before[visible], axis=1).max() <= radius
    )


def _error(lines: list[_Line], matrix: Array) -> float:
    residuals = np.concatenate(
        [
            np.abs(_project(line.pitch, matrix) @ line.equation[:2] + line.equation[2])
            for line in lines
        ]
    )
    return float(np.quantile(residuals, 0.8))
