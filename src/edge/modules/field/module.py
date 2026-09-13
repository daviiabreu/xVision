"""Especie de campo, com indisponibilidade observavel por quadro."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, replace
from typing import Any, Protocol, cast

import numpy as np

from edge.artifacts import entrypoint_of
from edge.frames.preprocess import to_source_box
from edge.modules.base import (
    FrameBatch,
    ModuleError,
    ModuleInputs,
    ModuleNotLoaded,
    ModuleOutput,
    ModuleSpec,
    PreprocessSpec,
    check_preprocess,
)
from edge.modules.device import preferred_device
from edge.modules.field.geometry import (
    PITCH_105X68,
    CalibrationConfig,
    CalibrationResult,
    ImageKeypoint,
    calibrate,
    pitch_to_image,
)
from edge.modules.field.refinement import LineRefinement, refine_white_lines
from edge.modules.precision import PADRAO, precisao_valida

PREPROCESS = PreprocessSpec(width=960, height=544, color="bgr", letterbox=True, normalize="none")


class FieldKeypointEstimator(Protocol):
    def estimate(self, frame: Any) -> tuple[ImageKeypoint, ...] | None: ...


class FieldConfigError(ModuleError): ...


class YoloField:
    spec = ModuleSpec(
        kind="field",
        name="yolo-field",
        version="0.1.1",
        frames="coarse",
        preprocess=PREPROCESS,
    )

    def __init__(
        self,
        estimator: FieldKeypointEstimator | None = None,
        *,
        calibration: CalibrationConfig | None = None,
    ) -> None:
        self._estimator = estimator
        self._owns_estimator = False
        self._calibration = calibration or CalibrationConfig()
        self._imgsz = PREPROCESS.width
        self._conf = 0.25
        self._device = ""
        self._quantize = PADRAO
        self._engine = ""
        self._engine_onnx_sha256 = ""
        self._ready = False
        self._line_refinement = False

    def fingerprint(self) -> str:
        return json.dumps(
            {
                **asdict(self._calibration),
                "imgsz": self._imgsz,
                "conf": self._conf,
                "device": self._device,
                "quantize": self._quantize,
                "line_refinement": self._line_refinement,
            },
            sort_keys=True,
        )

    def configure(self, options: Mapping[str, object]) -> None:
        allowed = {
            "method",
            "keypoint_confidence",
            "threshold_diagonal_fraction",
            "min_inliers",
            "max_reprojection_p95_px",
            "imgsz",
            "conf",
            "device",
            "quantize",
            "engine",
            "engine_onnx_sha256",
            "line_refinement",
        }
        unknown = set(options) - allowed
        if unknown:
            raise FieldConfigError(f"field nao reconhece {', '.join(sorted(unknown))}")
        changes: dict[str, object] = {}
        if "line_refinement" in options:
            value = options["line_refinement"]
            if not isinstance(value, bool):
                raise FieldConfigError("field.line_refinement espera booleano")
            self._line_refinement = value
        if "method" in options:
            method = options["method"]
            if method not in {"least-squares", "ransac", "magsac"}:
                raise FieldConfigError("field.method espera least-squares, ransac ou magsac")
            changes["method"] = cast("Any", method)
        for name in (
            "keypoint_confidence",
            "threshold_diagonal_fraction",
            "max_reprojection_p95_px",
        ):
            if name not in options:
                continue
            value = options[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
            ):
                raise FieldConfigError(f"field.{name} espera numero finito")
            if name == "keypoint_confidence" and not 0 <= value <= 1:
                raise FieldConfigError("field.keypoint_confidence espera numero entre 0 e 1")
            if name != "keypoint_confidence" and value <= 0:
                raise FieldConfigError(f"field.{name} espera numero positivo")
            changes[name] = float(value)
        if "min_inliers" in options:
            value = options["min_inliers"]
            if isinstance(value, bool) or not isinstance(value, int) or value < 4:
                raise FieldConfigError("field.min_inliers espera inteiro a partir de 4")
            changes["min_inliers"] = value
        if "imgsz" in options:
            value = options["imgsz"]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise FieldConfigError("field.imgsz espera inteiro positivo")
            self._imgsz = value
        if "conf" in options:
            value = options["conf"]
            if isinstance(value, bool) or not isinstance(value, int | float) or not 0 <= value <= 1:
                raise FieldConfigError("field.conf espera numero entre 0 e 1")
            self._conf = float(value)
        if "device" in options:
            value = options["device"]
            if not isinstance(value, str) or not value:
                raise FieldConfigError("field.device espera texto nao vazio")
            self._device = value
        if "quantize" in options:
            self._quantize = precisao_valida(
                "field.quantize", options["quantize"], FieldConfigError
            )
        for nome in ("engine", "engine_onnx_sha256"):
            if nome in options:
                valor = options[nome]
                if not isinstance(valor, str) or not valor:
                    raise FieldConfigError(f"field.{nome} espera texto nao vazio")
                setattr(self, f"_{nome}", valor)
        self._calibration = replace(self._calibration, **changes)  # pyright: ignore[reportArgumentType]

    def load(self) -> None:
        if self._ready:
            return
        if self._estimator is None:
            weights = entrypoint_of(self.spec.artifact)
            if weights is not None:
                from edge.modules.field.yolo import UltralyticsFieldEstimator

                self._estimator = UltralyticsFieldEstimator.from_weights(
                    str(weights),
                    imgsz=self._imgsz,
                    conf=self._conf,
                    device=self._device or preferred_device(),
                    quantize=self._quantize,
                    engine=self._engine,
                    engine_onnx_sha256=self._engine_onnx_sha256,
                )
                self._owns_estimator = True
        self._ready = True

    def infer(self, batch: FrameBatch, ctx: ModuleInputs) -> ModuleOutput:
        if not self._ready:
            raise ModuleNotLoaded(self.spec.name)
        check_preprocess(batch, self.spec.preprocess)
        return ModuleOutput(
            kind=self.spec.kind,
            frames=batch.indices,
            records=tuple(self._frame(batch, position) for position, _ in enumerate(batch.indices)),
        )

    def close(self) -> None:
        if self._owns_estimator:
            close = getattr(self._estimator, "close", None)
            if callable(close):
                close()
            self._estimator = None
            self._owns_estimator = False
        self._ready = False

    def _frame(self, batch: FrameBatch, position: int) -> tuple[dict[str, object], ...]:
        if self._estimator is None:
            record = _unavailable("no-field-instance")
            if self._line_refinement:
                record["refinement"] = "unavailable"
            return (record,)
        measured = self._estimator.estimate(batch.tensor[position])
        if not measured:
            record = _unavailable("no-field-instance")
            if self._line_refinement:
                record["refinement"] = "unavailable"
            return (record,)
        points = _source_points(measured, batch)
        source_size = batch.source_size or (batch.preprocess.width, batch.preprocess.height)
        result = calibrate(points, frame_size=source_size, config=self._calibration)
        if not self._line_refinement:
            return (_record(result, points),)
        if result.homography is None:
            return ({**_record(result, points), "refinement": "unavailable"},)
        result, refinement = self._refine(batch, position, result, points)
        return (
            {
                **_record(result, points),
                "refinement": refinement.status,
                "line_support_count": refinement.supported_lines,
            },
        )

    def _refine(
        self,
        batch: FrameBatch,
        position: int,
        result: CalibrationResult,
        points: tuple[ImageKeypoint, ...],
    ) -> tuple[CalibrationResult, LineRefinement]:
        source_size = batch.source_size or (batch.preprocess.width, batch.preprocess.height)
        x0, y0, x1, y1 = to_source_box((0, 0, 1, 1), batch.preprocess, source_size)
        to_source = np.array([[x1 - x0, 0, x0], [0, y1 - y0, y0], [0, 0, 1]])
        original = np.asarray(result.homography, dtype=np.float64).reshape(3, 3)
        prepared = np.linalg.inv(to_source) @ original
        canonical = {point.name: point.pitch_xy for point in PITCH_105X68.keypoints}
        by_name = {point.name: point for point in sorted(points, key=lambda p: p.confidence)}

        def source_homography(values: tuple[float, ...]) -> tuple[float, ...]:
            matrix = to_source @ np.asarray(values).reshape(3, 3)
            return tuple(float(v) for v in (matrix / matrix[2, 2]).flat)

        def errors_for(values: tuple[float, ...]) -> np.ndarray[Any, Any]:
            return np.asarray(
                [
                    np.linalg.norm(
                        np.asarray(pitch_to_image(values, canonical[name])) - by_name[name].image_xy
                    )
                    for name in result.inlier_keypoint_names
                ]
            )

        def accepts(values: tuple[float, ...]) -> bool:
            return (
                float(np.quantile(errors_for(source_homography(values)), 0.95))
                <= self._calibration.max_reprojection_p95_px
            )

        refinement = refine_white_lines(
            batch.tensor[position],
            tuple(float(v) for v in prepared.flat),
            accepts=accepts,
        )
        if refinement.status != "accepted":
            return result, refinement
        homography = source_homography(refinement.homography)
        errors = errors_for(homography)
        p95 = float(np.quantile(errors, 0.95))
        return replace(
            result,
            homography=homography,
            reprojection_error_px=float(np.median(errors)),
            reprojection_error_p95_px=p95,
        ), refinement


def _unavailable(reason: str) -> dict[str, object]:
    return {
        "homography": None,
        "reprojection_error_px": None,
        "reprojection_error_p95_px": None,
        "keypoints": (),
        "observed_keypoint_count": 0,
        "inlier_count": 0,
        "source": "unavailable",
        "age_frames": 0,
        "template_id": PITCH_105X68.template_id,
        "reason": reason,
    }


def _record(result: CalibrationResult, points: tuple[ImageKeypoint, ...]) -> dict[str, object]:
    return {
        "homography": result.homography,
        "reprojection_error_px": result.reprojection_error_px,
        "reprojection_error_p95_px": result.reprojection_error_p95_px,
        "keypoints": tuple((point.name, *point.image_xy) for point in points),
        "observed_keypoint_count": result.observed_keypoint_count,
        "inlier_count": result.inlier_count,
        "source": "detected" if result.homography is not None else "unavailable",
        "age_frames": 0,
        "template_id": PITCH_105X68.template_id,
        "reason": result.reason,
    }


def _source_points(
    points: tuple[ImageKeypoint, ...], batch: FrameBatch
) -> tuple[ImageKeypoint, ...]:
    if batch.source_size is None:
        return points
    converted: list[ImageKeypoint] = []
    for point in points:
        x, y = point.image_xy
        source = to_source_box((x, y, x, y), batch.preprocess, batch.source_size)
        converted.append(ImageKeypoint(point.name, (source[0], source[1]), point.confidence))
    return tuple(converted)
