"""Adapter do Ultralytics para os 29 keypoints semanticos do campo."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import numpy as np

from edge.artifacts import sha256_of
from edge.modules.base import ModuleError
from edge.modules.field.geometry import PITCH_105X68, ImageKeypoint
from edge.runtime import (
    Array,
    PyTorchBackend,
    RuntimeArtifact,
    RuntimeInfo,
    RuntimeModel,
    TensorSpec,
)

INPUT = TensorSpec("images", (1, 3, 544, 960), "float32", "NCHW")
OUTPUT = TensorSpec("predictions", (1, 92, 10710), "float32", "flat")


class FieldModelIncompatible(ModuleError): ...


class UltralyticsFieldEstimator:
    def __init__(
        self,
        model: Any | None,
        *,
        imgsz: int,
        conf: float,
        device: str,
        backend: PyTorchBackend | None = None,
    ) -> None:
        self._model = model
        self._imgsz = imgsz
        self._conf = conf
        self._device = device
        self._backend = backend

    @classmethod
    def from_weights(
        cls, weights: str, *, imgsz: int, conf: float, device: str
    ) -> UltralyticsFieldEstimator:
        if imgsz != 960:
            raise FieldModelIncompatible(
                f"backend de campo foi congelado em imgsz=960, recebeu {imgsz}"
            )
        path = Path(weights)
        backend = PyTorchBackend(
            provider="ultralytics",
            device=device,
            loader=_load_ultralytics,
            runtime_versions=(("torch", version("torch")), ("ultralytics", version("ultralytics"))),
        )
        backend.load(
            RuntimeArtifact(
                path=path,
                sha256=sha256_of(path),
                format="pt",
                parent_onnx_sha256=None,
                inputs=(INPUT,),
                outputs=(OUTPUT,),
            )
        )
        return cls(None, imgsz=imgsz, conf=conf, device=device, backend=backend)

    @property
    def runtime_info(self) -> RuntimeInfo | None:
        return self._backend.info if self._backend is not None else None

    def close(self) -> None:
        if self._backend is not None:
            self._backend.close()

    def estimate(self, frame: Any) -> tuple[ImageKeypoint, ...] | None:
        if self._backend is not None:
            return self._estimate_with_backend(frame)
        if self._model is None:
            return None
        outputs = self._model.predict(
            source=frame,
            imgsz=self._imgsz,
            conf=self._conf,
            device=self._device,
            verbose=False,
        )
        if not outputs:
            return None
        output = outputs[0]
        keypoints = getattr(output, "keypoints", None)
        xy = getattr(keypoints, "xy", None)
        confidence = getattr(keypoints, "conf", None)
        if xy is None or confidence is None or len(xy) == 0:
            return None
        index = _largest(getattr(output, "boxes", None))
        coordinates = xy[index].tolist()
        scores = confidence[index].tolist()
        if len(coordinates) != len(PITCH_105X68.keypoints) or len(scores) != len(coordinates):
            return None
        return tuple(
            ImageKeypoint(point.name, (float(pixel[0]), float(pixel[1])), float(score))
            for point, pixel, score in zip(PITCH_105X68.keypoints, coordinates, scores, strict=True)
        )

    def _estimate_with_backend(self, frame: Any) -> tuple[ImageKeypoint, ...] | None:
        image = np.asarray(frame)
        if image.shape != (544, 960, 3) or image.dtype != np.uint8:
            raise FieldModelIncompatible(
                f"field recebeu tensor {image.shape} {image.dtype}, esperado (544, 960, 3) uint8"
            )
        rgb = image[:, :, ::-1].transpose(2, 0, 1)[None]
        tensor = np.ascontiguousarray(rgb, dtype=np.float32) / np.float32(255.0)
        backend = self._backend
        if backend is None:
            raise FieldModelIncompatible("field sem backend carregado")
        raw = backend.infer({"images": tensor})[0].value
        torch = cast("Any", import_module("torch"))
        nms = cast("Any", import_module("ultralytics.utils.nms"))
        detections = nms.non_max_suppression(
            torch.from_numpy(np.array(raw, copy=True)),
            conf_thres=self._conf,
            nc=1,
            max_det=10,
        )[0]
        if len(detections) == 0:
            return None
        rows = detections.detach().cpu().numpy()
        index = max(
            range(len(rows)),
            key=lambda i: (rows[i, 2] - rows[i, 0]) * (rows[i, 3] - rows[i, 1]),
        )
        keypoints = rows[index, 6:].reshape(len(PITCH_105X68.keypoints), 3)
        return tuple(
            ImageKeypoint(
                point.name, (float(raw_point[0]), float(raw_point[1])), float(raw_point[2])
            )
            for point, raw_point in zip(PITCH_105X68.keypoints, keypoints, strict=True)
        )


def _largest(boxes: object) -> int:
    xyxy = getattr(boxes, "xyxy", None)
    if xyxy is None or len(xyxy) <= 1:
        return 0
    rows = xyxy.tolist()
    return max(
        range(len(rows)),
        key=lambda index: (rows[index][2] - rows[index][0]) * (rows[index][3] - rows[index][1]),
    )


def check_field_model(model: object) -> None:
    task = getattr(model, "task", None)
    inner = getattr(model, "model", None)
    metadata = cast("object", getattr(inner, "yaml", {}))
    metadata = cast("dict[str, object]", metadata) if isinstance(metadata, dict) else {}
    keypoint_shape = metadata.get("kpt_shape")
    if task != "pose" or keypoint_shape not in ([29, 3], (29, 3)):
        raise FieldModelIncompatible(
            f"field exige task=pose e kpt_shape=[29, 3], recebeu task={task!r} "
            f"e kpt_shape={keypoint_shape!r}"
        )


def _load_ultralytics(path: Path, device: str) -> RuntimeModel:
    ultralytics = cast("Any", import_module("ultralytics"))
    torch = cast("Any", import_module("torch"))
    yolo = ultralytics.YOLO(str(path))
    check_field_model(yolo)
    model = yolo.model.to(device).eval()

    def infer(inputs: Mapping[str, Array]) -> Mapping[str, Array]:
        tensor = torch.from_numpy(inputs["images"]).to(device)
        with torch.inference_mode():
            output = model(tensor)
            prediction = cast("Any", output[0] if isinstance(output, tuple) else output)
            host = np.asarray(prediction.detach().float().cpu().numpy(), dtype=np.float32)
        return {"predictions": cast("Array", np.ascontiguousarray(host))}

    return infer
