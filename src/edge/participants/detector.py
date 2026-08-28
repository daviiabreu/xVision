"""Adaptador minimo entre o Ultralytics YOLO e o contrato do xVision."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from edge.participants.types import (
    Box,
    PlayerDetection,
    Role,
    clip_box,
    player,
    roles_from_names,
)

Frame = NDArray[np.uint8]


@dataclass(frozen=True)
class RawDetection:
    box: Box
    confidence: float
    class_id: int
    track_id: int | None = None


class DetectionBackend(Protocol):
    @property
    def class_roles(self) -> Mapping[int, Role]:
        """Papel por indice de classe, lido do peso que este backend carregou."""
        ...

    def infer(self, frame: Frame) -> tuple[RawDetection, ...]: ...


def resolve_device(requested: str) -> str:
    """Escolhe MPS no Mac, CUDA quando disponivel e CPU como fallback."""
    if requested != "auto":
        return requested

    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def read_class_roles(model: Any) -> dict[int, Role]:
    """Le `model.names` do peso carregado e traduz para o vocabulario de papeis."""
    names = cast("Mapping[int, str]", getattr(model, "names", {}) or {})
    return roles_from_names(names)


def boxes_to_raw(boxes: Any, *, with_track_ids: bool = False) -> tuple[RawDetection, ...]:
    """Converte o tensor de caixas do Ultralytics, que os dois backends leem igual."""
    coordinates = cast("NDArray[np.float32]", boxes.xyxy.detach().cpu().numpy())
    confidences = cast("NDArray[np.float32]", boxes.conf.detach().cpu().numpy())
    class_ids = cast("NDArray[np.float32]", boxes.cls.detach().cpu().numpy())
    if not with_track_ids or boxes.id is None:
        track_ids: list[int | None] = [None] * len(coordinates)
    else:
        track_ids = [int(value) for value in boxes.id.detach().cpu().tolist()]

    return tuple(
        RawDetection(
            box=Box(*(float(value) for value in xyxy)),
            confidence=float(confidence),
            class_id=int(class_id),
            track_id=track_id,
        )
        for xyxy, confidence, class_id, track_id in zip(
            coordinates, confidences, class_ids, track_ids, strict=True
        )
    )


class UltralyticsBackend:
    def __init__(
        self,
        weights: Path,
        *,
        image_size: int,
        confidence_threshold: float,
        device: str,
    ) -> None:
        if not weights.is_file():
            raise FileNotFoundError(f"peso do modelo nao encontrado: {weights}")

        from ultralytics import YOLO

        self._model: Any = YOLO(str(weights))
        self._class_roles = read_class_roles(self._model)
        self._image_size = image_size
        self._confidence_threshold = confidence_threshold
        self._device = device

    @property
    def class_roles(self) -> Mapping[int, Role]:
        return self._class_roles

    def infer(self, frame: Frame) -> tuple[RawDetection, ...]:
        results = self._model.predict(
            source=frame,
            imgsz=self._image_size,
            conf=self._confidence_threshold,
            device=self._device,
            verbose=False,
        )
        if not results or results[0].boxes is None:
            return ()
        return boxes_to_raw(results[0].boxes)


class PlayerDetector:
    def __init__(
        self,
        weights: Path | None = None,
        *,
        image_size: int = 1280,
        confidence_threshold: float = 0.25,
        device: str = "auto",
        backend: DetectionBackend | None = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold deve estar entre 0 e 1")

        self.image_size = image_size
        self.confidence_threshold = confidence_threshold
        self.device = resolve_device(device)

        if backend is not None:
            self._backend = backend
        elif weights is None:
            raise ValueError("weights e obrigatorio quando nenhum backend e fornecido")
        else:
            self._backend = UltralyticsBackend(
                weights,
                image_size=image_size,
                confidence_threshold=confidence_threshold,
                device=self.device,
            )

    def detect(self, frame: Frame) -> tuple[PlayerDetection, ...]:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("o frame deve ter formato HxWx3")

        height, width = frame.shape[:2]
        papeis = self._backend.class_roles
        detections: list[PlayerDetection] = []
        for raw in self._backend.infer(frame):
            role = papeis.get(raw.class_id)
            if role is None:
                continue
            detections.append(
                player(
                    box=clip_box(raw.box, width=width, height=height),
                    confidence=raw.confidence,
                    role=role,
                )
            )
        return tuple(detections)
