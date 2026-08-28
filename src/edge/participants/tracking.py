"""Tracking de participantes com ByteTrack sobre as deteccoes do YOLO."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from edge.participants.detector import (
    Frame,
    RawDetection,
    boxes_to_raw,
    read_class_roles,
    resolve_device,
)
from edge.participants.types import PlayerDetection, Role, clip_box, player


class TrackingBackend(Protocol):
    @property
    def class_roles(self) -> Mapping[int, Role]:
        """Papel por indice de classe, lido do peso que este backend carregou."""
        ...

    def start_clip(self) -> None: ...

    def track(self, frame: Frame) -> tuple[RawDetection, ...]: ...


class UltralyticsTrackingBackend:
    """Adapta o `model.track` usado na implementacao original da Yas."""

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

    def start_clip(self) -> None:
        predictor = getattr(self._model, "predictor", None)
        for tracker in getattr(predictor, "trackers", ()):
            tracker.reset()

    def track(self, frame: Frame) -> tuple[RawDetection, ...]:
        results = self._model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=sorted(self._class_roles),
            imgsz=self._image_size,
            conf=self._confidence_threshold,
            iou=0.7,
            agnostic_nms=True,
            device=self._device,
            verbose=False,
        )
        if not results or results[0].boxes is None:
            return ()
        return boxes_to_raw(results[0].boxes, with_track_ids=True)


class PlayerTracker:
    def __init__(
        self,
        weights: Path | None = None,
        *,
        image_size: int = 1280,
        confidence_threshold: float = 0.25,
        device: str = "auto",
        backend: TrackingBackend | None = None,
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
            self._backend = UltralyticsTrackingBackend(
                weights,
                image_size=image_size,
                confidence_threshold=confidence_threshold,
                device=self.device,
            )

    def start_clip(self) -> None:
        self._backend.start_clip()

    def track(self, frame: Frame) -> tuple[PlayerDetection, ...]:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("o frame deve ter formato HxWx3")

        height, width = frame.shape[:2]
        papeis = self._backend.class_roles
        detections: list[PlayerDetection] = []
        for raw in self._backend.track(frame):
            role = papeis.get(raw.class_id)
            if role is None:
                continue
            detections.append(
                player(
                    box=clip_box(raw.box, width=width, height=height),
                    confidence=raw.confidence,
                    role=role,
                    track_id=raw.track_id,
                )
            )
        return tuple(detections)
