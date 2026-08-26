"""Processamento offline de um clipe e geração dos artefatos da v0."""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Protocol, TypedDict, cast

import cv2

from xvision_players_yolo.detector import Frame
from xvision_players_yolo.types import (
    FramePerception,
    PlayerDetection,
    frame_to_dict,
    ground_point,
)

UNKNOWN_COLOR = (200, 200, 200)
COLORS: Mapping[str, tuple[int, int, int]] = {
    "player": (255, 120, 0),
    "goalkeeper": (0, 215, 255),
    "referee": (255, 0, 180),
}
SCHEMA_VERSION = "player-tracking.v1"
TRAIL_LENGTH = 35
Trajectories = dict[int, deque[tuple[int, int]]]


class FrameTracker(Protocol):
    image_size: int
    confidence_threshold: float
    device: str

    def start_clip(self) -> None: ...

    def track(self, frame: Frame) -> tuple[PlayerDetection, ...]: ...


class ClipInfo(TypedDict):
    id: str
    fps: float
    width: int
    height: int
    frame_count: int


class Metrics(TypedDict):
    elapsed_seconds: float
    average_inference_ms: float
    processing_fps: float


class VideoResult(TypedDict):
    schema_version: str
    clip: ClipInfo
    model: dict[str, object]
    metrics: Metrics
    frames: list[dict[str, object]]


def _ground_point(detection: PlayerDetection, width: int, height: int) -> tuple[int, int]:
    x, y = ground_point(detection.box)
    return (
        min(max(round(x), 0), width - 1),
        min(max(round(y), 0), height - 1),
    )


def _draw_badge(
    frame: Frame,
    center: tuple[int, int],
    label: str,
    color: tuple[int, int, int],
    ellipse_height: int,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    padding = 4
    text_size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
    text_width, text_height = text_size
    frame_height, frame_width = frame.shape[:2]
    badge_width = text_width + 2 * padding
    badge_height = text_height + baseline + 2 * padding
    left = min(max(center[0] - badge_width // 2, 0), max(frame_width - badge_width, 0))
    top = center[1] + ellipse_height + 4
    if top + badge_height >= frame_height:
        top = max(center[1] - ellipse_height - badge_height - 4, 0)
    right = min(left + badge_width, frame_width - 1)
    bottom = min(top + badge_height, frame_height - 1)

    cv2.rectangle(frame, (left, top), (right, bottom), color, -1)
    cv2.putText(
        frame,
        label,
        (left + padding, min(top + padding + text_height, frame_height - 1)),
        font,
        font_scale,
        (20, 20, 20),
        thickness,
        cv2.LINE_AA,
    )


def annotate_frame(
    frame: Frame,
    detections: tuple[PlayerDetection, ...],
    trajectories: Trajectories | None = None,
) -> Frame:
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    for detection in detections:
        box = detection.box
        color = COLORS.get(detection.role or "", UNKNOWN_COLOR)
        center = _ground_point(detection, width, height)
        box_width = max(round(box.x2 - box.x1), 1)
        axes = (max(box_width // 2, 4), max(box_width // 8, 2))

        if trajectories is not None and detection.track_id is not None:
            trail = trajectories.setdefault(
                detection.track_id,
                deque(maxlen=TRAIL_LENGTH),
            )
            trail.append(center)
            points = tuple(trail)
            for point_a, point_b in zip(points, points[1:], strict=False):
                cv2.line(annotated, point_a, point_b, color, 2, cv2.LINE_AA)

        cv2.ellipse(
            annotated,
            center,
            axes,
            0,
            -45,
            235,
            color,
            2,
            cv2.LINE_AA,
        )
        identifier = "?" if detection.track_id is None else str(detection.track_id)
        label = f"#{identifier} {detection.role or '?'} {detection.confidence:.2f}"
        _draw_badge(annotated, center, label, color, axes[1])
    return annotated


def process_video(
    input_path: Path,
    output_dir: Path,
    tracker: FrameTracker,
    show_trails: bool = False,
) -> VideoResult:
    if not input_path.is_file():
        raise FileNotFoundError(f"vídeo não encontrado: {input_path}")

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError(f"não foi possível abrir o vídeo: {input_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError(f"metadados de vídeo inválidos: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "annotated.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"não foi possível criar o vídeo de saída: {video_path}")

    frames: list[dict[str, object]] = []
    inference_seconds = 0.0
    started_at = perf_counter()
    frame_index = 0
    trajectories: Trajectories | None = {} if show_trails else None
    tracker.start_clip()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            typed_frame = cast(Frame, frame)
            inference_started = perf_counter()
            detections = tracker.track(typed_frame)
            inference_seconds += perf_counter() - inference_started
            frame_result = FramePerception(frame_index, frame_index / fps, detections)
            frames.append(frame_to_dict(frame_result))
            writer.write(annotate_frame(typed_frame, detections, trajectories))
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    if frame_index == 0:
        raise ValueError(f"o vídeo não possui frames decodificáveis: {input_path}")

    elapsed_seconds = perf_counter() - started_at
    result: VideoResult = {
        "schema_version": SCHEMA_VERSION,
        "clip": {
            "id": input_path.stem,
            "fps": fps,
            "width": width,
            "height": height,
            "frame_count": frame_index,
        },
        "model": {
            "name": "football-player-detection",
            "image_size": tracker.image_size,
            "confidence_threshold": tracker.confidence_threshold,
            "device": tracker.device,
        },
        "metrics": {
            "elapsed_seconds": elapsed_seconds,
            "average_inference_ms": inference_seconds * 1000 / frame_index,
            "processing_fps": frame_index / elapsed_seconds,
        },
        "frames": frames,
    }
    json_path = output_dir / "detections.json"
    temporary_path = output_dir / "detections.json.tmp"
    temporary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_path.replace(json_path)
    return result
