"""Acrescenta a bola da pipeline ao video, preservando seu painel de top view."""

from __future__ import annotations

import math
import subprocess
import tempfile
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np

from contracts import BallDetection, FramePerception, Perception

Image = np.ndarray[Any, Any]
MAX_BALL_DISPLAY_AGE_S = 0.1


@dataclass(frozen=True)
class BallMarker:
    detection: BallDetection
    age_s: float = 0


def _draw_ball(image: Image, marker: BallMarker) -> None:
    ball = marker.detection
    box = ball.box
    height, width = image.shape[:2]
    center = (round((box.x1 + box.x2) / 2), round((box.y1 + box.y2) / 2))
    if not (0 <= center[0] < width and 0 <= center[1] < height):
        return
    radius = max(10, round(height / 54))
    stroke = max(2, round(height / 360))
    color = (255, 80, 255) if ball.source == "detected" else (60, 170, 250)
    status = {"detected": "detectada", "interpolated": "interpolada"}.get(
        ball.source or "", "origem desconhecida"
    )
    if marker.age_s > 0:
        status = "ultima posicao"
    label = f"BOLA: {status}"
    scale = max(0.5, height / 900)
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, stroke)
    left = max(4, min(center[0] + radius + 8, width - label_width - 16))
    bottom = max(label_height + 12, min(center[1] - radius - 8, height - 8))
    top = bottom - label_height - 8
    dx, dy = left - center[0], bottom - center[1]
    distance = max(1.0, math.hypot(dx, dy))
    pointer = (round(center[0] + radius * dx / distance), round(center[1] + radius * dy / distance))
    cv2.line(image, pointer, (left, bottom), (0, 0, 0), stroke + 4, cv2.LINE_AA)
    cv2.line(image, pointer, (left, bottom), color, stroke, cv2.LINE_AA)
    cv2.circle(image, center, radius, (0, 0, 0), stroke + 4, cv2.LINE_AA)
    cv2.circle(image, center, radius, color, stroke, cv2.LINE_AA)
    cv2.rectangle(image, (left, top), (left + label_width + 12, bottom + 8), (0, 0, 0), -1)
    cv2.rectangle(image, (left, top), (left + label_width + 12, bottom + 8), color, stroke)
    cv2.putText(
        image,
        label,
        (left + 6, bottom),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        stroke,
        cv2.LINE_AA,
    )


def draw_ball(frame: Image, marker: BallMarker) -> Image:
    image = frame.copy()
    if 0 <= marker.age_s <= MAX_BALL_DISPLAY_AGE_S:
        _draw_ball(image, marker)
    return image


def write_video(clip: Path, destination: Path, perception: Perception, *, image_width: int) -> None:
    """MP4 H.264 completo ou nenhum arquivo novo; temporarios ficam somente em var/."""
    capture = cv2.VideoCapture(str(clip))
    try:
        if not capture.isOpened():
            raise ValueError("video indisponivel para render")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("video sem FPS valido para render")
        declared_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(declared_count) or declared_count <= 0:
            raise ValueError("video sem contagem valida de quadros para render")
        expected_count = round(declared_count)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not 0 < image_width <= width:
            raise ValueError("largura invalida para desenhar a bola na imagem original")
        frames = {frame.frame_index: frame for frame in perception.frames}
        if len(frames) != len(perception.frames):
            raise ValueError("percepcao com quadros duplicados")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="ball-overlay-", dir=destination.parent
        ) as temporary:
            raw = Path(temporary) / "raw.mp4"
            target = Path(temporary) / "video.mp4"
            writer = cv2.VideoWriter(
                str(raw),
                cv2.VideoWriter.fourcc(*"mp4v"),
                fps,
                (width, height),
            )
            count = 0
            last_sample: FramePerception | None = None
            try:
                if not writer.isOpened():
                    raise ValueError("encoder indisponivel para render")
                while True:
                    ok, image = capture.read()
                    if not ok:
                        break
                    sampled = frames.get(count)
                    if sampled is not None:
                        last_sample = sampled
                    marker = None
                    if last_sample is not None:
                        age = (count - last_sample.frame_index) / fps
                        if last_sample.ball is not None and age <= MAX_BALL_DISPLAY_AGE_S:
                            marker = BallMarker(last_sample.ball, age_s=age)
                    if marker is not None:
                        _draw_ball(image[:, :image_width], marker)
                    writer.write(image)
                    count += 1
            finally:
                writer.release()
            if count == 0:
                raise ValueError("video sem quadros para render")
            if count != expected_count:
                raise ValueError("decode incompleto; render anterior preservado")
            ffmpeg = cast("Any", import_module("imageio_ffmpeg")).get_ffmpeg_exe()
            subprocess.run(
                [
                    ffmpeg,
                    "-nostdin",
                    "-v",
                    "error",
                    "-i",
                    str(raw),
                    "-an",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-vf",
                    "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-movflags",
                    "+faststart",
                    str(target),
                ],
                check=True,
                capture_output=True,
            )
            target.replace(destination)
    finally:
        capture.release()
