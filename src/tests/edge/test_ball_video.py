import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from contracts import BallDetection, Box, FramePerception, Perception
from edge.ball_video import BallMarker, draw_ball, write_video
from edge.testing.make_clip import make_synthetic_clip


def test_ball_marker_stays_visible_when_the_video_is_scaled_for_a_player() -> None:
    source = np.full((1080, 2730, 3), 32, dtype=np.uint8)
    ball = BallDetection(Box(496, 496, 504, 504), 0.9, source="detected")
    rendered = draw_ball(source, BallMarker(ball))
    scaled = cv2.resize(rendered, (800, 316), interpolation=cv2.INTER_AREA)
    region = scaled[125:170, 130:250].astype(int)

    assert np.count_nonzero(region.max(axis=2) - region.min(axis=2) > 70) > 100


def test_ball_marker_survives_sampling_but_stops_at_an_explicit_absence(tmp_path: Path) -> None:
    pytest.importorskip("imageio_ffmpeg")
    clip = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter.fourcc(*"mp4v"), 60, (640, 360))
    assert writer.isOpened()
    for _ in range(20):
        writer.write(np.full((360, 640, 3), 32, dtype=np.uint8))
    writer.release()
    ball = BallDetection(Box(296, 196, 304, 204), 0.9, source="detected")
    perception = Perception(
        "clip",
        (
            FramePerception(0, 0, ball=ball),
            FramePerception(4, 4 / 60, ball=ball),
            FramePerception(8, 8 / 60),
            FramePerception(12, 12 / 60, ball=ball),
        ),
    )
    target = tmp_path / "annotated.mp4"

    write_video(clip, target, perception, image_width=640)

    capture = cv2.VideoCapture(str(target))
    marked: list[int] = []
    for index in range(20):
        ok, frame = capture.read()
        assert ok
        region = frame[175:225, 275:325].astype(int)
        if np.count_nonzero(region.max(axis=2) - region.min(axis=2) > 70) > 15:
            marked.append(index)
    capture.release()
    assert marked == [*range(8), *range(12, 19)]


def test_unsampled_ball_is_not_reported_as_a_fresh_detection() -> None:
    source = np.full((360, 640, 3), 32, dtype=np.uint8)
    ball = BallDetection(Box(296, 196, 304, 204), 0.9, source="detected")
    fresh = draw_ball(source, BallMarker(ball))
    held = draw_ball(source, BallMarker(ball, age_s=0.05))

    assert not np.array_equal(fresh[165:220, 275:570], held[165:220, 275:570])
    assert np.array_equal(source, np.full_like(source, 32))


def test_encoder_failure_preserves_previous_render_and_cleans_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("imageio_ffmpeg")
    from edge import ball_video

    target = tmp_path / "out" / "video.mp4"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"previous render")
    clip = make_synthetic_clip(tmp_path / "clip.mp4")

    def fail(*args: object, **kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, "ffmpeg")

    monkeypatch.setattr(ball_video.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        write_video(clip, target, Perception("video-1", ()), image_width=320)
    assert target.read_bytes() == b"previous render"
    assert list(target.parent.iterdir()) == [target]


def test_truncated_decode_does_not_publish_partial_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from edge import ball_video

    clip = make_synthetic_clip(tmp_path / "clip.mp4")
    capture = cv2.VideoCapture(str(clip))
    target = tmp_path / "out" / "video.mp4"
    target.parent.mkdir()
    target.write_bytes(b"previous render")

    class TruncatedCapture:
        reads = 0

        def isOpened(self) -> bool:
            return capture.isOpened()

        def get(self, prop: int) -> float:
            return capture.get(prop)

        def read(self) -> tuple[bool, Any]:
            self.reads += 1
            return capture.read() if self.reads == 1 else (False, None)

        def release(self) -> None:
            capture.release()

    def truncated(path: str) -> TruncatedCapture:
        return TruncatedCapture()

    monkeypatch.setattr(ball_video.cv2, "VideoCapture", truncated)
    with pytest.raises(ValueError, match="incompleto"):
        ball_video.write_video(clip, target, Perception("video-1", ()), image_width=320)
    assert target.read_bytes() == b"previous render"
    assert list(target.parent.iterdir()) == [target]
