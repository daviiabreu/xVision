import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from xvision_players_yolo.detector import Frame
from xvision_players_yolo.types import Box, PlayerDetection, player
from xvision_players_yolo.video import Trajectories, annotate_frame, process_video


class FakeTracker:
    image_size = 1280
    confidence_threshold = 0.25
    device = "cpu"
    starts = 0

    def start_clip(self) -> None:
        self.starts += 1

    def track(self, frame: Frame) -> tuple[PlayerDetection, ...]:
        return (player(Box(5, 5, 20, 30), 0.9, "player", track_id=4),)


def make_video(path: Path, frame_count: int = 2) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 10, (64, 48))
    assert writer.isOpened()
    for value in range(frame_count):
        writer.write(np.full((48, 64, 3), value * 20, dtype=np.uint8))
    writer.release()


def test_process_video_generates_json_and_annotated_video(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "result"
    make_video(source)

    tracker = FakeTracker()
    result = process_video(source, output, tracker)

    document = json.loads((output / "detections.json").read_text(encoding="utf-8"))
    assert result["schema_version"] == "player-tracking.v1"
    assert document["clip"]["frame_count"] == 2
    assert len(document["frames"]) == 2
    assert document["frames"][0]["players"][0]["role"] == "player"
    assert document["frames"][0]["players"][0]["track_id"] == 4
    assert tracker.starts == 1
    assert (output / "annotated.mp4").is_file()

    capture = cv2.VideoCapture(str(output / "annotated.mp4"))
    assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 64
    assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 48
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 2
    capture.release()


def test_process_video_reports_missing_input(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="não encontrado"):
        process_video(tmp_path / "missing.mp4", tmp_path / "result", FakeTracker())


def test_optional_trail_uses_the_player_ground_point() -> None:
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    trajectories: Trajectories = {}
    first = player(Box(10, 20, 30, 60), 0.9, "player", track_id=4)
    second = player(Box(20, 20, 40, 60), 0.9, "player", track_id=4)

    annotate_frame(frame, (first,), trajectories)
    annotated = annotate_frame(frame, (second,), trajectories)

    assert list(trajectories[4]) == [(20, 60), (30, 60)]
    assert np.any(annotated != frame)
