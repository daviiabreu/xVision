from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from edge.participants import PlayerTracker
from edge.participants.detector import Frame, RawDetection
from edge.participants.types import Box, Role


class FakeTrackingBackend:
    class_roles: Mapping[int, Role] = {1: "goalkeeper", 2: "player", 3: "referee"}

    def __init__(self) -> None:
        self.reset_count = 0

    def start_clip(self) -> None:
        self.reset_count += 1

    def track(self, frame: Frame) -> tuple[RawDetection, ...]:
        return (
            RawDetection(Box(-5, 10, 120, 90), 0.9, class_id=2, track_id=7),
            RawDetection(Box(20, 20, 30, 30), 0.8, class_id=0, track_id=8),
            RawDetection(Box(30, 30, 50, 60), 0.7, class_id=3, track_id=9),
        )


def test_tracker_maps_roles_ids_and_clips_boxes() -> None:
    tracker = PlayerTracker(backend=FakeTrackingBackend(), device="cpu")

    detections = tracker.track(np.zeros((80, 100, 3), dtype=np.uint8))

    assert [(detection.role, detection.track_id) for detection in detections] == [
        ("player", 7),
        ("referee", 9),
    ]
    assert detections[0].box == Box(0, 10, 100, 80)


def test_tracker_resets_state_for_each_clip() -> None:
    backend = FakeTrackingBackend()
    tracker = PlayerTracker(backend=backend, device="cpu")

    tracker.start_clip()
    tracker.start_clip()

    assert backend.reset_count == 2


def test_tracker_requires_weights_without_injected_backend() -> None:
    with pytest.raises(ValueError, match="weights"):
        PlayerTracker(device="cpu")


def test_tracker_reports_missing_weight() -> None:
    with pytest.raises(FileNotFoundError, match="nao encontrado"):
        PlayerTracker(weights=Path("missing.pt"), device="cpu")
