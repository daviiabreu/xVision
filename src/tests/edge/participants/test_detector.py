from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from edge.participants import PlayerDetector
from edge.participants.detector import Frame, RawDetection, read_class_roles
from edge.participants.types import Box, Role, UnknownClassNames

ROBOFLOW_ROLES: Mapping[int, Role] = {1: "goalkeeper", 2: "player", 3: "referee"}


class FakeBackend:
    class_roles: Mapping[int, Role] = ROBOFLOW_ROLES

    def infer(self, frame: Frame) -> tuple[RawDetection, ...]:
        return (
            RawDetection(Box(-5, 10, 120, 90), confidence=0.9, class_id=2),
            RawDetection(Box(20, 20, 30, 30), confidence=0.8, class_id=0),
            RawDetection(Box(30, 30, 50, 60), confidence=0.7, class_id=3),
        )


class FakeModel:
    def __init__(self, names: Mapping[int, str]) -> None:
        self.names = names


def test_detector_filters_ball_maps_roles_and_clips_boxes() -> None:
    detector = PlayerDetector(backend=FakeBackend(), device="cpu")
    frame = np.zeros((80, 100, 3), dtype=np.uint8)

    detections = detector.detect(frame)

    assert [detection.role for detection in detections] == ["player", "referee"]
    assert detections[0].box == Box(0, 10, 100, 80)


def test_detector_rejects_frame_with_invalid_shape() -> None:
    detector = PlayerDetector(backend=FakeBackend(), device="cpu")

    with pytest.raises(ValueError, match="HxWx3"):
        detector.detect(np.zeros((80, 100), dtype=np.uint8))


def test_detector_requires_weights_without_injected_backend() -> None:
    with pytest.raises(ValueError, match="weights"):
        PlayerDetector(device="cpu")


def test_detector_reports_missing_weight() -> None:
    with pytest.raises(FileNotFoundError, match="nao encontrado"):
        PlayerDetector(weights=Path("missing.pt"), device="cpu")


def test_class_roles_are_read_from_the_loaded_weight() -> None:
    roboflow = FakeModel({0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"})
    treinado_no_repo = FakeModel({0: "jogador"})

    assert read_class_roles(roboflow) == ROBOFLOW_ROLES
    assert read_class_roles(treinado_no_repo) == {0: "player"}


def test_a_weight_with_no_known_class_is_refused_at_load() -> None:
    with pytest.raises(UnknownClassNames):
        read_class_roles(FakeModel({0: "car"}))
