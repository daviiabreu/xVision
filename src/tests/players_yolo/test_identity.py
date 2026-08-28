import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from xvision_players_yolo import Box
from xvision_players_yolo.identity import (
    TrackObservation,
    cluster_teams,
    extract_shirt_color,
    representative_shirt_color,
    select_sample_observations,
    stabilize_role,
)
from xvision_players_yolo.types import player


def observation(
    frame_index: int,
    role: str,
    confidence: float,
    *,
    track_id: int = 4,
    width: int = 20,
) -> TrackObservation:
    return TrackObservation(
        frame_index,
        player(Box(0, 0, width, 40), confidence, role, track_id),  # type: ignore[arg-type]
    )


def test_role_is_stabilized_by_accumulated_confidence() -> None:
    stable_role, confidence = stabilize_role(
        (
            observation(0, "goalkeeper", 0.9),
            observation(1, "player", 0.6),
            observation(2, "goalkeeper", 0.8),
        )
    )

    assert stable_role == "goalkeeper"
    assert confidence == pytest.approx(1.7 / 2.3)


def test_sample_selection_prefers_confidence_then_box_area() -> None:
    selected = select_sample_observations(
        (
            observation(0, "player", 0.8, width=30),
            observation(1, "player", 0.9, width=10),
            observation(2, "referee", 0.99),
            observation(3, "player", 0.8, width=20),
        ),
        limit=2,
    )

    assert [item.frame_index for item in selected] == [0, 1]


def test_sample_selection_can_include_every_role() -> None:
    selected = select_sample_observations(
        (
            observation(0, "player", 0.8),
            observation(1, "referee", 0.9),
        ),
        role=None,
    )

    assert [item.frame_index for item in selected] == [0, 1]


def test_observation_requires_track_id() -> None:
    with pytest.raises(ValueError, match="track_id"):
        TrackObservation(0, player(Box(0, 0, 10, 20), 0.8, "player"))


def test_shirt_color_discards_background_from_crop_corners() -> None:
    frame = np.full((40, 40, 3), (0, 160, 0), dtype=np.uint8)
    frame[2:18, 10:30] = (0, 0, 220)

    color = extract_shirt_color(frame, Box(0, 0, 40, 40))
    red_pixel: NDArray[np.uint8] = np.array([[[0, 0, 220]]], dtype=np.uint8)
    expected = cv2.cvtColor(red_pixel, cv2.COLOR_BGR2LAB)[0, 0]

    assert color == pytest.approx(tuple(float(value) for value in expected), abs=1)


def test_shirt_color_is_unavailable_without_color_contrast() -> None:
    frame = np.full((20, 20, 3), (0, 160, 0), dtype=np.uint8)

    assert extract_shirt_color(frame, Box(0, 0, 20, 20)) is None


def test_representative_color_uses_median_of_samples() -> None:
    assert representative_shirt_color(((10, 20, 30), (12, 22, 32), (200, 200, 200))) == (
        12,
        22,
        32,
    )


def test_team_clustering_is_deterministic_for_mapping_order() -> None:
    colors = {
        4: (40, 80, 70),
        1: (42, 82, 72),
        9: (180, 140, 190),
        2: (178, 138, 188),
    }

    first = cluster_teams(colors)
    second = cluster_teams(dict(reversed(tuple(colors.items()))))

    assert first is not None
    assert second is not None
    assert first == second
    assignments, _ = first
    assert assignments[1] == assignments[4]
    assert assignments[2] == assignments[9]
    assert assignments[1] != assignments[2]


def test_team_clustering_requires_two_distinct_colors() -> None:
    assert cluster_teams({1: (10, 20, 30)}) is None
    assert cluster_teams({1: (10, 20, 30), 2: (10, 20, 30)}) is None
