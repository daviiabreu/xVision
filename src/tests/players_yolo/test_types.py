import pytest

from xvision_players_yolo.types import (
    Box,
    FramePerception,
    UnknownClassNames,
    clip_box,
    frame_to_dict,
    ground_point,
    player,
    roles_from_names,
)


def test_box_is_clipped_to_frame_dimensions() -> None:
    box = Box(x1=-10, y1=20, x2=130, y2=90)

    assert clip_box(box, width=100, height=80) == Box(x1=0, y1=20, x2=100, y2=80)


def test_ground_point_is_the_bottom_center_of_the_box() -> None:
    assert ground_point(Box(10, 20, 30, 80)) == (20, 80)


def test_player_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        player(box=Box(0, 0, 10, 10), confidence=1.1, role="player")


def test_player_rejects_negative_track_id() -> None:
    with pytest.raises(ValueError, match="track_id"):
        player(Box(0, 0, 10, 10), 0.9, "player", track_id=-1)


def test_frame_serialization_uses_the_contract_vocabulary() -> None:
    frame = FramePerception(
        frame_index=7,
        timestamp_s=0.28,
        players=(
            player(
                box=Box(x1=1, y1=2, x2=30, y2=40),
                confidence=0.91,
                role="goalkeeper",
                track_id=12,
            ),
        ),
    )

    assert frame_to_dict(frame) == {
        "frame_index": 7,
        "timestamp_s": 0.28,
        "players": [
            {
                "box": {"x1": 1, "y1": 2, "x2": 30, "y2": 40},
                "confidence": 0.91,
                "track_id": 12,
                "team": None,
                "role": "goalkeeper",
                "jersey_number": None,
                "pitch_xy": None,
            }
        ],
    }


def test_empty_frame_is_supported() -> None:
    assert frame_to_dict(FramePerception(frame_index=0, timestamp_s=0))["players"] == []


def test_roles_come_from_class_names_and_not_from_indexes() -> None:
    """O peso do roboflow e o do `train_player_detector.py` usam indices diferentes."""
    roboflow = roles_from_names({0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"})
    treinado_no_repo = roles_from_names({0: "jogador"})

    assert roboflow == {1: "goalkeeper", 2: "player", 3: "referee"}
    assert treinado_no_repo == {0: "player"}


def test_unknown_class_names_are_ignored_when_something_else_maps() -> None:
    assert roles_from_names({0: "player", 1: "corner_flag"}) == {0: "player"}


def test_a_weight_with_no_known_class_is_refused_instead_of_detecting_nothing() -> None:
    with pytest.raises(UnknownClassNames, match="nenhuma delas e um papel conhecido"):
        roles_from_names({0: "car", 1: "dog"})


def test_class_names_are_matched_case_and_space_insensitively() -> None:
    assert roles_from_names({0: "  Goalkeeper "}) == {0: "goalkeeper"}
