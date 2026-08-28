import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from edge.participants import Box, Role
from edge.participants.identification import (
    deduplicate_detections,
    load_tracking,
    process_identification,
    resolve_roles_and_teams,
)
from edge.participants.identity import LabColor, Team
from edge.participants.types import player


def make_source_video(path: Path, frame_count: int = 6) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter.fourcc(*"mp4v"),
        10.0,
        (96, 64),
    )
    assert writer.isOpened()
    for _ in range(frame_count):
        frame = np.full((64, 96, 3), (0, 150, 0), dtype=np.uint8)
        frame[8:26, 10:26] = (0, 0, 220)
        frame[8:26, 68:84] = (0, 220, 220)
        frame[8:26, 39:55] = (30, 30, 30)
        writer.write(frame)
    writer.release()


def make_tracking(path: Path, frame_count: int = 6) -> None:
    frames: list[dict[str, object]] = []
    for frame_index in range(frame_count):
        referee_role = "player" if frame_index == 0 else "referee"
        frames.append(
            {
                "frame_index": frame_index,
                "timestamp_s": frame_index / 10,
                "players": [
                    {
                        "box": {"x1": 4, "y1": 4, "x2": 32, "y2": 52},
                        "confidence": 0.9,
                        "role": "player",
                        "track_id": 1,
                    },
                    {
                        "box": {"x1": 62, "y1": 4, "x2": 90, "y2": 52},
                        "confidence": 0.9,
                        "role": "player",
                        "track_id": 2,
                    },
                    {
                        "box": {"x1": 33, "y1": 4, "x2": 61, "y2": 52},
                        "confidence": 0.8,
                        "role": referee_role,
                        "track_id": 3,
                    },
                ],
            }
        )
    path.write_text(
        json.dumps(
            {
                "schema_version": "player-tracking.v1",
                "clip": {
                    "id": "synthetic",
                    "fps": 10.0,
                    "width": 96,
                    "height": 64,
                    "frame_count": frame_count,
                },
                "frames": frames,
            }
        ),
        encoding="utf-8",
    )


def test_identification_processes_tracking_without_running_detector(tmp_path: Path) -> None:
    video_path = tmp_path / "synthetic.mp4"
    tracking_path = tmp_path / "detections.json"
    output_dir = tmp_path / "output"
    make_source_video(video_path)
    make_tracking(tracking_path)

    result = process_identification(video_path, tracking_path, output_dir)

    tracks = {track["track_id"]: track for track in result["tracks"]}
    assert result["schema_version"] == "player-identification.v0"
    assert tracks[1]["status"] == "available"
    assert tracks[2]["status"] == "available"
    assert tracks[1]["team"] != tracks[2]["team"]
    assert tracks[3]["stable_role"] == "referee"
    assert tracks[3]["status"] == "not_applicable"
    assert tracks[3]["team"] is None
    assert (output_dir / "identities.json").is_file()
    assert (output_dir / "identified.mp4").is_file()

    capture = cv2.VideoCapture(str(output_dir / "identified.mp4"))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 6
    capture.release()


def test_identification_rejects_video_from_another_clip(tmp_path: Path) -> None:
    video_path = tmp_path / "another-clip.mp4"
    tracking_path = tmp_path / "detections.json"
    make_source_video(video_path)
    make_tracking(tracking_path)

    with pytest.raises(ValueError, match="clip.id"):
        process_identification(video_path, tracking_path, tmp_path / "output")


def test_colors_correct_swapped_player_and_referee_roles() -> None:
    stable_roles: dict[int, Role] = {
        1: "player",
        2: "player",
        3: "referee",
        4: "referee",
        5: "player",
    }
    colors: dict[int, LabColor] = {
        1: (210, 110, 140),
        2: (220, 120, 210),
        3: (70, 126, 132),
        4: (218, 119, 208),
        5: (72, 125, 133),
    }
    seed_assignments: dict[int, Team] = {1: "team_a", 2: "team_b", 5: "team_a"}
    centroids: dict[Team, LabColor] = {
        "team_a": (210, 110, 140),
        "team_b": (220, 120, 210),
    }

    roles, teams, sources = resolve_roles_and_teams(
        stable_roles,
        colors,
        seed_assignments,
        centroids,
        role_evidence={5: {"player", "referee"}},
    )

    assert roles[3] == "referee"
    assert roles[4] == "player"
    assert teams[4] == "team_b"
    assert sources[4] == "color_assisted"
    assert roles[5] == "referee"
    assert 5 not in teams
    assert sources[5] == "color_assisted"


def test_clean_referee_sample_can_correct_a_contaminated_track_color() -> None:
    roles, teams, sources = resolve_roles_and_teams(
        {1: "player", 2: "player", 3: "referee", 4: "player"},
        {
            1: (190, 120, 140),
            2: (210, 120, 205),
            3: (65, 124, 136),
            4: (180, 124, 138),
        },
        {1: "team_a", 2: "team_b", 4: "team_a"},
        {"team_a": (185, 122, 139), "team_b": (210, 120, 205)},
        role_evidence={4: {"player", "referee"}},
        color_samples={4: [(180, 124, 138), (64, 124, 136)]},
    )

    assert roles[4] == "referee"
    assert 4 not in teams
    assert sources[4] == "color_assisted"


def test_overlapping_cross_class_detections_keep_highest_confidence() -> None:
    detections = (
        player(Box(10, 10, 30, 50), 0.9, "referee", 3),
        player(Box(11, 11, 31, 51), 0.6, "player", 5),
        player(Box(60, 10, 80, 50), 0.8, "player", 8),
    )

    deduplicated = deduplicate_detections(detections)

    assert [detection.track_id for detection in deduplicated] == [3, 8]


def test_offset_cross_class_detections_are_deduplicated() -> None:
    detections = (
        player(Box(10, 10, 30, 50), 0.6, "player", 3),
        player(Box(11, 25, 31, 70), 0.9, "referee", 5),
    )

    deduplicated = deduplicate_detections(detections)

    assert [detection.track_id for detection in deduplicated] == [5]


def test_offset_referee_detections_are_deduplicated() -> None:
    detections = (
        player(Box(10, 10, 30, 50), 0.6, "referee", 3),
        player(Box(11, 25, 31, 70), 0.9, "referee", 5),
    )

    deduplicated = deduplicate_detections(detections)

    assert [detection.track_id for detection in deduplicated] == [5]


def downgrade_to_v0(path: Path) -> None:
    """Reescreve o artefato no formato antigo: `detections` em vez de `players`."""
    document = json.loads(path.read_text(encoding="utf-8"))
    document["schema_version"] = "player-tracking.v0"
    for frame in document["frames"]:
        frame["detections"] = frame.pop("players")
    path.write_text(json.dumps(document), encoding="utf-8")


def test_le_o_tracking_no_formato_antigo_que_o_grupo_ja_produziu(tmp_path: Path) -> None:
    """Artefato em `v0` continua legivel, sem reprocessar video."""
    v1_path = tmp_path / "v1.json"
    v0_path = tmp_path / "v0.json"
    make_tracking(v1_path)
    make_tracking(v0_path)
    downgrade_to_v0(v0_path)

    antigo = load_tracking(v0_path)
    novo = load_tracking(v1_path)

    assert antigo == novo


def test_schema_desconhecido_diz_o_que_a_identificacao_le(tmp_path: Path) -> None:
    path = tmp_path / "futuro.json"
    make_tracking(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["schema_version"] = "player-tracking.v99"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="player-tracking.v0, player-tracking.v1"):
        load_tracking(path)
