"""Pipeline de classificação de jogadores por equipe sobre tracking existente."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, replace
from math import dist
from pathlib import Path
from statistics import median
from typing import Any, Literal, TypedDict, cast

import cv2

from edge.participants.detector import Frame
from edge.participants.identity import (
    LabColor,
    Team,
    TrackObservation,
    cluster_teams,
    extract_shirt_color,
    representative_shirt_color,
    select_sample_observations,
    stabilize_role,
)
from edge.participants.types import (
    Box,
    FramePerception,
    PlayerDetection,
    Role,
    as_role,
    player,
)
from edge.participants.video import SCHEMA_VERSION, annotate_frame

IDENTIFICATION_SCHEMA_VERSION = "player-identification.v0"
ACCEPTED_TRACKING_SCHEMAS = frozenset({SCHEMA_VERSION, "player-tracking.v0"})
"""A `v0` usava `frames[].detections`; a `v1` usa `players`. Artefatos antigos seguem legiveis."""
IdentityStatus = Literal["available", "unavailable", "not_applicable"]
RoleSource = Literal["temporal_vote", "color_assisted"]


@dataclass(frozen=True)
class TrackingInput:
    clip_id: str
    fps: float
    width: int
    height: int
    frame_count: int
    frames: tuple[FramePerception, ...]


class TrackIdentity(TypedDict):
    track_id: int
    stable_role: Role
    stable_role_confidence: float
    resolved_role: Role
    role_source: RoleSource
    team: Team | None
    sample_count: int
    status: IdentityStatus
    reason: str | None


class IdentificationResult(TypedDict):
    schema_version: str
    clip: dict[str, object]
    source: dict[str, object]
    config: dict[str, object]
    teams: dict[Team, dict[str, object]]
    tracks: list[TrackIdentity]


def _parse_detection(raw: dict[str, Any]) -> PlayerDetection:
    raw_box = cast(dict[str, Any], raw["box"])
    role = as_role(str(raw["role"]))
    if role is None:
        raise ValueError(f"função inválida no tracking: {raw['role']}")
    track_id_value = raw.get("track_id")
    track_id = None if track_id_value is None else int(track_id_value)
    return player(
        box=Box(
            float(raw_box["x1"]),
            float(raw_box["y1"]),
            float(raw_box["x2"]),
            float(raw_box["y2"]),
        ),
        confidence=float(raw["confidence"]),
        role=role,
        track_id=track_id,
    )


def _frame_players(raw_frame: dict[str, Any]) -> list[dict[str, Any]]:
    """Le a lista do quadro pela chave da `v1`, caindo na da `v0`."""
    bruto = raw_frame["players"] if "players" in raw_frame else raw_frame["detections"]
    return cast("list[dict[str, Any]]", bruto)


def load_tracking(path: Path) -> TrackingInput:
    if not path.is_file():
        raise FileNotFoundError(f"JSON de tracking não encontrado: {path}")
    try:
        document = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
        if document.get("schema_version") not in ACCEPTED_TRACKING_SCHEMAS:
            aceitos = ", ".join(sorted(ACCEPTED_TRACKING_SCHEMAS))
            raise ValueError(
                f"schema de tracking incompatível: {document.get('schema_version')}. "
                f"A identificação lê {aceitos}"
            )
        raw_clip = cast(dict[str, Any], document["clip"])
        raw_frames = cast(list[dict[str, Any]], document["frames"])
        frames = tuple(
            FramePerception(
                frame_index=int(raw_frame["frame_index"]),
                timestamp_s=float(raw_frame["timestamp_s"]),
                players=tuple(
                    _parse_detection(raw_player) for raw_player in _frame_players(raw_frame)
                ),
            )
            for raw_frame in raw_frames
        )
        tracking = TrackingInput(
            clip_id=str(raw_clip["id"]),
            fps=float(raw_clip["fps"]),
            width=int(raw_clip["width"]),
            height=int(raw_clip["height"]),
            frame_count=int(raw_clip["frame_count"]),
            frames=frames,
        )
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"JSON de tracking inválido: {path}") from error

    if tracking.frame_count != len(tracking.frames):
        raise ValueError("frame_count difere da quantidade de registros do tracking")
    if any(frame.frame_index != index for index, frame in enumerate(tracking.frames)):
        raise ValueError("frames do tracking não estão em ordem contínua")
    return tracking


def _validate_video(capture: cv2.VideoCapture, path: Path, tracking: TrackingInput) -> None:
    if not capture.isOpened():
        raise ValueError(f"não foi possível abrir o vídeo: {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if (
        (width, height) != (tracking.width, tracking.height)
        or abs(fps - tracking.fps) > 0.02
        or frame_count != tracking.frame_count
    ):
        raise ValueError("vídeo e tracking possuem metadados incompatíveis")


def _group_observations(tracking: TrackingInput) -> dict[int, list[TrackObservation]]:
    observations: dict[int, list[TrackObservation]] = defaultdict(list)
    for frame in tracking.frames:
        for detection in frame.players:
            if detection.track_id is not None:
                observations[detection.track_id].append(
                    TrackObservation(frame.frame_index, detection)
                )
    return dict(observations)


def _nearest_team(color: LabColor, centroids: dict[Team, LabColor]) -> tuple[Team, float]:
    distances: dict[Team, float] = {
        team: dist(color[1:], centroid[1:]) for team, centroid in centroids.items()
    }
    return min(distances.items(), key=lambda item: (item[1], item[0]))


def _robust_distance_limit(values: list[float]) -> float:
    if not values:
        return 30.0
    center = median(values)
    deviation = median(abs(value - center) for value in values)
    return min(max(center + 3 * deviation + 5, 18.0), 60.0)


def resolve_roles_and_teams(
    stable_roles: dict[int, Role],
    colors: dict[int, LabColor],
    seed_assignments: dict[int, Team],
    centroids: dict[Team, LabColor],
    *,
    role_evidence: dict[int, set[Role]] | None = None,
    color_samples: dict[int, list[LabColor]] | None = None,
) -> tuple[dict[int, Role], dict[int, Team], dict[int, RoleSource]]:
    resolved_roles = dict(stable_roles)
    resolved_teams: dict[int, Team] = {}
    sources: dict[int, RoleSource] = {track_id: "temporal_vote" for track_id in stable_roles}
    if len(centroids) != 2:
        return resolved_roles, resolved_teams, sources
    role_evidence = role_evidence or {}
    color_samples = color_samples or {}

    distances_by_team: dict[Team, list[float]] = defaultdict(list)
    for track_id, team in seed_assignments.items():
        color = colors.get(track_id)
        if color is not None:
            distances_by_team[team].append(dist(color[1:], centroids[team][1:]))
    team_limits = {
        team: max(_robust_distance_limit(distances_by_team.get(team, [])), 30.0)
        for team in centroids
    }
    full_distances_by_team: dict[Team, list[float]] = defaultdict(list)
    for track_id, team in seed_assignments.items():
        color = colors.get(track_id)
        if color is not None:
            full_distances_by_team[team].append(dist(color, centroids[team]))
    full_team_limits = {
        team: _robust_distance_limit(full_distances_by_team.get(team, [])) for team in centroids
    }

    referee_candidates: list[LabColor] = []
    for track_id, role in stable_roles.items():
        color = colors.get(track_id)
        if role != "referee" or color is None:
            continue
        nearest_full_team, full_team_distance = min(
            ((team, dist(color, centroid)) for team, centroid in centroids.items()),
            key=lambda item: (item[1], item[0]),
        )
        if full_team_distance > full_team_limits[nearest_full_team]:
            referee_candidates.append(color)
    referee_centroid = representative_shirt_color(referee_candidates)
    referee_limit = _robust_distance_limit(
        [dist(color, referee_centroid) for color in referee_candidates]
        if referee_centroid is not None
        else []
    )

    for track_id, stable_role in stable_roles.items():
        if stable_role == "goalkeeper":
            continue
        color = colors.get(track_id)
        if color is None:
            continue

        nearest_team, team_distance = _nearest_team(color, centroids)
        referee_distance = (
            dist(color, referee_centroid) if referee_centroid is not None else float("inf")
        )
        sample_referee_distance = (
            min(
                (dist(sample, referee_centroid) for sample in color_samples.get(track_id, ())),
                default=float("inf"),
            )
            if referee_centroid is not None
            else float("inf")
        )
        has_referee_evidence = stable_role == "referee" or "referee" in role_evidence.get(
            track_id, set()
        )
        if has_referee_evidence and (
            sample_referee_distance <= referee_limit or referee_distance <= referee_limit
        ):
            resolved_roles[track_id] = "referee"
            if stable_role != "referee":
                sources[track_id] = "color_assisted"
        elif team_distance <= team_limits[nearest_team]:
            resolved_roles[track_id] = "player"
            resolved_teams[track_id] = nearest_team
            if stable_role != "player":
                sources[track_id] = "color_assisted"

    return resolved_roles, resolved_teams, sources


def _box_iou(first: Box, second: Box) -> float:
    x1 = max(first.x1, second.x1)
    y1 = max(first.y1, second.y1)
    x2 = min(first.x2, second.x2)
    y2 = min(first.y2, second.y2)
    intersection = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    first_area = max(first.x2 - first.x1, 0.0) * max(first.y2 - first.y1, 0.0)
    second_area = max(second.x2 - second.x1, 0.0) * max(second.y2 - second.y1, 0.0)
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _box_smaller_overlap(first: Box, second: Box) -> float:
    x1 = max(first.x1, second.x1)
    y1 = max(first.y1, second.y1)
    x2 = min(first.x2, second.x2)
    y2 = min(first.y2, second.y2)
    intersection = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    first_area = max(first.x2 - first.x1, 0.0) * max(first.y2 - first.y1, 0.0)
    second_area = max(second.x2 - second.x1, 0.0) * max(second.y2 - second.y1, 0.0)
    smaller_area = min(first_area, second_area)
    return intersection / smaller_area if smaller_area > 0 else 0.0


def deduplicate_detections(
    detections: tuple[PlayerDetection, ...],
    *,
    iou_threshold: float = 0.7,
) -> tuple[PlayerDetection, ...]:
    indexed = sorted(enumerate(detections), key=lambda item: item[1].confidence, reverse=True)
    kept: list[tuple[int, PlayerDetection]] = []
    for index, detection in indexed:
        if all(
            _box_iou(detection.box, other.box) < iou_threshold
            and not (
                (detection.role != other.role or detection.role != "player")
                and _box_smaller_overlap(detection.box, other.box) >= 0.55
            )
            for _, other in kept
        ):
            kept.append((index, detection))
    return tuple(detection for _, detection in sorted(kept))


def _collect_color_samples(
    video_path: Path,
    tracking: TrackingInput,
    selected: dict[int, tuple[TrackObservation, ...]],
) -> dict[int, list[LabColor]]:
    requested: dict[int, list[TrackObservation]] = defaultdict(list)
    for observations in selected.values():
        for observation in observations:
            requested[observation.frame_index].append(observation)
    if not requested:
        return {}

    capture = cv2.VideoCapture(str(video_path))
    _validate_video(capture, video_path, tracking)
    samples: dict[int, list[LabColor]] = defaultdict(list)
    try:
        for frame_index in range(max(requested) + 1):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"vídeo terminou antes do frame {frame_index}")
            typed_frame = cast(Frame, frame)
            for observation in requested.get(frame_index, ()):
                color = extract_shirt_color(typed_frame, observation.detection.box)
                if color is not None:
                    track_id = observation.detection.track_id
                    if track_id is not None:
                        samples[track_id].append(color)
    finally:
        capture.release()
    return dict(samples)


def _render_video(
    video_path: Path,
    output_path: Path,
    tracking: TrackingInput,
    stable_roles: dict[int, Role],
    teams: dict[int, Team],
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    _validate_video(capture, video_path, tracking)
    temporary_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    writer = cv2.VideoWriter(
        str(temporary_path),
        cv2.VideoWriter.fourcc(*"mp4v"),
        tracking.fps,
        (tracking.width, tracking.height),
    )
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"não foi possível criar o vídeo: {output_path}")

    try:
        for frame_record in tracking.frames:
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"vídeo terminou antes do frame {frame_record.frame_index}")
            detections: list[PlayerDetection] = []
            for detection in frame_record.players:
                track_id = detection.track_id
                stable_role = (
                    stable_roles.get(track_id, detection.role)
                    if track_id is not None
                    else detection.role
                )
                detections.append(replace(detection, role=stable_role))
            writer.write(
                annotate_frame(
                    cast(Frame, frame),
                    deduplicate_detections(tuple(detections)),
                    teams=teams,
                )
            )
    finally:
        capture.release()
        writer.release()
    temporary_path.replace(output_path)


def process_identification(
    video_path: Path,
    tracking_path: Path,
    output_dir: Path,
    *,
    max_samples: int = 5,
    minimum_samples: int = 2,
) -> IdentificationResult:
    if not video_path.is_file():
        raise FileNotFoundError(f"vídeo não encontrado: {video_path}")
    if max_samples <= 0 or minimum_samples <= 0 or minimum_samples > max_samples:
        raise ValueError("limites de amostragem inválidos")

    tracking = load_tracking(tracking_path)
    if video_path.stem != tracking.clip_id:
        raise ValueError("o vídeo não corresponde ao clip.id do tracking")
    observations = _group_observations(tracking)
    stable = {track_id: stabilize_role(items) for track_id, items in observations.items()}
    stable_roles: dict[int, Role] = {track_id: role for track_id, (role, _) in stable.items()}
    role_evidence: dict[int, set[Role]] = {
        track_id: {
            role
            for observation in items
            if (role := as_role(observation.detection.role)) is not None
        }
        for track_id, items in observations.items()
    }
    selected = {
        track_id: select_sample_observations(items, limit=max_samples, role=None)
        for track_id, items in observations.items()
    }
    samples = _collect_color_samples(video_path, tracking, selected)
    representative_colors = {
        track_id: color
        for track_id, track_samples in samples.items()
        if len(track_samples) >= minimum_samples
        and (color := representative_shirt_color(track_samples)) is not None
    }
    team_seed_colors = {
        track_id: color
        for track_id, color in representative_colors.items()
        if stable_roles[track_id] == "player"
    }
    clustered = cluster_teams(team_seed_colors)
    seed_assignments, centroids = clustered if clustered is not None else ({}, {})
    resolved_roles, assignments, role_sources = resolve_roles_and_teams(
        stable_roles,
        representative_colors,
        seed_assignments,
        centroids,
        role_evidence=role_evidence,
        color_samples=samples,
    )

    tracks: list[TrackIdentity] = []
    for track_id in sorted(observations):
        stable_role, stable_confidence = stable[track_id]
        resolved_role = resolved_roles[track_id]
        sample_count = len(samples.get(track_id, ()))
        if resolved_role != "player":
            status: IdentityStatus = "not_applicable"
            reason = "role_not_player"
        elif sample_count < minimum_samples:
            status = "unavailable"
            reason = "insufficient_color_samples"
        elif track_id not in assignments:
            status = "unavailable"
            reason = "insufficient_team_contrast"
        else:
            status = "available"
            reason = None
        tracks.append(
            {
                "track_id": track_id,
                "stable_role": stable_role,
                "stable_role_confidence": stable_confidence,
                "resolved_role": resolved_role,
                "role_source": role_sources[track_id],
                "team": assignments.get(track_id),
                "sample_count": sample_count,
                "status": status,
                "reason": reason,
            }
        )

    result: IdentificationResult = {
        "schema_version": IDENTIFICATION_SCHEMA_VERSION,
        "clip": {
            "id": tracking.clip_id,
            "fps": tracking.fps,
            "width": tracking.width,
            "height": tracking.height,
            "frame_count": tracking.frame_count,
        },
        "source": {"schema_version": SCHEMA_VERSION},
        "config": {
            "color_space": "Lab",
            "max_samples_per_track": max_samples,
            "minimum_samples_per_track": minimum_samples,
            "team_count": 2,
        },
        "teams": {team: {"centroid_lab": list(color)} for team, color in centroids.items()},
        "tracks": tracks,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _render_video(
        video_path,
        output_dir / "identified.mp4",
        tracking,
        resolved_roles,
        assignments,
    )
    temporary_json = output_dir / "identities.json.tmp"
    temporary_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_json.replace(output_dir / "identities.json")
    return result
