from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
from field_keypoints import detect_keypoints
from homography import (
    find_transformation,
    project_players,
    select_half,
    smooth_positions,
)

from edge.participants.identification import TrackingInput, load_tracking, process_identification
from edge.participants.identity import Team
from edge.participants.marking import (
    MarkingPair,
    candidates_from_frame,
    pair_markings,
    teams_from_identities,
)
from edge.participants.types import FramePerception, frame_to_dict
from edge.participants.video import SCHEMA_VERSION, FrameTracker

MAX_FRAMES_WITHOUT_CALIBRATION = 15
MINIMUM_POINTS_TO_START = 7
MINIMUM_POINTS_TO_UPDATE = 6


@dataclass(frozen=True)
class FieldFrame:
    keypoints: Any
    confidence: Any
    inliers: Any
    matrix: Any


@dataclass(frozen=True)
class TopViewFrame:
    frame_index: int
    field: FieldFrame
    positions: list[tuple[Any, int]]
    mirror: bool | None
    markings: tuple[MarkingPair, ...]


@dataclass(frozen=True)
class PipelineResult:
    tracking: TrackingInput
    teams: dict[int, Team]
    frames: tuple[TopViewFrame, ...]
    timings: dict[str, float]


def log(message: str) -> None:
    print(f"[TOP VIEW] {message}", flush=True)


def timed(function, *args):
    started = perf_counter()
    return function(*args), perf_counter() - started


class HomographyTracker:
    def __init__(self) -> None:
        self.matrix = None
        self.missing = 0

    def process(self, keypoints: Any, confidence: Any) -> FieldFrame:
        current, inliers = find_transformation(keypoints, confidence)
        count = 0 if inliers is None else int(inliers.sum())
        minimum = MINIMUM_POINTS_TO_START if self.matrix is None else MINIMUM_POINTS_TO_UPDATE
        if current is not None and count >= minimum:
            self.matrix = current
            self.missing = 0
        else:
            self.missing += 1
            if self.missing > MAX_FRAMES_WITHOUT_CALIBRATION:
                self.matrix = None
        return FieldFrame(keypoints, confidence, inliers, self.matrix)


def run_pipeline(
    tracker: FrameTracker,
    video_path: Path,
    output_dir: Path,
    keypoint_model: Any,
    device: str,
) -> PipelineResult:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Nao foi possivel abrir: {video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0 or width <= 0 or height <= 0:
            raise ValueError(f"Metadados de video invalidos: {video_path}")
        frames, fields = [], []
        players_seconds = 0.0
        keypoints_seconds = 0.0
        homography_seconds = 0.0
        homography = HomographyTracker()
        tracker.start_clip()
        started = perf_counter()
        log("ETAPA 1/3 — inferência paralela iniciada (jogadores + pontos do campo)")
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="top-view") as pool:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                index = len(frames)
                players = pool.submit(timed, tracker.track, frame.copy())
                field = pool.submit(timed, detect_keypoints, keypoint_model, frame.copy(), device)
                player_detections, player_elapsed = players.result()
                field_detection, field_elapsed = field.result()
                field_frame, homography_elapsed = timed(homography.process, *field_detection)
                players_seconds += player_elapsed
                keypoints_seconds += field_elapsed
                homography_seconds += homography_elapsed
                frames.append(FramePerception(index, index / fps, player_detections))
                fields.append(field_frame)
                if len(frames) % 30 == 0:
                    wall = perf_counter() - started
                    log(
                        f"ETAPA 1/3 — {len(frames)} frames | parede: {wall:.2f}s | "
                        f"jogadores: {players_seconds:.2f}s | pontos: {keypoints_seconds:.2f}s | "
                        f"homografia: {homography_seconds:.2f}s"
                    )
        inference_seconds = perf_counter() - started
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"O video nao possui frames decodificaveis: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": SCHEMA_VERSION,
        "clip": {
            "id": video_path.stem,
            "fps": fps,
            "width": width,
            "height": height,
            "frame_count": len(frames),
        },
        "model": {
            "name": "football-player-detection",
            "image_size": tracker.image_size,
            "confidence_threshold": tracker.confidence_threshold,
            "device": tracker.device,
        },
        "metrics": {
            "elapsed_seconds": inference_seconds,
            "processing_fps": len(frames) / inference_seconds,
            "players_model_seconds": players_seconds,
            "field_keypoints_model_seconds": keypoints_seconds,
            "homography_seconds": homography_seconds,
        },
        "frames": [frame_to_dict(frame) for frame in frames],
    }
    tracking_path = output_dir / "detections.json"
    temporary = tracking_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    temporary.replace(tracking_path)
    log(
        f"ETAPA 1/3 concluída — {len(frames)} frames em {inference_seconds:.2f}s | "
        f"modelo de jogadores: {players_seconds:.2f}s | "
        f"modelo de pontos: {keypoints_seconds:.2f}s | "
        f"cálculo da homografia: {homography_seconds:.2f}s"
    )
    log("ETAPA 2/3 — identificação de equipes iniciada")
    identity_started = perf_counter()
    identity = process_identification(
        video_path,
        tracking_path,
        output_dir / "identification",
    )
    identity_seconds = perf_counter() - identity_started
    (output_dir / "identification" / "identified.mp4").unlink(missing_ok=True)
    teams = teams_from_identities(identity)
    tracking = load_tracking(tracking_path)
    log(f"ETAPA 2/3 concluída — identificação: {identity_seconds:.2f}s")
    log("ETAPA 3/3 — projeção e marcação iniciadas")
    projection_started = perf_counter()
    previous, mirror, joined = {}, None, []
    for frame, field in zip(frames, fields, strict=True):
        boxes, ids = [], []
        for player in frame.players:
            if player.track_id is not None:
                box = player.box
                boxes.append([box.x1, box.y1, box.x2, box.y2])
                ids.append(player.track_id)
        positions = smooth_positions(project_players(field.matrix, boxes, ids), previous)
        mirror = select_half(positions, mirror)
        pairs = ()
        if field.matrix is not None:

            def project(point, matrix=field.matrix):
                source = np.array([[point]], dtype=np.float32)
                xy = cv2.perspectiveTransform(source, matrix)[0, 0]
                return float(xy[0]) / 100, float(xy[1]) / 100

            pairs = pair_markings(candidates_from_frame(frame, teams, project=project))
        joined.append(TopViewFrame(frame.frame_index, field, positions, mirror, pairs))
    projection_seconds = perf_counter() - projection_started
    log(f"ETAPA 3/3 concluída — projeção/marcação: {projection_seconds:.2f}s")
    return PipelineResult(
        tracking,
        teams,
        tuple(joined),
        {
            "parallel_inference": inference_seconds,
            "players_model": players_seconds,
            "field_keypoints_model": keypoints_seconds,
            "homography": homography_seconds,
            "team_identification": identity_seconds,
            "projection_marking": projection_seconds,
        },
    )
