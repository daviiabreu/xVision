"""Processamento sequencial e retomável de um diretório de clipes."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict, cast

from edge.participants.video import SCHEMA_VERSION, FrameTracker, VideoResult, process_video


class BatchItem(TypedDict, total=False):
    clip_id: str
    status: str
    frame_count: int
    detection_count: int
    processing_fps: float
    error: str


class BatchSummary(TypedDict):
    schema_version: str
    input_count: int
    processed: int
    skipped: int
    failed: int
    clips: list[BatchItem]


VideoProcessor = Callable[[Path, Path, FrameTracker, bool], VideoResult]


def _write_summary(path: Path, summary: BatchSummary) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _is_complete(output_dir: Path) -> bool:
    document_path = output_dir / "detections.json"
    if not document_path.is_file() or not (output_dir / "annotated.mp4").is_file():
        return False
    try:
        document = json.loads(document_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return document.get("schema_version") == SCHEMA_VERSION


def process_batch(
    input_dir: Path,
    output_dir: Path,
    tracker: FrameTracker,
    *,
    overwrite: bool = False,
    show_trails: bool = False,
    processor: VideoProcessor = process_video,
) -> BatchSummary:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"diretório de vídeos não encontrado: {input_dir}")

    videos = sorted(input_dir.glob("*.mp4"))
    if not videos:
        raise ValueError(f"nenhum vídeo MP4 encontrado em: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary: BatchSummary = {
        "schema_version": "player-tracking-batch.v0",
        "input_count": len(videos),
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "clips": [],
    }

    for video in videos:
        clip_output = output_dir / video.stem
        complete = _is_complete(clip_output)
        if complete and not overwrite:
            summary["skipped"] += 1
            summary["clips"].append({"clip_id": video.stem, "status": "skipped"})
            continue

        try:
            result = processor(video, clip_output, tracker, show_trails)
            detection_count = sum(
                len(cast(list[object], frame["players"])) for frame in result["frames"]
            )
            summary["processed"] += 1
            summary["clips"].append(
                {
                    "clip_id": video.stem,
                    "status": "processed",
                    "frame_count": result["clip"]["frame_count"],
                    "detection_count": detection_count,
                    "processing_fps": result["metrics"]["processing_fps"],
                }
            )
        except Exception as error:
            summary["failed"] += 1
            summary["clips"].append(
                {"clip_id": video.stem, "status": "failed", "error": str(error)}
            )

        _write_summary(output_dir / "summary.json", summary)

    _write_summary(output_dir / "summary.json", summary)
    return summary
