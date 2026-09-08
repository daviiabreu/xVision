"""Trajectory evaluation must isolate frame caches across independent clips."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from cli import app
from contracts import Box, Clip
from edge.modules.base import FrameBatch, ModuleInputs, ModuleOutput, ModuleSpec, PreprocessSpec
from edge.modules.inference import reset, shared
from mlops.ball_trajectory import BallTrajectoryLabel
from mlops.catalog import ShotRecord

PREPROCESS = PreprocessSpec(64, 64, "bgr", True, "none")


class CachedBall:
    spec = ModuleSpec(
        kind="ball", name="cached-fixture", version="1", frames="dense", preprocess=PREPROCESS
    )

    def load(self) -> None: ...

    def close(self) -> None: ...

    def infer(self, batch: FrameBatch, ctx: ModuleInputs) -> ModuleOutput:
        center = float(batch.tensor[0, 0, 0, 0])
        detections = shared().get(
            weights="synthetic-weight",
            preprocess=PREPROCESS,
            frame_index=batch.indices[0],
            compute=lambda: ((Box(center - 2, 28, center + 2, 32), 0.9, 0),),
        )
        records = tuple(
            {"box": (box.x1, box.y1, box.x2, box.y2), "confidence": confidence}
            for box, confidence, _ in detections
        )
        return ModuleOutput(kind="ball", frames=batch.indices, records=(records,))


class SyntheticSource:
    fps = 30.0

    def __init__(self, path: Path) -> None:
        self.value = int(path.read_text())

    def read(self, indices: Sequence[int]) -> dict[int, Any]:
        return {index: np.full((64, 64, 3), self.value, dtype=np.uint8) for index in indices}

    def close(self) -> None: ...


def test_evaluation_does_not_reuse_detections_from_another_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records: list[ShotRecord] = []
    labels: list[BallTrajectoryLabel] = []
    for center in (20, 40):
        clip_id = f"synthetic-{center}"
        path = tmp_path / f"{clip_id}.txt"
        path.write_text(str(center))
        records.append(ShotRecord(clip_id, clip_id, path, True, 0, {}))
        labels.append(
            BallTrajectoryLabel(
                clip_id=clip_id,
                match_id=clip_id,
                video_sha256="a" * 64,
                frame_index=0,
                timestamp_s=0.0,
                center_xy=(float(center), 30.0),
                apparent_diameter_px=4.0,
                visibility="visible",
                trajectory_segment=0,
                is_keyframe=True,
                motion_blur=False,
                camera_cut=False,
                annotator="synthetic",
                reviewed_by=None,
                quality_flags=(),
            )
        )

    def probe(path: Path, *, clip_id: str) -> Clip:
        return Clip(clip_id, str(path), 30.0, 1, 64, 64)

    def parsed_config(path: Path) -> dict[str, object]:
        return {}

    def modules(config: object) -> dict[str, CachedBall]:
        return {"ball": CachedBall()}

    def digest(path: Path) -> str:
        return "a" * 64

    def read_labels(path: Path) -> list[BallTrajectoryLabel]:
        return labels

    config = tmp_path / "pipeline.toml"
    config.write_text("")
    report = tmp_path / "metrics.json"
    monkeypatch.setattr("cli.load_catalog", lambda: tuple(records))
    monkeypatch.setattr("cli.read_config", parsed_config)
    monkeypatch.setattr("cli.load_modules", modules)
    monkeypatch.setattr("cli.VideoSource", SyntheticSource)
    monkeypatch.setattr("cli.probe", probe)
    monkeypatch.setattr("cli.sha256_of", digest)
    monkeypatch.setattr("mlops.ball_trajectory.read_ball_trajectory_labels", read_labels)
    reset()
    try:
        result = CliRunner().invoke(
            app,
            [
                "eval-ball-trajectory",
                "--labels",
                str(tmp_path / "labels.jsonl"),
                "--config",
                str(config),
                "--saida",
                str(report),
            ],
        )
        assert result.exit_code == 0, result.output
        metrics = json.loads(report.read_text())
        assert metrics["true_positive"] == 2
        assert metrics["center_error_p95_px"] == 0.0
    finally:
        reset()
