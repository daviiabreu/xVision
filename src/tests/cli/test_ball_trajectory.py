"""Trajectory evaluation must isolate frame caches across independent clips."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from typer.testing import CliRunner, Result

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


class OffsetBall(CachedBall):
    """A mesma bola deslocada em x, para uma configuracao errar onde a outra acerta."""

    def __init__(self, offset: float) -> None:
        self.offset = offset
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def infer(self, batch: FrameBatch, ctx: ModuleInputs) -> ModuleOutput:
        center = float(batch.tensor[0, 0, 0, 0]) + self.offset
        box = (center - 2, 28.0, center + 2, 32.0)
        return ModuleOutput(
            kind="ball", frames=batch.indices, records=(({"box": box, "confidence": 0.9},),)
        )


def _digest(path: Path) -> str:
    return "a" * 64


def _label(clip_id: str, center: float, diameter: float) -> BallTrajectoryLabel:
    return BallTrajectoryLabel(
        clip_id=clip_id,
        match_id=clip_id,
        video_sha256="a" * 64,
        frame_index=0,
        timestamp_s=0.0,
        center_xy=(center, 30.0),
        apparent_diameter_px=diameter,
        visibility="visible",
        trajectory_segment=0,
        is_keyframe=True,
        motion_blur=False,
        camera_cut=False,
        annotator="synthetic",
        reviewed_by=None,
        quality_flags=(),
    )


class Trajectory:
    """Dois clipes sinteticos com a bola em x = 20 e x = 40, e o deslocamento lido do TOML."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.tmp_path = tmp_path
        self.diameter = 4.0
        self.balls: list[OffsetBall] = []
        records: list[ShotRecord] = []
        for center in (20, 40):
            clip_id = f"synthetic-{center}"
            path = tmp_path / f"{clip_id}.txt"
            path.write_text(str(center))
            records.append(ShotRecord(clip_id, clip_id, path, True, 0, {}))

        def probe(path: Path, *, clip_id: str) -> Clip:
            return Clip(clip_id, str(path), 30.0, 1, 64, 64)

        def parsed_config(path: Path) -> dict[str, object]:
            return {"offset": float(path.read_text())}

        def modules(config: dict[str, object]) -> dict[str, OffsetBall]:
            ball = OffsetBall(cast("float", config["offset"]))
            self.balls.append(ball)
            return {"ball": ball}

        def read_labels(path: Path) -> list[BallTrajectoryLabel]:
            return [_label(f"synthetic-{c}", float(c), self.diameter) for c in (20, 40)]

        monkeypatch.setattr("cli.load_catalog", lambda: tuple(records))
        monkeypatch.setattr("cli.read_config", parsed_config)
        monkeypatch.setattr("cli.load_modules", modules)
        monkeypatch.setattr("cli.VideoSource", SyntheticSource)
        monkeypatch.setattr("cli.probe", probe)
        monkeypatch.setattr("cli.sha256_of", _digest)
        monkeypatch.setattr("mlops.ball_trajectory.read_ball_trajectory_labels", read_labels)

    def run(self, offset: float, saida: str, *extra: str) -> Result:
        config = self.tmp_path / f"offset-{offset:g}.toml"
        config.write_text(str(offset))
        arguments = ["eval-ball-trajectory", "--labels", str(self.tmp_path / "labels.jsonl")]
        arguments += ["--config", str(config), "--saida", str(self.tmp_path / saida), *extra]
        return CliRunner().invoke(app, arguments)


def test_saved_run_keeps_the_aggregate_and_adds_each_visible_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trajectory = Trajectory(tmp_path, monkeypatch)

    result = trajectory.run(0.0, "run.json")

    assert result.exit_code == 0, result.output
    saved = json.loads((tmp_path / "run.json").read_text())
    printed = json.loads(result.output[: result.output.rindex("}") + 1])
    assert {k: v for k, v in saved.items() if k in printed} == printed
    assert set(saved) - set(printed) == {"frame_outcomes", "label_set_sha256"}
    assert saved["frame_outcomes"] == {"synthetic-20": {"0": "hit"}, "synthetic-40": {"0": "hit"}}


def test_base_reports_frames_gained_and_lost_with_the_sign_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trajectory = Trajectory(tmp_path, monkeypatch)
    assert trajectory.run(10.0, "base.json").exit_code == 0

    result = trajectory.run(0.0, "new.json", "--base", str(tmp_path / "base.json"))

    assert result.exit_code == 0, result.output
    assert "PCK 0.0000 -> 1.0000" in result.output
    assert "ganhou 2, perdeu 0, igual 0, em 2 quadros visiveis" in result.output
    paired = json.loads((tmp_path / "new.json").read_text())["paired"]
    assert (paired["gains"], paired["losses"], paired["unchanged"]) == (2, 0, 0)
    assert paired["p_value"] == pytest.approx(0.5)


def test_base_measured_on_other_labels_is_refused_before_loading_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trajectory = Trajectory(tmp_path, monkeypatch)
    assert trajectory.run(0.0, "base.json").exit_code == 0
    trajectory.diameter = 5.0

    result = trajectory.run(0.0, "new.json", "--base", str(tmp_path / "base.json"))

    assert result.exit_code == 1
    assert "outro conjunto de rotulos" in result.output
    assert trajectory.balls[-1].loaded is False
    assert not (tmp_path / "new.json").exists()


def test_base_without_per_frame_outcomes_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trajectory = Trajectory(tmp_path, monkeypatch)
    (tmp_path / "old.json").write_text(json.dumps({"pck_diameter": 1.0}))

    result = trajectory.run(0.0, "new.json", "--base", str(tmp_path / "old.json"))

    assert result.exit_code == 1
    assert "desfecho por quadro" in result.output


@pytest.mark.parametrize(
    "document", [[], {"label_set_sha256": "f" * 64, "frame_outcomes": {"c": []}}]
)
def test_malformed_base_is_reported_without_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, document: object
) -> None:
    trajectory = Trajectory(tmp_path, monkeypatch)
    path = tmp_path / "base.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    result = trajectory.run(0.0, "new.json", "--base", str(path))

    assert result.exit_code == 1
    assert "invalido" in result.output
    assert isinstance(result.exception, SystemExit)
