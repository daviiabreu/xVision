from pathlib import Path

import pytest

from xvision_players_yolo.batch import process_batch
from xvision_players_yolo.detector import Frame
from xvision_players_yolo.types import PlayerDetection
from xvision_players_yolo.video import VideoResult


class FakeTracker:
    image_size = 1280
    confidence_threshold = 0.25
    device = "cpu"

    def start_clip(self) -> None:
        pass

    def track(self, frame: Frame) -> tuple[PlayerDetection, ...]:
        return ()


def make_result(clip_id: str) -> VideoResult:
    return {
        "schema_version": "player-tracking.v1",
        "clip": {"id": clip_id, "fps": 10.0, "width": 64, "height": 48, "frame_count": 2},
        "model": {},
        "metrics": {
            "elapsed_seconds": 1.0,
            "average_inference_ms": 10.0,
            "processing_fps": 2.0,
        },
        "frames": [
            {"frame_index": 0, "timestamp_s": 0.0, "players": [{"role": "player"}]},
            {"frame_index": 1, "timestamp_s": 0.1, "players": []},
        ],
    }


def test_batch_is_ordered_and_continues_after_failure(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    for name in ("c.mp4", "a.mp4", "b.mp4"):
        (input_dir / name).touch()

    calls: list[str] = []

    def processor(source: Path, target: Path, tracker: object, show_trails: bool) -> VideoResult:
        calls.append(source.name)
        assert show_trails is False
        if source.name == "b.mp4":
            raise ValueError("vídeo inválido")
        return make_result(source.stem)

    summary = process_batch(input_dir, output_dir, FakeTracker(), processor=processor)

    assert calls == ["a.mp4", "b.mp4", "c.mp4"]
    assert summary["processed"] == 2
    assert summary["failed"] == 1
    assert summary["clips"][0].get("detection_count") == 1
    assert (output_dir / "summary.json").is_file()


def test_batch_skips_complete_result_by_default(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "a.mp4").touch()
    complete = output_dir / "a"
    complete.mkdir(parents=True)
    (complete / "detections.json").write_text(
        '{"schema_version": "player-tracking.v1"}', encoding="utf-8"
    )
    (complete / "annotated.mp4").touch()

    def unexpected_processor(
        source: Path, target: Path, tracker: object, show_trails: bool
    ) -> VideoResult:
        raise AssertionError("resultado completo não deveria ser reprocessado")

    summary = process_batch(input_dir, output_dir, FakeTracker(), processor=unexpected_processor)

    assert summary["skipped"] == 1
    assert summary["processed"] == 0


def test_batch_reprocesses_result_from_detection_only_schema(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "a.mp4").touch()
    previous = output_dir / "a"
    previous.mkdir(parents=True)
    (previous / "detections.json").write_text(
        '{"schema_version": "player-detection.v0"}', encoding="utf-8"
    )
    (previous / "annotated.mp4").touch()

    def processor(source: Path, target: Path, tracker: object, show_trails: bool) -> VideoResult:
        return make_result(source.stem)

    summary = process_batch(
        input_dir,
        output_dir,
        FakeTracker(),
        processor=processor,
    )

    assert summary["processed"] == 1
    assert summary["skipped"] == 0


def test_batch_rejects_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="nenhum vídeo"):
        process_batch(tmp_path, tmp_path / "output", FakeTracker())
