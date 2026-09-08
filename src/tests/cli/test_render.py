from pathlib import Path

import pytest
from typer.testing import CliRunner

from cli import app
from contracts import Perception
from edge import video
from edge.testing.make_clip import make_synthetic_clip


def test_catalog_render_is_refused_explicitly() -> None:
    result = CliRunner().invoke(app, ["run", "--all", "--render"])

    assert result.exit_code != 0
    assert "--clip" in result.output


def test_run_honors_the_original_top_view_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XVISION_VAR_DIR", str(tmp_path / "var"))
    clip = make_synthetic_clip(tmp_path / "clip.mp4")
    config = tmp_path / "pipeline.toml"
    source = Path(__file__).parents[2] / "config" / "pipeline.toml"
    config.write_text(
        source.read_text()
        .replace("models/manifests/players/football-player-detection-1.0.0.toml", "p.toml")
        .replace("models/manifests/field/pitch-keypoints-yolo11n-pose-0.1.0.toml", "f.toml")
    )
    calls: list[video.RenderConfig] = []

    def configured_render(
        clip_path: Path, target: Path, selected: video.RenderConfig, *, perception: Perception
    ) -> Path:
        assert clip_path == clip and perception.frames
        calls.append(selected)
        target.parent.mkdir(parents=True)
        target.write_bytes(b"configured top view with ball")
        return target

    monkeypatch.setattr(video, "render_video", configured_render)
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--clip",
            str(clip),
            "--config",
            str(config),
            "--out",
            str(tmp_path / "out"),
            "--render",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [video.RenderConfig(players="p.toml", field="f.toml")]
    assert (tmp_path / "out/clip/annotated.mp4").read_bytes() == b"configured top view with ball"


def test_run_render_requires_the_original_top_view_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XVISION_VAR_DIR", str(tmp_path / "var"))
    clip = make_synthetic_clip(tmp_path / "clip.mp4")
    config = tmp_path / "pipeline.toml"
    source = Path(__file__).parents[2] / "config" / "pipeline.toml"
    text = source.read_text()
    config.write_text(text[text.index("[frames.coarse]") :])
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--clip",
            str(clip),
            "--config",
            str(config),
            "--out",
            str(tmp_path / "out"),
            "--render",
        ],
    )
    assert result.exit_code != 0
    assert "[render.top_view]" in result.output
    assert not (tmp_path / "out/clip/annotated.mp4").exists()
