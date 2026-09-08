"""Enderecos e validacao da API para o render compartilhado da pipeline."""

from pathlib import Path

from contracts import Perception
from edge import video
from edge.paths import var_dir
from edge.video import SECAO as SECAO
from edge.video import SUFIXO as SUFIXO
from edge.video import RenderConfig as RenderConfig
from edge.video import RenderUnavailable as RenderUnavailable
from edge.video import read_config as read_config


def renders_dir() -> Path:
    return var_dir() / "api" / "renders"


def render_path(video_id: int) -> Path:
    return renders_dir() / f"{video_id}.mp4"


def render(
    video_id: int, clipe: Path, config: RenderConfig, *, perception: Perception | None = None
) -> Path:
    if perception is not None and perception.clip_id != f"video-{video_id}":
        raise RenderUnavailable("percepcao pertence a outro video")
    return video.render_video(clipe, render_path(video_id), config, perception=perception)
