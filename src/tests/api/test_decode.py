"""A API roda o grafo inteiro pelo mesmo TOML do `xvision run`, entao le o mesmo `[decode].mode`."""

from __future__ import annotations

from pathlib import Path

import pytest

from api import execution, store
from edge.frames.provider import Decode, FrameProvider, FrameSource
from edge.testing.make_clip import make_synthetic_clip


def test_the_service_hands_the_declared_mode_to_the_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    modos: list[Decode] = []
    original = execution.FrameProvider

    def espiao(source: FrameSource, *, decode: Decode = "keep") -> FrameProvider:
        modos.append(decode)
        return original(source, decode=decode)

    monkeypatch.setattr(execution, "FrameProvider", espiao)
    config = tmp_path / "pipeline.toml"
    config.write_text(
        Path("config/pipeline.toml").read_text(encoding="utf-8")
        + '\n[decode]\nmode = "redecode"\n',
        encoding="utf-8",
    )
    video = store.criar("c.mp4", "video/mp4")

    execution.processar_video(video.id, make_synthetic_clip(tmp_path / "c.mp4"), config)

    lido = store.ler(video.id)
    assert lido is not None
    assert lido.estado == store.FINALIZADO, lido.motivo
    assert modos == ["redecode"]
