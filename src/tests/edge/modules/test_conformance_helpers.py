"""O detector de abertura de video da suite precisa detectar de verdade."""

from __future__ import annotations

import sys
import types
from pathlib import Path

from edge.modules.conformance import no_video_access


def test_no_video_access_catches_a_python_open(tmp_path: Path) -> None:
    clip = tmp_path / "clipe.mp4"
    clip.write_bytes(b"nao e video, e so o sufixo que importa aqui")

    with no_video_access() as aberturas:
        clip.read_bytes()
    assert [Path(p).name for p in aberturas] == ["clipe.mp4"]


def test_no_video_access_catches_cv2_video_capture(
    monkeypatch: object,
) -> None:
    fake = types.ModuleType("cv2")
    chamadas: list[str] = []
    fake.VideoCapture = lambda path: chamadas.append(str(path))  # type: ignore[attr-defined]
    original = fake.VideoCapture  # type: ignore[attr-defined]
    sys.modules["cv2"] = fake
    try:
        with no_video_access() as aberturas:
            fake.VideoCapture("clipe.mkv")  # type: ignore[attr-defined]
        assert aberturas == ["clipe.mkv"]
        assert chamadas == ["clipe.mkv"]
        assert fake.VideoCapture is original  # type: ignore[attr-defined]
    finally:
        del sys.modules["cv2"]


def test_nothing_is_recorded_outside_the_block(tmp_path: Path) -> None:
    clip = tmp_path / "fora.mp4"
    clip.write_bytes(b"x")
    with no_video_access() as aberturas:
        pass
    clip.read_bytes()
    assert aberturas == []
