from __future__ import annotations

from pathlib import Path

import cv2

from contracts import Box, FramePerception, Perception, PlayerDetection
from edge.participants.video import render_perception
from edge.testing.make_clip import make_synthetic_clip


def test_render_central_preserva_duracao_e_nao_deixa_temporario(tmp_path: Path) -> None:
    clip = make_synthetic_clip(tmp_path / "clip.mp4")
    perception = Perception(
        clip_id="c",
        frames=(
            FramePerception(
                frame_index=0,
                timestamp_s=0.0,
                players=(
                    PlayerDetection(
                        box=Box(20.0, 20.0, 60.0, 100.0),
                        confidence=0.9,
                        track_id=7,
                        role="player",
                    ),
                ),
            ),
        ),
    )

    target = render_perception(clip, tmp_path / "out", perception, show_trails=True)

    source_capture = cv2.VideoCapture(str(clip))
    target_capture = cv2.VideoCapture(str(target))
    try:
        assert target_capture.isOpened()
        assert int(target_capture.get(cv2.CAP_PROP_FRAME_COUNT)) == int(
            source_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )
    finally:
        source_capture.release()
        target_capture.release()
    assert not (tmp_path / "out/annotated.tmp.mp4").exists()


def test_render_central_recusa_percepcao_com_quadro_duplicado(tmp_path: Path) -> None:
    clip = make_synthetic_clip(tmp_path / "clip.mp4")
    frame = FramePerception(frame_index=0, timestamp_s=0.0)
    perception = Perception(clip_id="c", frames=(frame, frame))

    try:
        render_perception(clip, tmp_path / "out", perception)
    except ValueError as error:
        assert "duplicados" in str(error)
    else:
        raise AssertionError("percepcao duplicada foi aceita")
