from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np
import pytest

from api.evidence import gerar_evidencias
from contracts import FramePerception, Perception


def test_evidence_uses_the_image_of_the_selected_perception_frame(tmp_path: Path) -> None:
    video = tmp_path / "distinct-frames.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter.fourcc(*"mp4v"), 30, (320, 180))
    assert writer.isOpened()
    for shade in (20, 120, 240):
        writer.write(np.full((180, 320, 3), shade, dtype=np.uint8))
    writer.release()
    perception = Perception(
        clip_id="distinct-frames", frames=(FramePerception(frame_index=0, timestamp_s=0),)
    )

    encoded = gerar_evidencias(video, 2, perception, ["ball"])["ball"]

    decoded = cv2.imdecode(
        np.frombuffer(base64.b64decode(encoded.split(",", 1)[1]), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert decoded is not None
    assert float(decoded[100:150, 100:200].mean()) == pytest.approx(20, abs=10)
