"""Prova ao vivo: monta um evento completo, valida contra o schema e imprime o JSON."""

from __future__ import annotations

import json

from contracts.events import FeatureValue, PipelineMetadata, ShotEvent, make_shot_id
from contracts.schema import SCHEMA_VERSION
from contracts.validation import validate_shot_event

CLIP_ID = "clip-demo-001"
SHOT_FRAME_INDEX = 187


def _demo_features() -> tuple[FeatureValue, ...]:
    return (
        FeatureValue(
            name="shot_distance_m",
            version="1.0.0",
            status="ok",
            value=16.24,
            confidence=0.91,
            reason=None,
            evidence={
                "frames": [185, 186, 187],
                "shooter_track_id": 9,
                "pitch_xy_m": {"x": -3.4, "y": 15.87},
                "intermediate": {"homography_error_px": 2.1, "used_keypoints": ["big_box_tl"]},
            },
        ),
        FeatureValue(
            name="shot_angle_deg",
            version="1.0.0",
            status="low_confidence",
            value=27.9,
            confidence=0.42,
            reason="erro de reprojeccao de 9.7 px acima do limite de 5.0 px",
            evidence={"frames": [187], "homography_error_px": 9.7},
        ),
        FeatureValue(
            name="gaze_on_ball",
            version="0.1.0",
            status="unavailable",
            value=None,
            confidence=None,
            reason="modulo pose nao rodou: nenhum modelo configurado para a janela fina",
            evidence={"missing_module": "pose"},
        ),
    )


def _demo_pipeline() -> PipelineMetadata:
    return PipelineMetadata(
        versions={
            "players": "stub-players@0.1.0",
            "ball": "stub-ball@0.1.0",
            "field": "stub-field@0.1.0",
            "shot": "stub-shot@0.1.0",
        },
        timings_ms={"decode": 118.4, "coarse": 342.7, "window": 96.2, "features": 11.9},
        host="dev-station",
        started_at="2026-08-18T13:00:00Z",
    )


def build_demo_event() -> ShotEvent:
    return ShotEvent(
        schema_version=SCHEMA_VERSION,
        shot_id=make_shot_id(CLIP_ID, SHOT_FRAME_INDEX),
        clip_id=CLIP_ID,
        match_id="match-2026-08-18-flamengo-palmeiras",
        shot_timestamp_s=7.48,
        features=_demo_features(),
        pipeline=_demo_pipeline(),
    )


def main() -> None:
    payload = build_demo_event().to_dict()
    validate_shot_event(payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
