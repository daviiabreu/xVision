"""Adapter do peso de keypoints do campo."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from edge.modules.field.yolo import (
    FieldModelIncompatible,
    UltralyticsFieldEstimator,
    check_field_model,
)


class _Model:
    def predict(self, **kwargs: object) -> list[object]:
        xy = np.zeros((2, 29, 2), dtype=np.float32)
        xy[0, :, :] = (10.0, 20.0)
        xy[1, :, :] = (30.0, 40.0)
        confidence = np.full((2, 29), 0.8, dtype=np.float32)
        return [
            SimpleNamespace(
                keypoints=SimpleNamespace(xy=xy, conf=confidence),
                boxes=SimpleNamespace(
                    xyxy=np.asarray(((0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 30.0, 20.0)))
                ),
            )
        ]


def test_largest_field_instance_is_mapped_to_the_29_semantic_names() -> None:
    estimator = UltralyticsFieldEstimator(_Model(), imgsz=960, conf=0.25, device="cpu")

    points = estimator.estimate(np.zeros((544, 960, 3), dtype=np.uint8))

    assert points is not None
    assert len(points) == 29
    assert points[0].name == "sideline_top_left"
    assert points[0].image_xy == (30.0, 40.0)
    assert points[-1].name == "center_circle_right"


def test_weight_must_declare_pose_with_29_keypoints() -> None:
    model = SimpleNamespace(task="detect", model=SimpleNamespace(yaml={}))

    with pytest.raises(FieldModelIncompatible, match="task=pose"):
        check_field_model(model)
