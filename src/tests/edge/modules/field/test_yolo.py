"""Adapter do peso de keypoints do campo."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

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


def test_the_declared_precision_reaches_predict() -> None:
    """Alavanca que ninguem consegue puxar nao e alavanca: o valor tem de chegar na chamada."""

    class _Recording(_Model):
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def predict(self, **kwargs: object) -> list[object]:
            self.calls.append(dict(kwargs))
            return super().predict(**kwargs)

    model = _Recording()
    estimator = UltralyticsFieldEstimator(
        model, imgsz=960, conf=0.25, device="cuda", quantize="fp16"
    )

    estimator.estimate(np.zeros((544, 960, 3), dtype=np.uint8))

    assert model.calls[0]["quantize"] == "fp16"


def test_without_a_declaration_predict_gets_no_quantize() -> None:
    """O padrao continua fp32, e a prova e uma chamada sem nada acrescentado a ela."""

    class _Recording(_Model):
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def predict(self, **kwargs: object) -> list[object]:
            self.calls.append(dict(kwargs))
            return super().predict(**kwargs)

    model = _Recording()

    UltralyticsFieldEstimator(model, imgsz=960, conf=0.25, device="cuda").estimate(
        np.zeros((544, 960, 3), dtype=np.uint8)
    )

    assert "quantize" not in model.calls[0]
    assert "half" not in model.calls[0], "half esta depreciado no 8.4 e mediu 62,4 ms"


def test_half_precision_reaches_the_backend_loader(peso_de_mentira: Callable[..., Any]) -> None:
    """Em producao o campo passa pelo backend fixo, e nunca pelo predict.

    `from_weights` sempre monta o `PyTorchBackend`, entao um `quantize` que so alcancasse o predict
    seria uma alavanca ligada apenas no caminho de teste. O binding do artefato continua float32
    dos dois lados; quem vira meia precisao sao o peso e o tensor de entrada, dentro do loader.
    """
    from pathlib import Path

    from edge.modules.field.yolo import ultralytics_loader

    weight = peso_de_mentira("edge.modules.field.yolo", task="pose", yaml={"kpt_shape": [29, 3]})

    infer = ultralytics_loader("fp16")(Path("field.pt"), "cuda")
    infer({"images": np.zeros((1, 3, 544, 960), dtype=np.float32)})

    assert weight.meio_peso is True
    assert weight.entradas_meias == [True]


def test_without_a_declaration_the_backend_loader_stays_fp32(
    peso_de_mentira: Callable[..., Any],
) -> None:
    from pathlib import Path

    from edge.modules.field.yolo import ultralytics_loader

    weight = peso_de_mentira("edge.modules.field.yolo", task="pose", yaml={"kpt_shape": [29, 3]})

    infer = ultralytics_loader()(Path("field.pt"), "cuda")
    output = infer({"images": np.zeros((1, 3, 544, 960), dtype=np.float32)})

    assert weight.meio_peso is False
    assert weight.entradas_meias == [False]
    assert output["predictions"].dtype.name == "float32"
