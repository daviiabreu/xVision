"""Comportamento publico da especie field."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from edge.artifacts import ArtifactMissing
from edge.modules.base import FrameBatch
from edge.modules.field.geometry import PITCH_105X68, ImageKeypoint, pitch_to_image
from edge.modules.field.module import PREPROCESS, FieldConfigError, YoloField
from edge.modules.field.yolo import UltralyticsFieldEstimator
from edge.modules.records import check_records


def _batch(indices: tuple[int, ...] = (0, 5)) -> FrameBatch:
    return FrameBatch(
        indices=indices,
        tensor=np.zeros((len(indices), PREPROCESS.height, PREPROCESS.width, 3), dtype=np.uint8),
        preprocess=PREPROCESS,
        source_size=(1920, 1080),
    )


def test_no_estimator_returns_explicit_unavailability_per_frame() -> None:
    module = YoloField()
    module.load()

    output = module.infer(_batch(), {})

    assert output.frames == (0, 5)
    assert len(output.records) == 2
    assert all(records[0]["source"] == "unavailable" for records in output.records)
    assert all(records[0]["reason"] == "no-field-instance" for records in output.records)
    check_records(output)


class _Estimator:
    def estimate(self, frame: object) -> tuple[ImageKeypoint, ...]:
        selected = (0, 2, 9, 11, 12, 16, 18, 25)
        return tuple(
            ImageKeypoint(
                point.name,
                (
                    (point.pitch_xy[0] * 10.0 + 960.0) * 0.5,
                    (point.pitch_xy[1] * -10.0 + 540.0) * 0.5 + 2.0,
                ),
                0.9,
            )
            for point in (PITCH_105X68.keypoints[index] for index in selected)
        )


class _EstimatorWithOutlier(_Estimator):
    def estimate(self, frame: object) -> tuple[ImageKeypoint, ...]:
        points = list(super().estimate(frame))
        last = points[-1]
        points[-1] = ImageKeypoint(last.name, (50.0, 50.0), last.confidence)
        return tuple(points)


def test_detected_geometry_is_reported_in_source_pixels() -> None:
    module = YoloField(_Estimator())
    module.load()

    output = module.infer(_batch((0,)), {})

    record = output.records[0][0]
    assert record["source"] == "detected"
    assert record["observed_keypoint_count"] == 8
    assert record["inlier_count"] == 8
    assert pitch_to_image(record["homography"], (12.0, -8.0)) == pytest.approx(  # type: ignore[arg-type]
        (1080.0, 620.0), abs=0.01
    )


def test_white_lines_refine_one_homography_in_source_pixels() -> None:
    batch = _batch((0,))
    batch.tensor[:] = (30, 120, 30)
    # Independent synthetic pitch, shifted 3 tensor pixels from estimator predictions.
    for start, end in ((0, 16), (9, 25), (0, 9), (16, 25), (11, 12), (1, 2), (2, 4), (4, 3)):
        pixels = [
            tuple(round(v) for v in (p.pitch_xy[0] * 5 + 483, p.pitch_xy[1] * -5 + 272))
            for p in (PITCH_105X68.keypoints[start], PITCH_105X68.keypoints[end])
        ]
        cv2.line(batch.tensor[0], pixels[0], pixels[1], (245, 245, 245), 3)
    module = YoloField(_Estimator())
    module.configure({"line_refinement": True})
    module.load()

    record = module.infer(batch, {}).records[0][0]

    assert record["refinement"] == "accepted"
    assert record["line_support_count"] >= 4  # type: ignore[operator]
    assert pitch_to_image(record["homography"], (12.0, -8.0)) == pytest.approx(  # type: ignore[arg-type]
        (1086.0, 620.0), abs=1.5
    )
    assert record["reprojection_error_px"] > 3  # type: ignore[operator]
    check_records(module.infer(batch, {}))


def test_configuration_changes_the_keypoint_confidence_gate() -> None:
    module = YoloField(_Estimator())
    module.configure({"keypoint_confidence": 0.95})
    module.load()

    record = module.infer(_batch((0,)), {}).records[0][0]

    assert record["source"] == "unavailable"
    assert record["reason"] == "insufficient-keypoints"


def test_refinement_configuration_changes_the_perception_cache_fingerprint() -> None:
    module = YoloField(_Estimator())
    original = module.fingerprint()

    module.configure({"line_refinement": True})

    assert module.fingerprint() != original


@pytest.mark.parametrize("value", ["true", 1, None])
def test_refinement_configuration_requires_a_boolean(value: object) -> None:
    with pytest.raises(FieldConfigError, match="booleano"):
        YoloField().configure({"line_refinement": value})


def test_white_refinement_never_creates_a_field_without_an_estimator() -> None:
    module = YoloField()
    module.configure({"line_refinement": True})
    module.load()

    record = module.infer(_batch((0,)), {}).records[0][0]

    assert record["source"] == "unavailable"
    assert record["refinement"] == "unavailable"
    assert record["homography"] is None


def test_configuration_can_select_the_non_robust_baseline() -> None:
    module = YoloField(_EstimatorWithOutlier())
    module.configure({"method": "least-squares", "max_reprojection_p95_px": 2.0})
    module.load()

    record = module.infer(_batch((0,)), {}).records[0][0]

    assert record["source"] == "unavailable"
    assert record["reason"] == "high-reprojection-error"


@pytest.mark.parametrize("name", ["threshold_diagonal_fraction", "max_reprojection_p95_px"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_calibration_options_are_rejected(name: str, value: float) -> None:
    with pytest.raises(FieldConfigError, match=name):
        YoloField().configure({name: value})


def test_declared_missing_artifact_fails_with_its_manifest_path(tmp_path: Path) -> None:
    module = YoloField()
    module.spec = replace(module.spec, artifact=str(tmp_path / "field-missing.toml"))

    with pytest.raises(ArtifactMissing, match="field-missing.toml"):
        module.load()


def test_inference_options_reach_the_real_estimator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    def build(
        weights: str,
        *,
        imgsz: int,
        conf: float,
        device: str,
        quantize: str,
        engine: str,
        engine_onnx_sha256: str,
    ) -> _Estimator:
        calls.update(
            weights=weights,
            imgsz=imgsz,
            conf=conf,
            device=device,
            quantize=quantize,
            engine=engine,
            engine_onnx_sha256=engine_onnx_sha256,
        )
        return _Estimator()

    def entrypoint(artifact: str | None) -> Path:
        return Path("field.pt")

    monkeypatch.setattr("edge.modules.field.module.entrypoint_of", entrypoint)
    monkeypatch.setattr(UltralyticsFieldEstimator, "from_weights", build)
    module = YoloField()
    module.configure({"imgsz": 640, "conf": 0.4, "device": "cpu"})

    module.load()

    assert calls == {
        "weights": "field.pt",
        "imgsz": 640,
        "conf": 0.4,
        "device": "cpu",
        "quantize": "fp32",
        "engine": "",
        "engine_onnx_sha256": "",
    }


def test_plano_declarado_chega_ao_estimador(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sem isto, `engine` no TOML seria lido e descartado, e a execucao reportaria o tempo do peso.

    E o modo de falha caro desta especie: o campo ganha 4,2x com plano, entao uma medicao que
    silenciosamente rodou em PyTorch pareceria "TensorRT nao ajudou aqui".
    """
    calls: dict[str, object] = {}

    def build(weights: str, **kwargs: object) -> _Estimator:
        calls.update(kwargs)
        return _Estimator()

    def entrypoint(artifact: str | None) -> Path:
        return Path("field.pt")

    monkeypatch.setattr("edge.modules.field.module.entrypoint_of", entrypoint)
    monkeypatch.setattr(UltralyticsFieldEstimator, "from_weights", build)
    module = YoloField()
    module.configure({"engine": "/planos/field.engine", "engine_onnx_sha256": "a" * 64})

    module.load()

    assert calls["engine"] == "/planos/field.engine"
    assert calls["engine_onnx_sha256"] == "a" * 64


def test_plano_com_texto_vazio_e_recusado() -> None:
    module = YoloField()
    with pytest.raises(FieldConfigError, match="engine espera texto"):
        module.configure({"engine": ""})


def test_the_declared_precision_reaches_the_real_estimator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    def build(weights: str, **kwargs: object) -> _Estimator:
        calls.update(weights=weights, **kwargs)
        return _Estimator()

    def entrypoint(artifact: str | None) -> Path:
        return Path("field.pt")

    monkeypatch.setattr("edge.modules.field.module.entrypoint_of", entrypoint)
    monkeypatch.setattr(UltralyticsFieldEstimator, "from_weights", build)
    module = YoloField()
    module.configure({"quantize": "fp16"})

    module.load()

    assert calls["quantize"] == "fp16"


def test_precision_changes_the_perception_cache_fingerprint() -> None:
    """Meia precisao devolve outros numeros, entao a passada guardada da outra nao pode servir."""
    module = YoloField(_Estimator())
    original = module.fingerprint()

    module.configure({"quantize": "fp16"})

    assert module.fingerprint() != original


@pytest.mark.parametrize("value", ["fp8", True, 16, "half"])
def test_an_invalid_precision_is_refused_with_what_is_accepted(value: object) -> None:
    with pytest.raises(FieldConfigError, match="fp32.*fp16"):
        YoloField().configure({"quantize": value})


def test_a_checagem_visivel_e_opcao_e_muda_a_impressao_do_cache() -> None:
    visivel = YoloField()
    visivel.configure({"safe_projection": "visible"})

    assert visivel.fingerprint() != YoloField().fingerprint()
    with pytest.raises(FieldConfigError, match="safe_projection"):
        YoloField().configure({"safe_projection": "campo"})
