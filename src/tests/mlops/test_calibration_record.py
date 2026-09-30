"""A quantizacao so aceita tensores acompanhados da procedencia completa da extracao."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from edge.artifacts import sha256_of
from mlops.quantize_int8 import CALIBRACAO, CalibracaoError, le_calibracao


def _calibration(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    folder = tmp_path / "calibracao"
    folder.mkdir()
    tensor = folder / "0000.npy"
    np.save(tensor, np.zeros((1, 3, 8, 8), dtype=np.float32))
    report: dict[str, Any] = {
        "especie": "field",
        "config_sha256": "a" * 64,
        "frames": "dense",
        "fps": None,
        "preprocess": {
            "width": 8,
            "height": 8,
            "color": "bgr",
            "letterbox": True,
            "normalize": "none",
            "ops": [],
        },
        "ajuste": {},
        "dobras": [0, 1, 2],
        "semente": 13,
        "quadros_pedidos": 1,
        "quadros": 1,
        "clipes": 1,
        "shape": [1, 3, 8, 8],
        "dtype": "float32",
        "sha256": [sha256_of(tensor)],
        "versoes": {"numpy": "2.2.6", "opencv-python-headless": "4.12.0.88"},
        "extrator_sha256": "b" * 64,
    }
    _write_report(folder, report)
    return folder, report


def _write_report(folder: Path, report: object) -> None:
    (folder / CALIBRACAO).write_text(json.dumps(report), encoding="utf-8")


@pytest.mark.parametrize("text", ["[]", "null", '"texto"', "7", "{"])
def test_invalid_json_report_raises_calibration_error(tmp_path: Path, text: str) -> None:
    folder, _ = _calibration(tmp_path)
    (folder / CALIBRACAO).write_text(text, encoding="utf-8")

    with pytest.raises(CalibracaoError, match="calibracao.json"):
        le_calibracao(folder)


@pytest.mark.parametrize(
    "field_name",
    [
        "especie",
        "config_sha256",
        "frames",
        "fps",
        "preprocess",
        "ajuste",
        "dobras",
        "semente",
        "quadros_pedidos",
        "quadros",
        "clipes",
        "shape",
        "dtype",
        "sha256",
        "versoes",
        "extrator_sha256",
    ],
)
def test_missing_extraction_field_has_insufficient_provenance(
    tmp_path: Path, field_name: str
) -> None:
    folder, report = _calibration(tmp_path)
    del report[field_name]
    _write_report(folder, report)

    with pytest.raises(CalibracaoError, match=field_name):
        le_calibracao(folder)


def test_existing_complete_report_preserves_tensors_and_metadata(
    tmp_path: Path,
) -> None:
    folder, report = _calibration(tmp_path)
    tensor = folder / "0000.npy"
    before = tensor.read_bytes()

    tensors, digests, loaded = le_calibracao(folder)

    assert tensors == (tensor,)
    assert digests == report["sha256"]
    assert loaded == report
    assert tensor.read_bytes() == before


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("especie", "players"),
        ("especie", []),
        ("config_sha256", "sem-digest"),
        ("config_sha256", None),
        ("extrator_sha256", "g" * 64),
        ("frames", ""),
        ("frames", 3),
        ("fps", True),
        ("fps", 0),
        ("fps", float("nan")),
        ("fps", float("inf")),
        ("dtype", "float16"),
        ("shape", [1, 3, 8]),
        ("shape", [True, 3, 8, 8]),
        ("shape", [1, 3, 0, 8]),
        ("versoes", []),
        ("versoes", {"numpy": "2.2.6"}),
        ("versoes", {"numpy": "", "opencv-python-headless": "4.12.0.88"}),
        ("preprocess", []),
        ("preprocess", {}),
        ("ajuste", []),
    ],
)
def test_malformed_metadata_cannot_describe_calibration(
    tmp_path: Path, field_name: str, value: object
) -> None:
    folder, report = _calibration(tmp_path)
    report[field_name] = value
    _write_report(folder, report)

    with pytest.raises(CalibracaoError, match=field_name):
        le_calibracao(folder)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("width", True),
        ("height", -1),
        ("color", "desconhecida"),
        ("letterbox", "true"),
        ("normalize", []),
        ("ops", "clahe:2.0:8"),
        ("ops", [7]),
    ],
)
def test_malformed_preprocess_is_rejected(tmp_path: Path, field_name: str, value: object) -> None:
    folder, report = _calibration(tmp_path)
    report["preprocess"][field_name] = value
    _write_report(folder, report)

    with pytest.raises(CalibracaoError, match="preprocess"):
        le_calibracao(folder)


@pytest.mark.parametrize("folds", [[3, 4], [0, 1, 2, 3], [0], [False, 1, 2], [], "ajuste"])
def test_report_can_only_declare_adjustment_folds(tmp_path: Path, folds: object) -> None:
    folder, report = _calibration(tmp_path)
    report["dobras"] = folds
    _write_report(folder, report)

    with pytest.raises(CalibracaoError, match="dobras"):
        le_calibracao(folder)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("quadros_pedidos", 0),
        ("quadros_pedidos", True),
        ("quadros", 0),
        ("quadros", "1"),
        ("quadros", 2),
        ("clipes", -1),
        ("clipes", 1.0),
        ("clipes", 2),
    ],
)
def test_invalid_or_inconsistent_extraction_count_is_rejected(
    tmp_path: Path, field_name: str, value: object
) -> None:
    folder, report = _calibration(tmp_path)
    report[field_name] = value
    _write_report(folder, report)

    with pytest.raises(CalibracaoError, match=field_name):
        le_calibracao(folder)


def test_field_frame_count_must_match_tensors(tmp_path: Path) -> None:
    folder, report = _calibration(tmp_path)
    report.update(quadros_pedidos=2, quadros=2)
    _write_report(folder, report)

    with pytest.raises(CalibracaoError, match="tensores"):
        le_calibracao(folder)


@pytest.mark.parametrize("overlap", [0, 1])
def test_multiple_ball_dispatches_preserve_valid_frame_count(tmp_path: Path, overlap: int) -> None:
    folder, report = _calibration(tmp_path)
    tensors = [folder / f"{i:04d}.npy" for i in range(2)]
    for tensor in tensors:
        np.save(tensor, np.zeros((2, 3, 8, 8), dtype=np.float32))
    report.update(
        especie="ball",
        shape=[2, 3, 8, 8],
        ajuste={
            "imgsz": 8,
            "ladrilhos": 2,
            "sobreposicao_px": overlap,
            "ladrilhos_por_lote": 2,
            "letterbox": "cv2",
        },
        sha256=[sha256_of(t) for t in tensors],
    )
    _write_report(folder, report)

    assert le_calibracao(folder)[2] == report
    report["ajuste"]["ladrilhos_por_lote"] = 4
    _write_report(folder, report)
    with pytest.raises(CalibracaoError, match="ajuste"):
        le_calibracao(folder)


def test_incomplete_ball_settings_are_rejected(tmp_path: Path) -> None:
    folder, report = _calibration(tmp_path)
    report["especie"] = "ball"
    _write_report(folder, report)

    with pytest.raises(CalibracaoError, match="ajuste"):
        le_calibracao(folder)


@pytest.mark.parametrize(
    ("shape", "dtype"),
    [((1, 3, 8, 9), "float32"), ((1, 3, 8, 8), "float64")],
)
def test_tensor_must_match_report_shape_and_dtype(
    tmp_path: Path, shape: tuple[int, ...], dtype: str
) -> None:
    folder, report = _calibration(tmp_path)
    tensor = folder / "0000.npy"
    np.save(tensor, np.zeros(shape, dtype=dtype))
    report["sha256"] = [sha256_of(tensor)]
    _write_report(folder, report)

    with pytest.raises(CalibracaoError, match="0000.npy"):
        le_calibracao(folder)


def test_invalid_npy_is_rejected_with_matching_digest(tmp_path: Path) -> None:
    folder, report = _calibration(tmp_path)
    tensor = folder / "0000.npy"
    tensor.write_bytes(b"nao e um tensor numpy")
    report["sha256"] = [sha256_of(tensor)]
    _write_report(folder, report)

    with pytest.raises(CalibracaoError, match="0000.npy"):
        le_calibracao(folder)
