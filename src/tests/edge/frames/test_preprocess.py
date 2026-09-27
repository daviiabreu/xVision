"""O pre-processamento declarado na PreprocessSpec, aplicado na ordem fixa do pipeline."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import cv2
import numpy as np
import pytest

from edge.frames.preprocess import Frame, PreprocessError, prepare
from edge.modules.base import PreprocessSpec

BASE = PreprocessSpec(width=64, height=64, color="rgb", letterbox=False, normalize="none")


def spec(**mudancas: Any) -> PreprocessSpec:
    return replace(BASE, **mudancas)


def iguais(a: Frame, b: Frame) -> bool:
    return a.shape == b.shape and bool((a == b).all())


def _quadro() -> Frame:
    return np.random.default_rng(7).integers(0, 256, size=(90, 160, 3), dtype=np.uint8)


@pytest.fixture
def quadro() -> Frame:
    return _quadro()


def test_preprocess_applies_declared_size_and_color(quadro: Frame) -> None:
    pronto = prepare(quadro, spec(width=32, height=16, color="rgb"))

    assert pronto.shape == (16, 32, 3)
    assert iguais(pronto, cv2.resize(cv2.cvtColor(quadro, cv2.COLOR_BGR2RGB), (32, 16)))


def test_gray_keeps_the_channel_axis_that_the_frame_batch_cobra(quadro: Frame) -> None:
    assert prepare(quadro, spec(color="gray")).shape == (64, 64, 1)


def test_letterbox_preserves_aspect_ratio(quadro: Frame) -> None:
    pronto = prepare(quadro, spec(color="bgr", letterbox=True))

    assert pronto.shape == (64, 64, 3)
    assert iguais(pronto[14:50, :, :], cv2.resize(quadro, (64, 36)))
    assert not pronto[:14, :, :].any()
    assert not pronto[50:, :, :].any()


def test_ops_are_applied_in_declared_order(quadro: Frame) -> None:
    clahe = "clahe:2.0:8"
    bilateral = "bilateral:5:50:50"

    primeiro = prepare(quadro, spec(color="bgr", ops=(clahe, bilateral)))
    segundo = prepare(quadro, spec(color="bgr", ops=(bilateral, clahe)))

    assert iguais(primeiro, _bilateral(_clahe(cv2.resize(quadro, (64, 64)))))
    assert not iguais(primeiro, segundo)


def test_unknown_op_is_refused_instead_of_ignored(quadro: Frame) -> None:
    with pytest.raises(PreprocessError, match="sharpen"):
        prepare(quadro, spec(ops=("sharpen:3",)))


def test_op_with_the_wrong_number_of_arguments_is_refused(quadro: Frame) -> None:
    with pytest.raises(PreprocessError, match="clahe"):
        prepare(quadro, spec(ops=("clahe:2.0",)))


def test_normalize_unit_divides_the_byte_by_two_hundred_and_fifty_five(
    quadro: Frame,
) -> None:
    no_tamanho = cv2.resize(cv2.cvtColor(quadro, cv2.COLOR_BGR2RGB), (64, 64))

    pronto = prepare(quadro, spec(normalize="unit"))

    assert pronto.dtype == np.float32
    assert float(np.abs(pronto - no_tamanho.astype(np.float32) / 255.0).max()) == 0.0


def test_normalize_imagenet_uses_the_channel_statistics(quadro: Frame) -> None:
    no_tamanho = cv2.resize(cv2.cvtColor(quadro, cv2.COLOR_BGR2RGB), (64, 64))
    unit = no_tamanho.astype(np.float32) / 255.0
    media = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    desvio = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    pronto = prepare(quadro, spec(normalize="imagenet"))

    assert float(np.abs(pronto - (unit - media) / desvio).max()) < 1e-6


def test_normalize_imagenet_is_refused_outside_rgb_and_bgr(quadro: Frame) -> None:
    with pytest.raises(PreprocessError, match="imagenet"):
        prepare(quadro, spec(color="gray", normalize="imagenet"))


def _clahe(imagem: Frame) -> Frame:
    ycrcb = cv2.cvtColor(imagem, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def _bilateral(imagem: Frame) -> Frame:
    return cv2.bilateralFilter(imagem, 5, 50.0, 50.0)


def test_clahe_on_a_single_channel_frame_keeps_the_channel_axis(quadro: Frame) -> None:
    pronto = prepare(quadro, spec(color="gray", ops=("clahe:2.0:8",)))

    cinza = cv2.resize(cv2.cvtColor(quadro, cv2.COLOR_BGR2GRAY), (64, 64))
    esperado = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(cinza)
    assert pronto.shape == (64, 64, 1)
    assert iguais(pronto[:, :, 0], esperado)


def test_clahe_on_ycrcb_uses_the_first_channel_and_does_not_convert_back(
    quadro: Frame,
) -> None:
    pronto = prepare(quadro, spec(color="ycrcb", ops=("clahe:2.0:8",)))

    esperado = cv2.resize(cv2.cvtColor(quadro, cv2.COLOR_BGR2YCrCb), (64, 64))
    esperado[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(esperado[:, :, 0])
    assert iguais(pronto, esperado)


def test_normalize_imagenet_reverses_the_statistics_for_bgr(quadro: Frame) -> None:
    unit = cv2.resize(quadro, (64, 64)).astype(np.float32) / 255.0
    media = np.array([0.406, 0.456, 0.485], dtype=np.float32)
    desvio = np.array([0.225, 0.224, 0.229], dtype=np.float32)

    pronto = prepare(quadro, spec(color="bgr", normalize="imagenet"))

    assert float(np.abs(pronto - (unit - media) / desvio).max()) < 1e-6


def test_clahe_does_not_write_into_the_frame_it_received(quadro: Frame) -> None:
    prepare(quadro, spec(color="ycrcb", ops=("clahe:2.0:8",)))

    assert iguais(quadro, _quadro())


def test_ops_run_before_normalize_because_clahe_refuses_floating_point(
    quadro: Frame,
) -> None:
    pronto = prepare(quadro, spec(color="bgr", ops=("clahe:2.0:8",), normalize="unit"))

    esperado = _clahe(cv2.resize(quadro, (64, 64))).astype(np.float32) / 255.0
    assert pronto.dtype == np.float32
    assert float(np.abs(pronto - esperado).max()) == 0.0


def test_clahe_touches_only_the_luminance_of_a_ycrcb_frame(quadro: Frame) -> None:
    cru = prepare(quadro, spec(color="ycrcb"))

    tratado = prepare(quadro, spec(color="ycrcb", ops=("clahe:2.0:8",)))

    assert not iguais(tratado[:, :, 0], cru[:, :, 0])
    assert iguais(tratado[:, :, 1:], cru[:, :, 1:])


def test_letterbox_refuses_a_shape_that_collapses_the_content(quadro: Frame) -> None:
    magro = cv2.resize(quadro, (2000, 2))

    pronto = prepare(magro, spec(letterbox=True))

    assert pronto.shape == (64, 64, 3)
    assert pronto[:, :, :].any()


@pytest.mark.parametrize("color", ["rgb", "bgr"])
@pytest.mark.parametrize("normalization", ["unit", "imagenet"])
def test_normalization_preserves_pixels_and_does_not_modify_input(
    quadro: Frame, color: str, normalization: str
) -> None:
    original = quadro.copy()
    raw = prepare(quadro, spec(color=color))
    expected = raw.astype(np.float32) / 255.0
    if normalization == "imagenet":
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        if color == "bgr":
            mean, std = mean[::-1], std[::-1]
        expected = (expected - mean) / std

    actual = prepare(quadro, spec(color=color, normalize=normalization))

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(quadro, original)
