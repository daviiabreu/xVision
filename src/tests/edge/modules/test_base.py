"""Testes dos tipos que um dono de modelo preenche para declarar o proprio modulo."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from edge.modules.base import (
    FrameBatch,
    ModuleError,
    ModuleOutput,
    ModuleSpec,
    PreprocessMismatch,
    PreprocessSpec,
    UnknownModuleKind,
)

PREPROCESS = PreprocessSpec(
    width=640,
    height=640,
    color="rgb",
    letterbox=True,
    normalize="unit",
    ops=("clahe:2.0:8",),
)

PREPROCESS_FINGERPRINT = "1a8481b453ff0005d1ee9b60393415a018b84e5bd8e627ddddc053827d43d5fc"


def _fingerprint_in_subprocess(hash_seed: str) -> str:
    source = (
        "from edge.modules.base import PreprocessSpec;"
        "print(PreprocessSpec(width=640, height=640, color='rgb', letterbox=True,"
        " normalize='unit', ops=('clahe:2.0:8',)).fingerprint)"
    )
    done = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        check=True,
        text=True,
        env={"PYTHONHASHSEED": hash_seed, "PATH": "/usr/bin:/bin"},
    )
    return done.stdout.strip()


def test_module_spec_rejects_unknown_kind() -> None:
    with pytest.raises(UnknownModuleKind, match="offside"):
        ModuleSpec(
            kind="offside",
            name="qualquer",
            version="0.1.0",
            frames="coarse",
            preprocess=PREPROCESS,
        )


def test_preprocess_spec_is_hashable_and_stable() -> None:
    assert {PREPROCESS, replace(PREPROCESS)} == {PREPROCESS}
    assert PREPROCESS.fingerprint == PREPROCESS_FINGERPRINT

    seen = {_fingerprint_in_subprocess(seed) for seed in ("0", "1", "random")}
    assert seen == {PREPROCESS_FINGERPRINT}, "a chave mudou entre processos"

    assert replace(PREPROCESS, ops=()).fingerprint != PREPROCESS_FINGERPRINT
    assert replace(PREPROCESS, letterbox=False).fingerprint != PREPROCESS_FINGERPRINT


def test_frame_batch_rejects_a_tensor_that_does_not_match_the_spec() -> None:
    small = PreprocessSpec(width=8, height=8, color="gray", letterbox=False, normalize="none")
    with pytest.raises(PreprocessMismatch, match="nao corresponde"):
        FrameBatch(indices=(0,), tensor=np.zeros((1, 8, 8, 3), np.uint8), preprocess=small)
    with pytest.raises(PreprocessMismatch, match="normalize"):
        FrameBatch(indices=(0,), tensor=np.zeros((1, 8, 8, 1), np.float32), preprocess=small)


def test_module_output_requires_one_record_set_per_frame() -> None:
    with pytest.raises(ModuleError, match="2 quadros e 1 conjuntos"):
        ModuleOutput(kind="ball", frames=(0, 1), records=((),))
