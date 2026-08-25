"""Suite de conformidade: o que qualquer `PerceptionModule` precisa cumprir.

Quem escreve um modelo herda esta classe, fornece a fixture `module` e descobre
sozinho se o adaptador dele e integravel, sem depender de ninguem do backend.

    import pytest
    from edge.modules.conformance import ModuleConformance

    class TestMeuModulo(ModuleConformance):
        @pytest.fixture
        def module(self):
            return MeuModulo()
"""

from __future__ import annotations

import os
import sys
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import PurePath
from typing import Any, cast

import numpy as np
import pytest

from edge.modules.base import (
    FrameBatch,
    ModuleNotLoaded,
    ModuleOutput,
    PerceptionModule,
    PreprocessMismatch,
    PreprocessSpec,
    channels,
    tensor_dtype,
)
from edge.modules.records import check_records

VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg"})

_sinks: list[list[str]] = []
_hook_installed = False


def make_batch(
    preprocess: PreprocessSpec,
    indices: Sequence[int] = (0,),
    fill: float = 0.0,
) -> FrameBatch:
    dtype = tensor_dtype(preprocess.normalize)
    shape = (len(indices), preprocess.height, preprocess.width, channels(preprocess.color))
    value = round(fill * 255) if dtype == np.uint8 else fill
    return FrameBatch(
        indices=tuple(indices),
        tensor=np.full(shape, value, dtype=dtype),
        preprocess=preprocess,
    )


def _as_path(target: object) -> str | None:
    if isinstance(target, os.PathLike):
        target = cast("os.PathLike[str]", target).__fspath__()
    if isinstance(target, bytes):
        return target.decode("utf-8", "replace")
    return target if isinstance(target, str) else None


def _audit(event: str, args: tuple[Any, ...]) -> None:
    if not _sinks or event not in {"open", "os.open"}:
        return
    path = _as_path(args[0])
    if path is not None and PurePath(path).suffix.lower() in VIDEO_SUFFIXES:
        for sink in _sinks:
            sink.append(path)


@contextmanager
def no_video_access() -> Generator[list[str]]:
    global _hook_installed
    if not _hook_installed:
        sys.addaudithook(_audit)
        _hook_installed = True

    sink: list[str] = []
    _sinks.append(sink)
    cv2: Any = sys.modules.get("cv2")
    original = getattr(cv2, "VideoCapture", None)
    if original is not None:

        def guard(*args: Any, **kwargs: Any) -> Any:
            sink.append(str(args[0]) if args else "cv2.VideoCapture()")
            return original(*args, **kwargs)

        cv2.VideoCapture = guard
    try:
        yield sink
    finally:
        if original is not None:
            cv2.VideoCapture = original
        _sinks.remove(sink)


class ModuleConformance:
    @pytest.fixture
    def module(self) -> PerceptionModule:
        raise NotImplementedError("forneca a fixture `module` devolvendo a sua implementacao")

    @pytest.fixture
    def loaded(self, module: PerceptionModule) -> Generator[PerceptionModule]:
        module.load()
        yield module
        module.close()

    def test_conformance_respects_declared_preprocess(self, loaded: PerceptionModule) -> None:
        declared = loaded.spec.preprocess
        loaded.infer(make_batch(declared), {})

        outro = replace(declared, width=declared.width + 32)
        with pytest.raises(PreprocessMismatch):
            loaded.infer(make_batch(outro), {})

    def test_conformance_returns_declared_output_type(self, loaded: PerceptionModule) -> None:
        batch = make_batch(loaded.spec.preprocess, indices=(0, 7))
        output = loaded.infer(batch, {})
        assert isinstance(output, ModuleOutput)
        assert output.kind == loaded.spec.kind
        assert output.frames == batch.indices

    def test_conformance_handles_empty_detections(self, loaded: PerceptionModule) -> None:
        declared = loaded.spec.preprocess
        vazio = loaded.infer(make_batch(declared, indices=(0, 1), fill=0.0), {})
        assert len(vazio.records) == 2

        sem_quadro = loaded.infer(make_batch(declared, indices=()), {})
        assert sem_quadro.records == ()

    def test_conformance_is_deterministic_for_same_input(self, loaded: PerceptionModule) -> None:
        batch = make_batch(loaded.spec.preprocess, indices=(3, 4), fill=0.9)
        assert loaded.infer(batch, {}) == loaded.infer(batch, {})

    def test_conformance_close_releases_what_load_acquired(self, module: PerceptionModule) -> None:
        module.load()
        batch = make_batch(module.spec.preprocess)
        module.infer(batch, {})

        module.close()
        with pytest.raises(ModuleNotLoaded):
            module.infer(batch, {})
        module.close()

    def test_conformance_does_not_open_video_itself(self, loaded: PerceptionModule) -> None:
        batch = make_batch(loaded.spec.preprocess)
        with no_video_access() as aberturas:
            loaded.infer(batch, {})
        assert aberturas == []

    def test_conformance_records_carry_what_the_species_promises(
        self, loaded: PerceptionModule
    ) -> None:
        batch = make_batch(loaded.spec.preprocess, fill=0.9)
        check_records(loaded.infer(batch, {}))
