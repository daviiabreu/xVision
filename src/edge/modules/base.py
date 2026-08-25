"""O contrato que todo modelo de percepcao cumpre para entrar no pipeline.

Cinco pessoas trazem cinco pilhas diferentes, entao o backend nao assume framework,
formato nem ambiente: assume estes tipos. O modulo declara o que consome e em que
forma, e o pipeline entrega exatamente isso.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

MODULE_KINDS = frozenset({"players", "ball", "field", "tracking", "pose", "identity", "shot"})

Color = Literal["rgb", "bgr", "gray", "ycrcb"]
Normalize = Literal["none", "unit", "imagenet"]

_CHANNELS: Mapping[str, int] = {"rgb": 3, "bgr": 3, "ycrcb": 3, "gray": 1}


class ModuleError(Exception): ...


class UnknownModuleKind(ModuleError): ...


class PreprocessMismatch(ModuleError): ...


class ModuleNotLoaded(ModuleError): ...


@dataclass(frozen=True)
class PerFrame: ...


@dataclass(frozen=True)
class Temporal:
    size: int
    stride: int = 1


Batching = PerFrame | Temporal


@dataclass(frozen=True)
class PreprocessSpec:
    width: int
    height: int
    color: Color
    letterbox: bool
    normalize: Normalize
    ops: tuple[str, ...] = ()

    @property
    def fingerprint(self) -> str:
        canonical = "|".join(
            (
                str(self.width),
                str(self.height),
                self.color,
                "1" if self.letterbox else "0",
                self.normalize,
                ",".join(self.ops),
            )
        )
        return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True)
class ModuleSpec:
    kind: str
    name: str
    version: str
    frames: str
    preprocess: PreprocessSpec
    needs: tuple[str, ...] = ()
    batching: Batching = PerFrame()
    artifact: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in MODULE_KINDS:
            raise UnknownModuleKind(
                f"{self.kind!r} nao e uma especie de modulo conhecida: "
                f"{', '.join(sorted(MODULE_KINDS))}"
            )


def channels(color: Color) -> int:
    return _CHANNELS[color]


def tensor_dtype(normalize: Normalize) -> np.dtype[np.uint8] | np.dtype[np.float32]:
    return np.dtype(np.uint8) if normalize == "none" else np.dtype(np.float32)


@dataclass(frozen=True)
class FrameBatch:
    indices: tuple[int, ...]
    tensor: np.ndarray[tuple[int, ...], np.dtype[np.generic]]
    preprocess: PreprocessSpec

    def __post_init__(self) -> None:
        expected = (
            len(self.indices),
            self.preprocess.height,
            self.preprocess.width,
            channels(self.preprocess.color),
        )
        if self.tensor.shape != expected:
            raise PreprocessMismatch(
                f"tensor {self.tensor.shape} nao corresponde a spec declarada {expected}"
            )
        wanted = tensor_dtype(self.preprocess.normalize)
        if self.tensor.dtype != wanted:
            raise PreprocessMismatch(
                f"normalize={self.preprocess.normalize!r} pede {wanted}, "
                f"e o tensor veio {self.tensor.dtype}"
            )


@dataclass(frozen=True)
class ModuleOutput:
    kind: str
    frames: tuple[int, ...]
    records: tuple[tuple[Mapping[str, object], ...], ...]

    def __post_init__(self) -> None:
        if len(self.frames) != len(self.records):
            raise ModuleError(
                f"{len(self.frames)} quadros e {len(self.records)} conjuntos de registros"
            )


ModuleInputs = Mapping[str, ModuleOutput]
"""Saidas dos modulos declarados em `needs`, ja resolvidas pelo runner."""


class PerceptionModule(Protocol):
    spec: ModuleSpec

    def load(self) -> None: ...
    def infer(self, batch: FrameBatch, ctx: ModuleInputs) -> ModuleOutput: ...
    def close(self) -> None: ...


def check_preprocess(batch: FrameBatch, preprocess: PreprocessSpec) -> None:
    if batch.preprocess != preprocess:
        raise PreprocessMismatch(
            f"o modulo declarou {preprocess.fingerprint[:12]} e recebeu "
            f"{batch.preprocess.fingerprint[:12]}"
        )
