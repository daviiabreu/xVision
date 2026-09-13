"""Contrato pequeno e verificavel para runtimes de inferencia."""

from __future__ import annotations

import hashlib
import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from importlib import import_module
from pathlib import Path
from typing import Literal, Protocol, cast

import numpy as np
from numpy.typing import NDArray

Precision = Literal["float16", "float32"]
Layout = Literal["NCHW", "flat"]
Array = NDArray[np.float16] | NDArray[np.float32]
RuntimeModel = Callable[[Mapping[str, Array]], Mapping[str, Array]]
ModelLoader = Callable[[Path, str], RuntimeModel]


class RuntimeError(ValueError): ...


class RuntimeUnavailable(RuntimeError): ...


class ArtifactIncompatible(RuntimeError): ...


class ShapeMismatch(RuntimeError): ...


class ProviderMismatch(RuntimeError): ...


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: Precision
    layout: Layout

    def __post_init__(self) -> None:
        if not self.name:
            raise ArtifactIncompatible("binding sem nome")
        if not self.shape or any(dimension <= 0 for dimension in self.shape):
            raise ArtifactIncompatible(
                f"{self.name} exige shape fixo positivo, recebeu {self.shape}"
            )


@dataclass(frozen=True)
class RuntimeArtifact:
    path: Path
    sha256: str
    format: Literal["pt", "engine"]
    parent_onnx_sha256: str | None
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]

    def __post_init__(self) -> None:
        if re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None:
            raise ArtifactIncompatible("artefato sem sha256 valido")
        if self.format == "engine" and self.parent_onnx_sha256 is None:
            raise ArtifactIncompatible("engine exige parent_onnx_sha256")
        _unique(self.inputs, "entrada")
        _unique(self.outputs, "saida")


@dataclass(frozen=True)
class RuntimeTensor:
    name: str
    value: Array


@dataclass(frozen=True)
class RuntimeInfo:
    backend: Literal["pytorch", "tensorrt"]
    provider: str
    device: str
    precision: Precision
    runtime_versions: tuple[tuple[str, str], ...]
    artifact_sha256: str
    parent_onnx_sha256: str | None
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]
    inference_count: int
    fallback_reason: str | None


class InferenceBackend(Protocol):
    @property
    def info(self) -> RuntimeInfo: ...

    def load(self, artifact: RuntimeArtifact) -> None: ...

    def infer(self, inputs: Mapping[str, Array]) -> tuple[RuntimeTensor, ...]: ...

    def close(self) -> None: ...


class PyTorchBackend:
    """Executa um modelo PyTorch carregado por um adapter especifico do artefato."""

    def __init__(
        self,
        *,
        provider: str,
        device: str,
        loader: ModelLoader,
        runtime_versions: tuple[tuple[str, str], ...] = (),
    ) -> None:
        self._provider = provider
        self._device = device
        self._loader = loader
        self._runtime_versions = runtime_versions
        self._artifact: RuntimeArtifact | None = None
        self._model: RuntimeModel | None = None
        self._info: RuntimeInfo | None = None
        self._inference_lock = threading.Lock()

    @property
    def info(self) -> RuntimeInfo:
        if self._info is None:
            raise RuntimeUnavailable("runtime ainda nao foi carregado")
        return self._info

    def load(self, artifact: RuntimeArtifact) -> None:
        if self._artifact is not None:
            if self._artifact.sha256 == artifact.sha256:
                return
            raise ArtifactIncompatible("runtime ja foi carregado com outro artefato")
        if artifact.format != "pt":
            raise ArtifactIncompatible(f"PyTorch nao abre formato {artifact.format!r}")
        if not artifact.path.is_file():
            raise RuntimeUnavailable(f"{artifact.path} nao existe")
        actual = _sha256(artifact.path)
        if actual != artifact.sha256:
            raise ArtifactIncompatible(
                f"{artifact.path.name} tem sha256 {actual[:12]}, esperado {artifact.sha256[:12]}"
            )
        self._model = self._loader(artifact.path, self._device)
        self._artifact = artifact
        self._info = RuntimeInfo(
            backend="pytorch",
            provider=self._provider,
            device=self._device,
            precision="float32",
            runtime_versions=self._runtime_versions,
            artifact_sha256=artifact.sha256,
            parent_onnx_sha256=artifact.parent_onnx_sha256,
            inputs=artifact.inputs,
            outputs=artifact.outputs,
            inference_count=0,
            fallback_reason=None,
        )

    def infer(self, inputs: Mapping[str, Array]) -> tuple[RuntimeTensor, ...]:
        if self._model is None or self._artifact is None or self._info is None:
            raise RuntimeUnavailable("runtime nao esta carregado")
        if not self._inference_lock.acquire(blocking=False):
            raise RuntimeUnavailable("runtime nao aceita inferencia concorrente")
        try:
            _validate_tensors(inputs, self._artifact.inputs)
            raw = self._model(inputs)
            _validate_tensors(raw, self._artifact.outputs)
            outputs = tuple(
                RuntimeTensor(spec.name, _immutable_copy(raw[spec.name]))
                for spec in self._artifact.outputs
            )
            self._info = replace(self._info, inference_count=self._info.inference_count + 1)
            return outputs
        finally:
            self._inference_lock.release()

    def close(self) -> None:
        self._model = None
        self._artifact = None
        self._info = None


class TensorRTBackend:
    """Executa um engine TensorRT montado por um loader especifico da especie.

    E irmao do `PyTorchBackend`, e nao um modo dele, porque as duas diferencas sao de contrato e
    nao de ajuste: o formato aceito (`engine` contra `pt`) e o nome que vai para
    `RuntimeInfo.backend`, que o registro da execucao usa para dizer o que de fato rodou. Um
    booleano no construtor faria essas duas invariantes dependerem de quem chama.

    O loader tem a mesma assinatura do PyTorch de proposito: recebe caminho e device, devolve algo
    que leva numpy a numpy. Assim `edge.runtime` nao importa `tensorrt`, o venv de teste (que nao
    tem nem tensorrt nem torch) continua importando este modulo, e o engine de verdade mora em
    `edge.trt`, carregado so na placa.

    `precision` reporta o dtype dos bindings, nao o do computo interno. Um engine com peso em
    fp16 e pontas em fp32 aparece aqui como `float32`, que e o que o chamador entrega e recebe;
    a meia precisao de dentro fica registrada no ONNX que `parent_onnx_sha256` fixa. O TensorRT
    nao expoe a precisao de computo pelos bindings, entao inventar o campo seria pior que o
    limite declarado.
    """

    def __init__(
        self,
        *,
        provider: str,
        device: str,
        loader: ModelLoader,
        runtime_versions: tuple[tuple[str, str], ...] = (),
    ) -> None:
        self._provider = provider
        self._device = device
        self._loader = loader
        self._runtime_versions = runtime_versions
        self._artifact: RuntimeArtifact | None = None
        self._model: RuntimeModel | None = None
        self._info: RuntimeInfo | None = None
        self._inference_lock = threading.Lock()

    @property
    def info(self) -> RuntimeInfo:
        if self._info is None:
            raise RuntimeUnavailable("runtime ainda nao foi carregado")
        return self._info

    def load(self, artifact: RuntimeArtifact) -> None:
        if self._artifact is not None:
            if self._artifact.sha256 == artifact.sha256:
                return
            raise ArtifactIncompatible("runtime ja foi carregado com outro artefato")
        if artifact.format != "engine":
            raise ArtifactIncompatible(f"TensorRT nao abre formato {artifact.format!r}")
        if not artifact.path.is_file():
            raise RuntimeUnavailable(f"{artifact.path} nao existe")
        actual = _sha256(artifact.path)
        if actual != artifact.sha256:
            raise ArtifactIncompatible(
                f"{artifact.path.name} tem sha256 {actual[:12]}, esperado {artifact.sha256[:12]}"
            )
        self._model = self._loader(artifact.path, self._device)
        self._artifact = artifact
        self._info = RuntimeInfo(
            backend="tensorrt",
            provider=self._provider,
            device=self._device,
            precision=artifact.inputs[0].dtype,
            runtime_versions=self._runtime_versions,
            artifact_sha256=artifact.sha256,
            parent_onnx_sha256=artifact.parent_onnx_sha256,
            inputs=artifact.inputs,
            outputs=artifact.outputs,
            inference_count=0,
            fallback_reason=None,
        )

    def infer(self, inputs: Mapping[str, Array]) -> tuple[RuntimeTensor, ...]:
        if self._model is None or self._artifact is None or self._info is None:
            raise RuntimeUnavailable("runtime nao esta carregado")
        if not self._inference_lock.acquire(blocking=False):
            raise RuntimeUnavailable("runtime nao aceita inferencia concorrente")
        try:
            _validate_tensors(inputs, self._artifact.inputs)
            raw = self._model(inputs)
            _validate_tensors(raw, self._artifact.outputs)
            outputs = tuple(
                RuntimeTensor(spec.name, _immutable_copy(raw[spec.name]))
                for spec in self._artifact.outputs
            )
            self._info = replace(self._info, inference_count=self._info.inference_count + 1)
            return outputs
        finally:
            self._inference_lock.release()

    def close(self) -> None:
        self._model = None
        self._artifact = None
        self._info = None


def verify_onnx(path: Path) -> None:
    """Recusa um ONNX estruturalmente invalido antes de qualquer runtime embarcado."""
    if not path.is_file():
        raise RuntimeUnavailable(f"{path} nao existe")
    onnx = import_module("onnx")
    model = onnx.load(path)
    onnx.checker.check_model(model)
    onnx.shape_inference.infer_shapes(model)


def _unique(specs: tuple[TensorSpec, ...], kind: str) -> None:
    names = [spec.name for spec in specs]
    if not names or len(names) != len(set(names)):
        raise ArtifactIncompatible(f"bindings de {kind} precisam de nomes unicos")


def _validate_tensors(tensors: Mapping[str, Array], specs: tuple[TensorSpec, ...]) -> None:
    expected = {spec.name for spec in specs}
    if set(tensors) != expected:
        raise ShapeMismatch(f"bindings recebidos {sorted(tensors)}, esperados {sorted(expected)}")
    for spec in specs:
        value = tensors[spec.name]
        if value.shape != spec.shape:
            raise ShapeMismatch(f"{spec.name} recebeu shape {value.shape}, esperado {spec.shape}")
        if value.dtype != np.dtype(spec.dtype):
            raise ShapeMismatch(f"{spec.name} recebeu dtype {value.dtype}, esperado {spec.dtype}")
        if not value.flags.c_contiguous:
            raise ShapeMismatch(f"{spec.name} exige array C-contiguo")


def _immutable_copy(value: Array) -> Array:
    copy = np.array(value, copy=True, order="C")
    copy.flags.writeable = False
    return cast("Array", copy)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()
