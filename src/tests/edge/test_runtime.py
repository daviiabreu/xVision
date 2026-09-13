"""Contrato comum dos runtimes de rede usados por campo e bola."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from edge.runtime import (
    ArtifactIncompatible,
    PyTorchBackend,
    RuntimeArtifact,
    RuntimeUnavailable,
    ShapeMismatch,
    TensorRTBackend,
    TensorSpec,
    verify_onnx,
)


def _artifact(path: Path, digest: str) -> RuntimeArtifact:
    return RuntimeArtifact(
        path=path,
        sha256=digest,
        format="pt",
        parent_onnx_sha256=None,
        inputs=(TensorSpec("images", (1, 3, 4, 4), "float32", "NCHW"),),
        outputs=(TensorSpec("predictions", (1, 2), "float32", "flat"),),
    )


def test_runtime_refuses_dynamic_shapes() -> None:
    with pytest.raises(ArtifactIncompatible, match="shape fixo"):
        TensorSpec("images", (1, 3, -1, 960), "float32", "NCHW")


def test_pytorch_backend_validates_input_and_owns_immutable_output(tmp_path: Path) -> None:
    weight = tmp_path / "model.pt"
    weight.write_bytes(b"weight")
    digest = hashlib.sha256(b"weight").hexdigest()
    loads: list[Path] = []

    def loader(path: Path, device: str) -> Any:
        loads.append(path)

        def model(inputs: dict[str, np.ndarray[Any, Any]]) -> dict[str, np.ndarray[Any, Any]]:
            return {"predictions": np.asarray([[inputs["images"].sum(), 2]], dtype=np.float32)}

        return model

    backend = PyTorchBackend(provider="test", device="cpu", loader=loader)
    artifact = _artifact(weight, digest)
    backend.load(artifact)
    backend.load(artifact)

    output = backend.infer({"images": np.ones((1, 3, 4, 4), dtype=np.float32)})[0]

    assert loads == [weight]
    assert output.name == "predictions"
    assert output.value.tolist() == [[48.0, 2.0]]
    assert not output.value.flags.writeable
    assert backend.info.inference_count == 1
    with pytest.raises(ShapeMismatch, match="images"):
        backend.infer({"images": np.ones((1, 3, 2, 2), dtype=np.float32)})


def test_backend_rejects_another_artifact_and_inference_after_close(tmp_path: Path) -> None:
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    def empty_loader(path: Path, device: str) -> Any:
        def empty_model(inputs: dict[str, np.ndarray[Any, Any]]) -> dict[str, np.ndarray[Any, Any]]:
            return {}

        return empty_model

    backend = PyTorchBackend(provider="test", device="cpu", loader=empty_loader)
    backend.load(_artifact(first, hashlib.sha256(b"first").hexdigest()))

    with pytest.raises(ArtifactIncompatible, match="outro artefato"):
        backend.load(_artifact(second, hashlib.sha256(b"second").hexdigest()))

    backend.close()
    backend.close()
    with pytest.raises(RuntimeUnavailable, match="carregado"):
        backend.infer({"images": np.ones((1, 3, 4, 4), dtype=np.float32)})


def test_onnx_verifier_runs_checker_and_shape_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def load(path: Path) -> str:
        calls.append(f"load:{Path(path).name}")
        return "model"

    def check(model: object) -> None:
        calls.append(f"check:{model}")

    def infer_shapes(model: object) -> str:
        calls.append(f"shape:{model}")
        return "shaped"

    fake = SimpleNamespace(
        load=load,
        checker=SimpleNamespace(check_model=check),
        shape_inference=SimpleNamespace(infer_shapes=infer_shapes),
    )

    def load_module(name: str) -> object:
        return fake

    monkeypatch.setattr("edge.runtime.import_module", load_module)
    path = tmp_path / "field.onnx"
    path.write_bytes(b"onnx")

    verify_onnx(path)

    assert calls == ["load:field.onnx", "check:model", "shape:model"]


def _engine(path: Path, digest: str) -> RuntimeArtifact:
    return RuntimeArtifact(
        path=path,
        sha256=digest,
        format="engine",
        parent_onnx_sha256="b" * 64,
        inputs=(TensorSpec("images", (1, 3, 4, 4), "float32", "NCHW"),),
        outputs=(TensorSpec("predictions", (1, 2), "float32", "flat"),),
    )


def _plano_no_disco(tmp_path: Path) -> tuple[Path, str]:
    plano = tmp_path / "ball.engine"
    plano.write_bytes(b"plano serializado")
    return plano, hashlib.sha256(b"plano serializado").hexdigest()


def _loader_falso(saida: Any = None) -> Any:
    def loader(path: Path, device: str) -> Any:
        def model(inputs: dict[str, Any]) -> dict[str, Any]:
            if saida is not None:
                return saida
            return {"predictions": np.asarray([[inputs["images"].sum(), 2]], dtype=np.float32)}

        return model

    return loader


def test_tensorrt_backend_reporta_ter_rodado_engine(tmp_path: Path) -> None:
    """O registro da execucao precisa distinguir engine de `.pt`, senao a comparacao entre os dois
    caminhos reporta numero de um e procedencia do outro."""
    plano, digest = _plano_no_disco(tmp_path)
    backend = TensorRTBackend(provider="trt", device="cuda:0", loader=_loader_falso())
    backend.load(_engine(plano, digest))

    assert backend.info.backend == "tensorrt"
    assert backend.info.parent_onnx_sha256 == "b" * 64
    assert backend.info.inference_count == 0

    (saida,) = backend.infer({"images": np.ones((1, 3, 4, 4), dtype=np.float32)})
    assert saida.name == "predictions"
    assert not saida.value.flags.writeable
    assert backend.info.inference_count == 1


def test_tensorrt_backend_recusa_peso_pytorch(tmp_path: Path) -> None:
    """Apontar o engine para um `.pt` e erro de configuracao, e falhar calado custaria a medicao."""
    peso = tmp_path / "ball.pt"
    peso.write_bytes(b"peso")
    artefato = RuntimeArtifact(
        path=peso,
        sha256=hashlib.sha256(b"peso").hexdigest(),
        format="pt",
        parent_onnx_sha256=None,
        inputs=(TensorSpec("images", (1, 3, 4, 4), "float32", "NCHW"),),
        outputs=(TensorSpec("predictions", (1, 2), "float32", "flat"),),
    )
    backend = TensorRTBackend(provider="trt", device="cuda:0", loader=_loader_falso())
    with pytest.raises(ArtifactIncompatible, match="formato 'pt'"):
        backend.load(artefato)


def test_tensorrt_backend_recusa_plano_trocado(tmp_path: Path) -> None:
    """Engine nao atravessa versao de TensorRT nem placa: sha256 diferente e artefato diferente."""
    plano, digest = _plano_no_disco(tmp_path)
    plano.write_bytes(b"outro plano")
    backend = TensorRTBackend(provider="trt", device="cuda:0", loader=_loader_falso())
    with pytest.raises(ArtifactIncompatible, match="sha256"):
        backend.load(_engine(plano, digest))


def test_tensorrt_backend_recusa_saida_com_shape_errado(tmp_path: Path) -> None:
    """O plano tem shape fixo; outra saida significa que ele nao e o que o spec declara."""
    plano, digest = _plano_no_disco(tmp_path)
    errada = {"predictions": np.zeros((1, 5), dtype=np.float32)}
    backend = TensorRTBackend(provider="trt", device="cuda:0", loader=_loader_falso(errada))
    backend.load(_engine(plano, digest))
    with pytest.raises(ShapeMismatch, match="predictions"):
        backend.infer({"images": np.ones((1, 3, 4, 4), dtype=np.float32)})


def test_tensorrt_backend_sem_carregar_nao_infere() -> None:
    backend = TensorRTBackend(provider="trt", device="cuda:0", loader=_loader_falso())
    with pytest.raises(RuntimeUnavailable, match="nao esta carregado"):
        backend.infer({"images": np.ones((1, 3, 4, 4), dtype=np.float32)})
    with pytest.raises(RuntimeUnavailable, match="ainda nao foi carregado"):
        _ = backend.info
