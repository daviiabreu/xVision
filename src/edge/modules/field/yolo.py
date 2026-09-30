"""Adapter do Ultralytics para os 29 keypoints semanticos do campo."""

from __future__ import annotations

import re
from collections.abc import Mapping
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import numpy as np

from edge.artifacts import sha256_of
from edge.modules.base import ModuleError
from edge.modules.field.geometry import PITCH_105X68, ImageKeypoint
from edge.modules.precision import PADRAO, ajuste_do_predict
from edge.runtime import (
    Array,
    InferenceBackend,
    ModelLoader,
    PyTorchBackend,
    RuntimeArtifact,
    RuntimeInfo,
    RuntimeModel,
    TensorRTBackend,
    TensorSpec,
)

INPUT = TensorSpec("images", (1, 3, 544, 960), "float32", "NCHW")
OUTPUT = TensorSpec("predictions", (1, 92, 10710), "float32", "flat")


class FieldModelIncompatible(ModuleError): ...


class UltralyticsFieldEstimator:
    def __init__(
        self,
        model: Any | None,
        *,
        imgsz: int,
        conf: float,
        device: str,
        quantize: str = PADRAO,
        backend: InferenceBackend | None = None,
    ) -> None:
        self._model = model
        self._imgsz = imgsz
        self._conf = conf
        self._device = device
        self._quantize = quantize
        self._backend = backend

    @classmethod
    def from_weights(
        cls,
        weights: str,
        *,
        imgsz: int,
        conf: float,
        device: str,
        quantize: str = PADRAO,
        engine: str = "",
        engine_onnx_sha256: str = "",
    ) -> UltralyticsFieldEstimator:
        """Monta o runtime da especie: um plano TensorRT quando declarado, senao o PyTorch.

        Medido nesta Jetson, entrada (1,3,544,960): o forward do PyTorch custa 32,9 ms em fp32 e
        37,5 ms em fp16, e o plano custa 7,9 ms — 76,1% a menos que o fp32, ou 4,2x.

        O campo e o caso que separa os dois mecanismos. Meia precisao o deixa MAIS LENTO, medido
        duas vezes por caminhos diferentes, porque o modelo e pequeno a 960 e fica preso em
        lancamento de kernel em vez de computo. A fusao de camadas do TensorRT ataca exatamente
        lancamento, e por isso rende aqui o maior ganho das tres especies. Nao assuma que uma coisa
        prediz a outra: aqui elas apontam para lados opostos.
        """
        if imgsz != 960:
            raise FieldModelIncompatible(
                f"backend de campo foi congelado em imgsz=960, recebeu {imgsz}"
            )
        if engine:
            backend = _backend_de_plano(engine, engine_onnx_sha256, device, quantize)
        else:
            backend = _backend_de_peso(weights, device, quantize)
        return cls(None, imgsz=imgsz, conf=conf, device=device, quantize=quantize, backend=backend)

    @property
    def runtime_info(self) -> RuntimeInfo | None:
        return self._backend.info if self._backend is not None else None

    def close(self) -> None:
        if self._backend is not None:
            self._backend.close()

    def estimate(self, frame: Any) -> tuple[ImageKeypoint, ...] | None:
        if self._backend is not None:
            return self._estimate_with_backend(frame)
        if self._model is None:
            return None
        outputs = self._model.predict(
            source=frame,
            imgsz=self._imgsz,
            conf=self._conf,
            device=self._device,
            verbose=False,
            **ajuste_do_predict(self._quantize),
        )
        if not outputs:
            return None
        output = outputs[0]
        keypoints = getattr(output, "keypoints", None)
        xy = getattr(keypoints, "xy", None)
        confidence = getattr(keypoints, "conf", None)
        if xy is None or confidence is None or len(xy) == 0:
            return None
        index = _largest(getattr(output, "boxes", None))
        coordinates = xy[index].tolist()
        scores = confidence[index].tolist()
        if len(coordinates) != len(PITCH_105X68.keypoints) or len(scores) != len(coordinates):
            return None
        return tuple(
            ImageKeypoint(point.name, (float(pixel[0]), float(pixel[1])), float(score))
            for point, pixel, score in zip(PITCH_105X68.keypoints, coordinates, scores, strict=True)
        )

    def _estimate_with_backend(self, frame: Any) -> tuple[ImageKeypoint, ...] | None:
        tensor = tensor_de_entrada(frame)
        backend = self._backend
        if backend is None:
            raise FieldModelIncompatible("field sem backend carregado")
        raw = backend.infer({"images": tensor})[0].value
        torch = cast("Any", import_module("torch"))
        nms = cast("Any", import_module("ultralytics.utils.nms"))
        detections = nms.non_max_suppression(
            torch.from_numpy(np.array(raw, copy=True)),
            conf_thres=self._conf,
            nc=1,
            max_det=10,
        )[0]
        if len(detections) == 0:
            return None
        rows = detections.detach().cpu().numpy()
        index = max(
            range(len(rows)),
            key=lambda i: (rows[i, 2] - rows[i, 0]) * (rows[i, 3] - rows[i, 1]),
        )
        keypoints = rows[index, 6:].reshape(len(PITCH_105X68.keypoints), 3)
        return tuple(
            ImageKeypoint(
                point.name, (float(raw_point[0]), float(raw_point[1])), float(raw_point[2])
            )
            for point, raw_point in zip(PITCH_105X68.keypoints, keypoints, strict=True)
        )


def tensor_de_entrada(frame: Any) -> np.ndarray[Any, Any]:
    """O tensor que o backend do campo recebe, montado aqui e em nenhum outro lugar.

    A calibracao do INT8, em `mlops.quantize_int8`, precisa do mesmo tensor que o plano recebe em
    producao. Uma copia desta conversao la calibraria o plano com outra entrada no dia em que esta
    mudasse.
    """
    image = np.asarray(frame)
    if image.shape != (544, 960, 3) or image.dtype != np.uint8:
        raise FieldModelIncompatible(
            f"field recebeu tensor {image.shape} {image.dtype}, esperado (544, 960, 3) uint8"
        )
    rgb = image[:, :, ::-1].transpose(2, 0, 1)[None]
    return np.ascontiguousarray(rgb, dtype=np.float32) / np.float32(255.0)


def _largest(boxes: object) -> int:
    xyxy = getattr(boxes, "xyxy", None)
    if xyxy is None or len(xyxy) <= 1:
        return 0
    rows = xyxy.tolist()
    return max(
        range(len(rows)),
        key=lambda index: (rows[index][2] - rows[index][0]) * (rows[index][3] - rows[index][1]),
    )


def check_field_model(model: object) -> None:
    task = getattr(model, "task", None)
    inner = getattr(model, "model", None)
    metadata = cast("object", getattr(inner, "yaml", {}))
    metadata = cast("dict[str, object]", metadata) if isinstance(metadata, dict) else {}
    keypoint_shape = metadata.get("kpt_shape")
    if task != "pose" or keypoint_shape not in ([29, 3], (29, 3)):
        raise FieldModelIncompatible(
            f"field exige task=pose e kpt_shape=[29, 3], recebeu task={task!r} "
            f"e kpt_shape={keypoint_shape!r}"
        )


def ultralytics_loader(quantize: str = PADRAO) -> ModelLoader:
    """O loader do backend com a precisao declarada ja fechada dentro dele.

    `ModelLoader` e `(path, device)` e a precisao nao cabe nessa assinatura: ela e do artefato e
    nao da chamada, e o `PyTorchBackend` mora em `edge.runtime`, que nao conhece especie nenhuma.
    """

    def load(path: Path, device: str) -> RuntimeModel:
        return _load_ultralytics(path, device, quantize)

    return load


def _load_ultralytics(path: Path, device: str, quantize: str = PADRAO) -> RuntimeModel:
    """A meia precisao mora aqui porque producao nao passa pelo `predict` deste arquivo.

    `from_weights` sempre monta o `PyTorchBackend`, e o `predict` acima so roda com modelo
    injetado. O binding declarado em `INPUT` e `OUTPUT` continua float32 nos dois casos, e a saida
    volta por `.float()`; o que vira meia precisao sao o peso e o tensor, dentro desta funcao.
    """
    ultralytics = cast("Any", import_module("ultralytics"))
    torch = cast("Any", import_module("torch"))
    yolo = ultralytics.YOLO(str(path))
    check_field_model(yolo)
    model = yolo.model.to(device).eval()
    half = quantize != PADRAO
    if half:
        model = model.half()

    def infer(inputs: Mapping[str, Array]) -> Mapping[str, Array]:
        tensor = torch.from_numpy(inputs["images"]).to(device)
        if half:
            tensor = tensor.half()
        with torch.inference_mode():
            output = model(tensor)
            prediction = cast("Any", output[0] if isinstance(output, tuple) else output)
            host = np.asarray(prediction.detach().float().cpu().numpy(), dtype=np.float32)
        return {"predictions": cast("Array", np.ascontiguousarray(host))}

    return infer


def _backend_de_peso(weights: str, device: str, quantize: str) -> InferenceBackend:
    path = Path(weights)
    backend = PyTorchBackend(
        provider="ultralytics",
        device=device,
        loader=ultralytics_loader(quantize),
        runtime_versions=(("torch", version("torch")), ("ultralytics", version("ultralytics"))),
    )
    backend.load(
        RuntimeArtifact(
            path=path,
            sha256=sha256_of(path),
            format="pt",
            parent_onnx_sha256=None,
            inputs=(INPUT,),
            outputs=(OUTPUT,),
        )
    )
    return backend


def _backend_de_plano(
    engine: str, engine_onnx_sha256: str, device: str, quantize: str
) -> InferenceBackend:
    """Recusa cedo o que so falharia na primeira inferencia, pelo mesmo motivo da bola.

    `quantize` junto de `engine` levanta em vez de ser ignorado: a precisao de um plano foi fixada
    quando ele foi construido, e aceitar o ajuste calado faria o TOML descrever algo que nao
    acontece — e no campo isso seria especialmente enganoso, porque `fp16` aqui PIORA o PyTorch.
    """
    from edge.modules.field.module import FieldConfigError
    from edge.trt import tensorrt_loader, tensorrt_version

    plano = Path(engine)
    if not plano.is_file():
        raise FieldConfigError(f"field.engine: {plano} nao existe")
    if re.fullmatch(r"[0-9a-f]{64}", engine_onnx_sha256) is None:
        raise FieldConfigError(
            "field.engine_onnx_sha256: um plano exige o sha256 do ONNX que o gerou, com 64 hex"
        )
    if quantize != PADRAO:
        raise FieldConfigError(
            f"field.quantize={quantize!r} nao se aplica a um plano: a precisao foi fixada "
            "quando ele foi construido"
        )
    backend = TensorRTBackend(
        provider="tensorrt",
        device=device,
        loader=tensorrt_loader(),
        runtime_versions=(("tensorrt", tensorrt_version()),),
    )
    backend.load(
        RuntimeArtifact(
            path=plano,
            sha256=sha256_of(plano),
            format="engine",
            parent_onnx_sha256=engine_onnx_sha256,
            inputs=(INPUT,),
            outputs=(OUTPUT,),
        )
    )
    return backend
