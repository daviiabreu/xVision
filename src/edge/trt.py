"""Loader de engine TensorRT, isolado de `edge.runtime` e carregado so na placa.

Mora aqui, e nao junto do `TensorRTBackend`, pelo mesmo motivo que o adapter do Ultralytics mora
na especie: `edge.runtime` guarda o contrato e e importado pelo venv de teste, que nao tem nem
`tensorrt` nem `torch`. Este modulo so e importado por quem vai de fato executar na Jetson.

Sobre a versao: um engine NAO atravessa versao de TensorRT nem placa. Nesta maquina convivem o
`trtexec` do apt (10.16) e o binding do pip `tensorrt_cu13` (11.3), e um plano gerado por um nao
abre no outro — a falha e um `magicTag` que nao casa, no momento de desserializar. Por isso o
`RuntimeInfo.runtime_versions` registra a versao usada: sem ela, um engine que para de carregar
depois de um `uv sync` vira investigacao em vez de mensagem.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

import numpy as np

from edge.runtime import Array, ArtifactIncompatible, ModelLoader, RuntimeModel, RuntimeUnavailable

_TORCH_POR_TRT = {"HALF": "float16", "FLOAT": "float32"}


DISTRIBUICOES = ("tensorrt-cu13", "tensorrt-cu12", "tensorrt")
"""Os nomes com que o binding chega pelo pip, na ordem em que sao procurados.

O pip da Jetson instala `tensorrt-cu13`, e nao `tensorrt`: perguntar a versao de `tensorrt`
levantava `PackageNotFoundError` e derrubava `ball` e `field` em todo clipe, medido em 24/09/2026.
"""


def tensorrt_version() -> str:
    """A versao do TensorRT instalado, para o `run.json` dizer qual abriu o plano."""
    for nome in DISTRIBUICOES:
        try:
            return version(nome)
        except PackageNotFoundError:
            continue
    raise RuntimeUnavailable(f"nenhum pacote de TensorRT instalado: {', '.join(DISTRIBUICOES)}")


def tensorrt_loader() -> ModelLoader:
    """Devolve um loader que abre o plano e entrega um callable de numpy para numpy.

    Os buffers do device sao alocados uma vez, no load, porque o engine tem shape fixo: realocar
    por chamada custaria mais que a copia e nao compraria nada. A copia para o device e de volta
    fica aqui de proposito — e o que deixa o `TensorRTBackend` falar o mesmo numpy que o
    `PyTorchBackend`, e nesta Jetson a memoria e unificada, entao a viagem e barata.
    """

    def loader(path: Path, device: str) -> RuntimeModel:
        trt = cast("Any", import_module("tensorrt"))
        torch = cast("Any", import_module("torch"))

        logger = trt.Logger(trt.Logger.ERROR)
        engine = trt.Runtime(logger).deserialize_cuda_engine(path.read_bytes())
        if engine is None:
            raise ArtifactIncompatible(
                f"{path.name} nao desserializa com TensorRT {trt.__version__}. Engine nao "
                "atravessa versao nem placa: reconstrua o plano nesta maquina"
            )
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeUnavailable(f"{path.name} abriu mas nao criou contexto de execucao")

        entradas: dict[str, Any] = {}
        saidas: dict[str, Any] = {}
        for indice in range(engine.num_io_tensors):
            nome = engine.get_tensor_name(indice)
            bruto = trt.nptype(engine.get_tensor_dtype(nome))
            tensor = torch.empty(
                tuple(engine.get_tensor_shape(nome)),
                dtype=getattr(torch, _TORCH_POR_TRT[engine.get_tensor_dtype(nome).name]),
                device=device,
            )
            context.set_tensor_address(nome, tensor.data_ptr())
            destino = entradas if engine.get_tensor_mode(nome) == trt.TensorIOMode.INPUT else saidas
            destino[nome] = (tensor, np.dtype(bruto))

        stream = torch.cuda.Stream(device=device)

        def model(inputs: Mapping[str, Array]) -> Mapping[str, Array]:
            faltando = sorted(set(entradas) - set(inputs))
            if faltando:
                raise ArtifactIncompatible(
                    f"o engine pede {sorted(entradas)}, faltou {', '.join(faltando)}"
                )
            for nome, (tensor, dtype) in entradas.items():
                tensor.copy_(torch.from_numpy(np.ascontiguousarray(inputs[nome], dtype=dtype)))
            if not context.execute_async_v3(stream.cuda_stream):
                raise RuntimeUnavailable("execute_async_v3 recusou o despacho")
            stream.synchronize()
            return {nome: tensor.cpu().numpy() for nome, (tensor, _) in saidas.items()}

        return model

    return loader
