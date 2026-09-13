"""Um Ultralytics de mentira para os loaders que o venv de teste nao consegue carregar.

`ball` e `field` nao rodam pelo `predict` em producao: as duas montam o `PyTorchBackend`, e a
precisao de inferencia delas e decidida dentro do loader, chamando o peso direto. Esse caminho
importa `torch` e `ultralytics` tarde, por nome, e nenhum dos dois existe no venv de teste de
proposito.

O fake registra em que precisao o peso e o tensor entraram, que e exatamente o que a alavanca de
`quantize` precisa provar: sem declaracao nada e convertido, e com `fp16` os dois sao.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import nullcontext
from importlib import import_module
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


class TensorDeMentira:
    """So o que os loaders usam: mover de dispositivo, meia precisao e voltar para numpy."""

    def __init__(self, valor: Any, *, meio: bool = False) -> None:
        self.valor = valor
        self.meio = meio

    def to(self, _device: str) -> TensorDeMentira:
        return self

    def half(self) -> TensorDeMentira:
        return TensorDeMentira(self.valor, meio=True)

    def detach(self) -> TensorDeMentira:
        return self

    def float(self) -> TensorDeMentira:
        return TensorDeMentira(self.valor)

    def cpu(self) -> TensorDeMentira:
        return self

    def numpy(self) -> Any:
        return self.valor


class PesoDeMentira:
    """O objeto que o `ultralytics.YOLO` devolveria, e tambem o `.model` de dentro dele."""

    def __init__(self, *, task: str, names: Mapping[int, str], yaml: Mapping[str, object]) -> None:
        self.task = task
        self.names = dict(names)
        self.yaml = dict(yaml)
        self.model = self
        self.dispositivo = ""
        self.meio_peso = False
        self.entradas_meias: list[bool] = []

    def to(self, device: str) -> PesoDeMentira:
        self.dispositivo = device
        return self

    def eval(self) -> PesoDeMentira:
        return self

    def half(self) -> PesoDeMentira:
        self.meio_peso = True
        return self

    def __call__(self, tensor: TensorDeMentira) -> TensorDeMentira:
        self.entradas_meias.append(tensor.meio)
        return TensorDeMentira(np.zeros((1, 5, 4), dtype=np.float32))


def _torch_de_mentira() -> SimpleNamespace:
    """`inference_mode` so precisa ser um gerenciador de contexto que nao faz nada."""

    def from_numpy(valor: Any) -> TensorDeMentira:
        return TensorDeMentira(valor)

    return SimpleNamespace(from_numpy=from_numpy, inference_mode=nullcontext)


@pytest.fixture
def peso_de_mentira(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., PesoDeMentira]:
    def montar(
        modulo: str,
        *,
        task: str = "detect",
        names: Mapping[int, str] | None = None,
        yaml: Mapping[str, object] | None = None,
    ) -> PesoDeMentira:
        peso = PesoDeMentira(task=task, names=names or {0: "ball"}, yaml=yaml or {})

        def abrir(_caminho: str) -> PesoDeMentira:
            return peso

        falsos = {"torch": _torch_de_mentira(), "ultralytics": SimpleNamespace(YOLO=abrir)}

        def importar(nome: str) -> Any:
            return falsos[nome] if nome in falsos else import_module(nome)

        monkeypatch.setattr(f"{modulo}.import_module", importar)
        return peso

    return montar
