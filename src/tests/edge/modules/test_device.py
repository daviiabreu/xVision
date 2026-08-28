"""Onde a inferencia roda, que muda entre as tres maquinas do time e a Jetson.

Cravar acelerador quebra alguem: `cuda` quebra o Mac, `mps` quebra o Linux, `cpu` desperdica a
placa. Estes testes fixam a ordem de decisao sem exigir torch instalado, porque o CI nao tem
placa nenhuma e a borda sem extra de modelo nao carrega torch.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from edge.modules.device import DEVICE_ENV, preferred_device


def _torch_falso(*, cuda: bool, mps: bool) -> types.ModuleType:
    modulo = types.ModuleType("torch")
    modulo.cuda = types.SimpleNamespace(is_available=lambda: cuda)  # type: ignore[attr-defined]
    modulo.backends = types.SimpleNamespace(  # type: ignore[attr-defined]
        mps=types.SimpleNamespace(is_available=lambda: mps)
    )
    return modulo


def test_a_variavel_de_ambiente_ganha_de_qualquer_placa(monkeypatch: pytest.MonkeyPatch) -> None:
    """Quem opera precisa forcar sem editar arquivo, e e assim que a suite roda igual no CI."""
    monkeypatch.setitem(sys.modules, "torch", _torch_falso(cuda=True, mps=True))
    monkeypatch.setenv(DEVICE_ENV, "cpu")

    assert preferred_device() == "cpu"


def test_sem_torch_instalado_o_piso_e_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """A borda sem extra de modelo nao tem torch, e ler um caminho nao pode exigir a pilha."""
    monkeypatch.delenv(DEVICE_ENV, raising=False)

    def recusa(nome: str) -> Any:
        raise ImportError(nome)

    monkeypatch.setattr("edge.modules.device.import_module", recusa)

    assert preferred_device() == "cpu"


@pytest.mark.parametrize(
    ("cuda", "mps", "esperado"),
    [(True, False, "cuda"), (False, True, "mps"), (False, False, "cpu"), (True, True, "cuda")],
)
def test_o_acelerador_so_entra_quando_o_torch_confirma(
    monkeypatch: pytest.MonkeyPatch, cuda: bool, mps: bool, esperado: str
) -> None:
    """`cuda` ganha de `mps` quando os dois respondem, e a ausencia dos dois cai em `cpu`."""
    monkeypatch.delenv(DEVICE_ENV, raising=False)
    monkeypatch.setitem(sys.modules, "torch", _torch_falso(cuda=cuda, mps=mps))

    assert preferred_device() == esperado
