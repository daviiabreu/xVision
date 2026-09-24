"""A versao do TensorRT sai do pacote que o pip de fato instalou."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError

import pytest

from edge import trt
from edge.runtime import RuntimeUnavailable


def _instalado(monkeypatch: pytest.MonkeyPatch, pacotes: dict[str, str]) -> None:
    def versao(nome: str) -> str:
        if nome not in pacotes:
            raise PackageNotFoundError(nome)
        return pacotes[nome]

    monkeypatch.setattr(trt, "version", versao)


def test_o_pacote_da_jetson_e_o_tensorrt_cu13(monkeypatch: pytest.MonkeyPatch) -> None:
    _instalado(monkeypatch, {"tensorrt-cu13": "11.3.0.99"})

    assert trt.tensorrt_version() == "11.3.0.99"


def test_o_pacote_generico_tambem_serve(monkeypatch: pytest.MonkeyPatch) -> None:
    _instalado(monkeypatch, {"tensorrt": "10.3.0"})

    assert trt.tensorrt_version() == "10.3.0"


def test_sem_tensorrt_a_mensagem_diz_o_que_foi_procurado(monkeypatch: pytest.MonkeyPatch) -> None:
    _instalado(monkeypatch, {})

    with pytest.raises(RuntimeUnavailable, match="tensorrt-cu13"):
        trt.tensorrt_version()
