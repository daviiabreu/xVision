"""O adaptador do Ultralytics para jogadores, declarado na configuracao de producao.

Ele estava a 0% de cobertura, apesar de ser o `impl` que `pipeline.models.toml` aponta.

Ele nao interpreta a classe de proposito: quem sabe o que o numero significa e a `PlayersConfig`,
porque depende do peso (o COCO chama 0 de `person` e o peso de futebol chama 2 de `player`). Entao
o que se cobra aqui e a travessia: o que o Ultralytics devolve vira `Box`, confianca e classe, com
os tipos certos, e nada e filtrado no caminho.
"""

from __future__ import annotations

from typing import Any

import pytest

from contracts import Box
from edge.modules.players.yolo import UltralyticsPlayers


class _Lista:
    def __init__(self, valores: list[Any]) -> None:
        self._valores = valores

    def tolist(self) -> list[Any]:
        return self._valores


class _Caixas:
    def __init__(self, xyxy: list[list[float]], conf: list[float], cls: list[float]) -> None:
        self.xyxy = _Lista(xyxy)
        self.conf = _Lista(conf)
        self.cls = _Lista(cls)


class _Saida:
    def __init__(self, boxes: _Caixas | None) -> None:
        self.boxes = boxes


class _Modelo:
    def __init__(self, saidas: list[_Saida]) -> None:
        self._saidas = saidas
        self.chamadas: list[dict[str, Any]] = []

    def predict(self, **kwargs: Any) -> list[_Saida]:
        self.chamadas.append(kwargs)
        return self._saidas


def test_a_saida_do_yolo_vira_caixa_confianca_e_classe() -> None:
    modelo = _Modelo([_Saida(_Caixas([[1.0, 2.0, 3.0, 4.0]], [0.9], [2.0]))])

    deteccoes = UltralyticsPlayers(modelo, imgsz=640, conf=0.3, device="cpu").detect(object())

    assert deteccoes == ((Box(1.0, 2.0, 3.0, 4.0), 0.9, 2),)
    assert isinstance(deteccoes[0][2], int), "a classe tem de sair inteira, e o YOLO devolve float"


def test_a_configuracao_chega_inteira_ao_ultralytics() -> None:
    """Sem isto, trocar `imgsz` ou `device` na configuracao nao mudaria nada e ninguem veria."""
    modelo = _Modelo([])

    UltralyticsPlayers(modelo, imgsz=1280, conf=0.15, device="mps").detect(object())

    assert modelo.chamadas[0]["imgsz"] == 1280
    assert modelo.chamadas[0]["conf"] == 0.15
    assert modelo.chamadas[0]["device"] == "mps"
    assert modelo.chamadas[0]["verbose"] is False, "a barra de progresso suja o log do worker"


def test_saida_sem_caixa_nao_derruba_a_travessia() -> None:
    """O Ultralytics devolve saida sem `boxes` quando nao achou nada, e isso nao e erro."""
    modelo = _Modelo([_Saida(None), _Saida(_Caixas([[0.0, 0.0, 1.0, 1.0]], [0.5], [0.0]))])

    deteccoes = UltralyticsPlayers(modelo, imgsz=640, conf=0.3, device="cpu").detect(object())

    assert len(deteccoes) == 1


def test_todo_quadro_da_resposta_e_lido() -> None:
    """Ler so a primeira saida perderia deteccao quando o lote tem mais de um quadro."""
    modelo = _Modelo(
        [
            _Saida(_Caixas([[0.0, 0.0, 1.0, 1.0]], [0.5], [0.0])),
            _Saida(_Caixas([[2.0, 2.0, 3.0, 3.0], [4.0, 4.0, 5.0, 5.0]], [0.6, 0.7], [2.0, 2.0])),
        ]
    )

    deteccoes = UltralyticsPlayers(modelo, imgsz=640, conf=0.3, device="cpu").detect(object())

    assert len(deteccoes) == 3


def test_from_weights_nao_importa_ultralytics_no_topo(monkeypatch: pytest.MonkeyPatch) -> None:
    """O import e tardio porque `ultralytics` puxa torch, e a conformidade instancia sem peso."""
    chamado: list[str] = []

    class _Yolo:
        def __init__(self, peso: str) -> None:
            chamado.append(peso)

    def _modulo_falso(_nome: str) -> Any:
        return type("m", (), {"YOLO": _Yolo})

    monkeypatch.setattr("edge.modules.players.yolo.import_module", _modulo_falso)

    adaptador = UltralyticsPlayers.from_weights("peso.pt", imgsz=640, conf=0.3, device="cpu")

    assert chamado == ["peso.pt"]
    assert isinstance(adaptador, UltralyticsPlayers)
