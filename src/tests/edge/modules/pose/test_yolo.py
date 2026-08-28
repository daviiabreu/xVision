"""O adaptador do estimador de pose, que a configuracao de producao declara e ninguem media.

Um recorte tem uma pessoa so por construcao, entao a escolha do corpo nao pode ficar na mao da
ordem interna do modelo: o adaptador devolve o de maior confianca. E o que estes testes cobram,
junto com o que acontece quando o estimador nao acha ninguem no recorte.
"""

from __future__ import annotations

from typing import Any

import pytest

from edge.modules.pose.yolo import CONF, IMGSZ, YoloPoseEstimator


class _Tensor:
    def __init__(self, valores: list[Any]) -> None:
        self._valores = valores

    def tolist(self) -> list[Any]:
        return self._valores

    def __len__(self) -> int:
        return len(self._valores)

    def __getitem__(self, indice: int) -> _Tensor:
        return _Tensor(self._valores[indice])


class _Saida:
    def __init__(self, corpos: list[list[list[float]]] | None, confiancas: list[float]) -> None:
        self.keypoints = None if corpos is None else type("k", (), {"data": _Tensor(corpos)})()
        self.boxes = None if corpos is None else type("b", (), {"conf": _Tensor(confiancas)})()


class _Modelo:
    def __init__(self, saidas: list[_Saida]) -> None:
        self._saidas = saidas
        self.chamadas: list[dict[str, Any]] = []

    def predict(self, **kwargs: Any) -> list[_Saida]:
        self.chamadas.append(kwargs)
        return self._saidas


def _corpo(x: float) -> list[list[float]]:
    return [[x, x, 0.9] for _ in range(17)]


def test_lote_vazio_nao_chama_o_modelo() -> None:
    """Chamar o estimador com lista vazia gasta uma passada de GPU para nada."""
    modelo = _Modelo([])

    assert YoloPoseEstimator(modelo).estimate_many([]) == ()
    assert modelo.chamadas == []


def test_o_corpo_escolhido_e_o_de_maior_confianca() -> None:
    """Devolver o primeiro da lista deixaria a escolha na ordem interna do modelo."""
    modelo = _Modelo([_Saida([_corpo(1.0), _corpo(9.0)], [0.2, 0.8])])

    keypoints = YoloPoseEstimator(modelo).estimate_many([object()])

    assert keypoints[0] is not None
    nome, x, _y, _conf = keypoints[0][0]
    assert nome == "nose", "o primeiro keypoint do COCO e o nariz, e a ordem e contrato"
    assert x == 9.0, "veio o corpo de confianca 0,2 em vez do de 0,8"


def test_recorte_sem_ninguem_devolve_none() -> None:
    """`None` e o contrato para 'nao achei', e nao uma pose de zeros."""
    resultado = YoloPoseEstimator(_Modelo([_Saida(None, [])])).estimate_many([object()])

    assert resultado == (None,)


def test_a_resposta_tem_um_item_por_recorte() -> None:
    """O chamador casa recorte com pose por indice, entao o tamanho tem de bater."""
    modelo = _Modelo([_Saida([_corpo(1.0)], [0.9]), _Saida(None, []), _Saida([_corpo(2.0)], [0.7])])

    assert len(YoloPoseEstimator(modelo).estimate_many([object()] * 3)) == 3


def test_os_padroes_declarados_sao_os_usados(monkeypatch: pytest.MonkeyPatch) -> None:
    """O recorte de um jogador de transmissao e pequeno, e o `IMGSZ` amplia sem inventar."""
    monkeypatch.setattr("edge.modules.device.preferred_device", lambda: "cpu")
    modelo = _Modelo([_Saida([_corpo(1.0)], [0.9])])

    YoloPoseEstimator(modelo).estimate_many([object()])

    assert modelo.chamadas[0]["imgsz"] == IMGSZ
    assert modelo.chamadas[0]["conf"] == CONF
    assert modelo.chamadas[0]["verbose"] is False
