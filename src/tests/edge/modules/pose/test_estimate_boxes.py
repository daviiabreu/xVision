"""O seam por onde `xvision eval-pose` entra: keypoints de uma caixa dada, sem escolher o corpo.

Estes testes cobram a margem e a volta de coordenada aqui, e nao so pelo caminho de `infer`, porque
e por aqui que o avaliador entra: margem divergente faria o relatorio medir um enquadramento que a
producao nao usa.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from contracts import Box
from edge.modules.base import FrameBatch, ModuleNotLoaded
from edge.modules.pose import COCO_KEYPOINTS, LOTE, MARGEM, PREPROCESS, Keypoints, YoloPose

FONTE = (1920, 1080)


class _EstimadorFalso:
    """Devolve o canto do recorte como keypoint, para o teste seguir a volta de coordenada."""

    def __init__(self, *, responde: bool = True) -> None:
        self.recortes: list[Any] = []
        self.chamadas = 0
        self._responde = responde

    def estimate_many(self, crops: Sequence[Any]) -> tuple[Keypoints | None, ...]:
        self.chamadas += 1
        self.recortes.extend(crops)
        if not self._responde:
            return tuple(None for _ in crops)
        return tuple(tuple((nome, 0.0, 0.0, 0.9) for nome in COCO_KEYPOINTS) for _ in crops)


def _lote(indices: tuple[int, ...], fonte: tuple[int, int] | None = FONTE) -> FrameBatch:
    tensor = np.zeros((len(indices), PREPROCESS.height, PREPROCESS.width, 3), dtype=np.uint8)
    return FrameBatch(indices=indices, tensor=tensor, preprocess=PREPROCESS, source_size=fonte)


def _modulo(estimador: _EstimadorFalso | None = None) -> YoloPose:
    modulo = YoloPose(estimador if estimador is not None else _EstimadorFalso())
    modulo.load()
    return modulo


def test_sem_load_levanta_em_vez_de_devolver_vazio() -> None:
    """Vazio seria indistinguivel de 'nao achei corpo', e o avaliador registraria ausencia."""
    with pytest.raises(ModuleNotLoaded):
        YoloPose(_EstimadorFalso()).estimate_boxes(_lote((0,)), {0: Box(100, 200, 140, 330)})


def test_devolve_um_conjunto_de_keypoints_por_quadro_com_caixa() -> None:
    modulo = _modulo()

    medido = modulo.estimate_boxes(_lote((3, 4, 5)), {3: Box(100, 200, 140, 330)})

    assert set(medido) == {3}
    assert len(medido[3]) == len(COCO_KEYPOINTS)
    assert tuple(nome for nome, *_ in medido[3]) == COCO_KEYPOINTS


def test_quadro_sem_caixa_nao_aparece_no_resultado() -> None:
    modulo = _modulo()

    medido = modulo.estimate_boxes(
        _lote((3, 4)), {3: Box(100, 200, 140, 330), 4: Box(200, 300, 240, 430)}
    )

    assert set(medido) == {3, 4}
    assert set(modulo.estimate_boxes(_lote((3, 4)), {})) == set()


def test_o_keypoint_volta_para_o_pixel_do_quadro_original() -> None:
    """O falso marca o canto do recorte, entao o ponto tem de voltar perto do canto da caixa
    menos a margem, e nao na origem do tensor."""
    modulo = _modulo()
    caixa = Box(800.0, 400.0, 840.0, 530.0)

    medido = modulo.estimate_boxes(_lote((7,)), {7: caixa})

    _nome, x, y, _conf = medido[7][0]
    assert x < caixa.x1, "o canto do recorte fica a esquerda da caixa por causa da margem"
    assert y < caixa.y1
    assert x == pytest.approx(caixa.x1 - (caixa.x2 - caixa.x1) * MARGEM, abs=4.0)
    assert y == pytest.approx(caixa.y1 - (caixa.y2 - caixa.y1) * MARGEM, abs=4.0)


def test_caixa_degenerada_nao_vira_recorte() -> None:
    estimador = _EstimadorFalso()
    modulo = _modulo(estimador)

    medido = modulo.estimate_boxes(_lote((1,)), {1: Box(10.0, 10.0, 10.0, 10.0)})

    assert medido == {}
    assert estimador.chamadas == 0


def test_o_estimador_que_nao_acha_ninguem_nao_inventa_quadro() -> None:
    modulo = _modulo(_EstimadorFalso(responde=False))

    assert modulo.estimate_boxes(_lote((1,)), {1: Box(100, 200, 140, 330)}) == {}


def test_os_recortes_vao_em_lote() -> None:
    estimador = _EstimadorFalso()
    modulo = _modulo(estimador)
    indices = tuple(range(LOTE + 5))

    medido = modulo.estimate_boxes(_lote(indices), {i: Box(100, 200, 140, 330) for i in indices})

    assert len(medido) == len(indices)
    assert estimador.chamadas == 2, f"{len(indices)} recortes em lotes de {LOTE} sao duas chamadas"


def test_sem_estimador_carregado_o_resultado_e_vazio_e_nao_estoura() -> None:
    """Configuracao sem peso e caminho legitimo: a suite de conformidade instancia assim."""
    modulo = YoloPose(None)
    modulo.load()

    assert modulo.estimate_boxes(_lote((1,)), {1: Box(100, 200, 140, 330)}) == {}
