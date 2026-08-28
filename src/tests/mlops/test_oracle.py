"""O oraculo de instante: devolve o quadro anotado, e nada quando nao ha anotacao.

Ele existe para separar dois erros que o numero final confunde, entao o modo de falha caro nao e
errar o quadro: e inventar um. Um oraculo que estimasse quando falta anotacao poria de volta no
experimento exatamente o erro que ele foi feito para eliminar, e a comparacao com o detector
mediria os dois misturados de novo.
"""

from __future__ import annotations

import numpy as np
import pytest

from edge.modules.base import FrameBatch, ModuleNotLoaded, PreprocessSpec
from mlops.oracle import PREPROCESS, AnnotatedShot


def _lote(indices: tuple[int, ...], clip_id: str | None, spec: PreprocessSpec = PREPROCESS):
    return FrameBatch(
        indices=indices,
        tensor=np.zeros((len(indices), spec.height, spec.width, 3), dtype=np.uint8),
        preprocess=spec,
        clip_id=clip_id,
    )


def _pronto(gabarito: dict[str, int]) -> AnnotatedShot:
    modulo = AnnotatedShot(gabarito)
    modulo.load()
    return modulo


def test_devolve_o_quadro_anotado() -> None:
    saida = _pronto({"c1": 207}).infer(_lote((0, 60, 120, 180, 240), "c1"), {})

    registros = [r for grupo in saida.records for r in grupo]
    assert len(registros) == 1
    assert registros[0]["frame_index"] == 207
    assert registros[0]["source"] == "gabarito"


def test_o_registro_viaja_no_quadro_amostrado_mais_proximo() -> None:
    """`_anchor_frame` le o `frame_index` do registro, mas a posicao ainda ordena a saida."""
    saida = _pronto({"c1": 207}).infer(_lote((0, 60, 120, 180, 240), "c1"), {})

    portador = [i for i, grupo in enumerate(saida.records) if grupo]
    assert portador == [3]


def test_clipe_sem_anotacao_nao_estima() -> None:
    """Estimar poria de volta o erro de instante que o experimento existe para eliminar."""
    saida = _pronto({"c1": 207}).infer(_lote((0, 60, 120), "c2"), {})

    assert saida.frames == (0, 60, 120)
    assert all(not grupo for grupo in saida.records)


def test_lote_sem_clip_id_tambem_nao_estima() -> None:
    """Lote montado a mao nao carrega identidade, e isso nao pode virar um quadro chutado."""
    saida = _pronto({"c1": 207}).infer(_lote((0, 60), None), {})

    assert all(not grupo for grupo in saida.records)


def test_nao_depende_de_bola_nem_de_jogador() -> None:
    """Depender delas faria uma falha de percepcao mudar o resultado por um motivo alheio."""
    assert AnnotatedShot().spec.needs == ()
    assert AnnotatedShot().spec.optional_needs == ()


def test_infer_antes_de_load_falha() -> None:
    with pytest.raises(ModuleNotLoaded):
        AnnotatedShot({"c1": 1}).infer(_lote((0,), "c1"), {})


def test_lote_vazio_devolve_saida_vazia() -> None:
    saida = _pronto({"c1": 207}).infer(_lote((), "c1"), {})

    assert saida.frames == ()
    assert saida.records == ()


def test_close_desarma_o_modulo() -> None:
    modulo = _pronto({"c1": 207})
    modulo.close()

    with pytest.raises(ModuleNotLoaded):
        modulo.infer(_lote((0,), "c1"), {})
