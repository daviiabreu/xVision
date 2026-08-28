"""O cache de inferencia: uma passada por quadro, mesmo com duas especies no mesmo peso.

O desperdicio que ele corta foi medido: `ball` e `players` usam o mesmo peso de 137 MB sobre o
mesmo conjunto de quadros, entao o clipe mediano roda 105 quadros duas vezes com a GPU em 95%.

O modo de falha que os testes fecham e o oposto e mais caro: cache que devolve a saida de um peso
quando quem pediu foi outro. Isso nao levanta nada, e a especie passa a medir a deteccao errada.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from contracts import Box
from edge.modules.base import PreprocessSpec
from edge.modules.inference import Deteccao, SharedInference, reset, shared

A = PreprocessSpec(width=64, height=64, color="bgr", letterbox=True, normalize="none")
B = PreprocessSpec(width=128, height=128, color="bgr", letterbox=True, normalize="none")


def _conta(registro: list[int]) -> Callable[[], tuple[Deteccao, ...]]:
    def calcular() -> tuple[Deteccao, ...]:
        registro.append(1)
        return ()

    return calcular


def _cache() -> SharedInference:
    reset()
    return shared()


def test_a_segunda_chamada_com_a_mesma_entrada_nao_recalcula() -> None:
    cache = _cache()
    chamadas: list[int] = []

    for _ in range(2):
        cache.get(weights="p.pt", preprocess=A, frame_index=4, compute=_conta(chamadas))

    assert len(chamadas) == 1
    assert (cache.hits, cache.misses) == (1, 1)


def test_peso_diferente_e_computacao_diferente() -> None:
    """Devolver a saida de um peso para quem pediu outro nao levanta nada, e mede a coisa errada."""
    cache = _cache()

    cache.get(weights="a.pt", preprocess=A, frame_index=0, compute=tuple)
    cache.get(weights="b.pt", preprocess=A, frame_index=0, compute=tuple)

    assert cache.misses == 2
    assert cache.hits == 0


def test_pre_processamento_diferente_e_computacao_diferente() -> None:
    """A caixa sai no espaco do tensor, entao resolucao diferente da coordenada diferente."""
    cache = _cache()

    cache.get(weights="a.pt", preprocess=A, frame_index=0, compute=tuple)
    cache.get(weights="a.pt", preprocess=B, frame_index=0, compute=tuple)

    assert cache.misses == 2


def test_quadro_diferente_e_computacao_diferente() -> None:
    cache = _cache()

    cache.get(weights="a.pt", preprocess=A, frame_index=0, compute=tuple)
    cache.get(weights="a.pt", preprocess=A, frame_index=1, compute=tuple)

    assert cache.misses == 2


def test_o_valor_guardado_e_o_que_volta_na_segunda_chamada() -> None:
    cache = _cache()
    esperado: tuple[Deteccao, ...] = ((Box(0.0, 0.0, 4.0, 4.0), 0.9, 0),)

    primeiro = cache.get(weights="a.pt", preprocess=A, frame_index=0, compute=lambda: esperado)
    segundo = cache.get(weights="a.pt", preprocess=A, frame_index=0, compute=tuple)

    assert primeiro == esperado
    assert segundo == esperado


def test_clear_esvazia_e_a_proxima_chamada_recalcula() -> None:
    cache = _cache()
    cache.get(weights="a.pt", preprocess=A, frame_index=0, compute=tuple)

    cache.clear()
    cache.get(weights="a.pt", preprocess=A, frame_index=0, compute=tuple)

    assert cache.size == 1
    assert cache.misses == 2


def test_reset_troca_a_instancia_do_processo() -> None:
    """Sem isto, um teste que meca `hits` depende do que os testes anteriores rodaram."""
    primeiro = _cache()
    primeiro.get(weights="a.pt", preprocess=A, frame_index=0, compute=tuple)

    segundo = _cache()

    assert segundo is not primeiro
    assert segundo.size == 0


def test_o_cache_nao_atravessa_clipes() -> None:
    """A chave nao tem identidade de clipe, entao quem o descarta e o `run_pipeline`.

    Sem o descarte o quadro 92 de um clipe serve o quadro 92 do proximo. Medido antes da correcao:
    seis clipes de duracoes diferentes devolveram o mesmo instante, todos vindos do primeiro.
    """
    from contracts import Clip
    from edge.frames.provider import FrameProvider
    from edge.graph.framesets import UniformFrames
    from edge.graph.runner import run_pipeline
    from edge.modules.stubs import STUBS

    cache = _cache()
    cache.get(weights="a.pt", preprocess=A, frame_index=92, compute=tuple)
    assert cache.size == 1

    clip = Clip(clip_id="c", path="c.mp4", fps=10.0, frame_count=8, width=64, height=64)
    modulos = {m.spec.kind: m() for m in STUBS}
    run_pipeline(
        clip=clip,
        modules=modulos,
        framesets={"coarse": UniformFrames(5.0), "window": UniformFrames(5.0)},
        provider=FrameProvider(_FonteVazia()),
    )

    assert shared().size == 0


class _FonteVazia:
    """Devolve quadro preto, porque o teste mede o descarte do cache e nao a deteccao."""

    def read(self, indices: Sequence[int]) -> dict[int, Any]:
        return {i: np.zeros((64, 64, 3), dtype=np.uint8) for i in indices}

    def close(self) -> None: ...
