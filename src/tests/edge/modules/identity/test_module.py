"""A espécie `identity`: equipes estáveis a partir das cores acumuladas por trajetória."""

from __future__ import annotations

from typing import cast

import numpy as np

from edge.frames.preprocess import to_tensor_box
from edge.modules.base import FrameBatch, ModuleOutput
from edge.modules.identity import TeamIdentity
from edge.modules.players import PREPROCESS

SOURCE_SIZE = (100, 100)
"""Quadro de origem menor que o tensor, para o teste exercitar mesmo a conversão de caixa."""

CAIXAS = ((10.0, 10.0, 50.0, 90.0), (50.0, 10.0, 90.0, 90.0))
CORES = ((0, 0, 255), (255, 0, 0))


def _pinta_torso(
    frame: np.ndarray[tuple[int, int, int], np.dtype[np.uint8]],
    caixa_origem: tuple[float, float, float, float],
    cor: tuple[int, int, int],
) -> None:
    """Pinta a camisa dentro do recorte que `extract_shirt_color` vai ler naquela caixa.

    A caixa chega em pixel do quadro de origem, e o tensor de `players` é letterboxed: a camisa só
    cai sob o recorte se passar por `to_tensor_box`. Pintar pelo caminho ingênuo — regra de três
    sobre a largura — deixaria o torso fora do recorte, que é exatamente o erro que este teste
    precisa flagrar.
    """
    x1, y1, x2, y2 = to_tensor_box(caixa_origem, PREPROCESS, SOURCE_SIZE)
    largura, altura = x2 - x1, y2 - y1
    # As mesmas frações do extrator, para o retângulo cair dentro do recorte de torso.
    recorte_x1, recorte_x2 = x1 + 0.2 * largura, x2 - 0.2 * largura
    recorte_y1, recorte_y2 = y1 + 0.2 * altura, y1 + 0.55 * altura
    # Recuo em cada lado: o centro do recorte vira camisa e os cantos seguem fundo, que é como o
    # extrator decide qual dos dois grupos é o uniforme.
    recuo_x = 0.3 * (recorte_x2 - recorte_x1)
    recuo_y = 0.3 * (recorte_y2 - recorte_y1)
    frame[
        round(recorte_y1 + recuo_y) : round(recorte_y2 - recuo_y),
        round(recorte_x1 + recuo_x) : round(recorte_x2 - recuo_x),
    ] = cor


def _frame() -> np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]:
    """Dois torsos coloridos, cada um cercado de fundo para o extrator separá-lo."""
    frame = np.zeros((PREPROCESS.height, PREPROCESS.width, 3), dtype=np.uint8)
    for caixa, cor in zip(CAIXAS, CORES, strict=True):
        _pinta_torso(frame, caixa, cor)
    return frame


def _batch(index: int) -> FrameBatch:
    return FrameBatch(
        indices=(index,),
        tensor=_frame()[np.newaxis, ...],
        preprocess=PREPROCESS,
        source_size=SOURCE_SIZE,
    )


def _inputs(
    index: int, *, tracking: tuple[dict[str, object], ...] | None = None
) -> dict[str, ModuleOutput]:
    return {
        "players": ModuleOutput(
            kind="players",
            frames=(index,),
            records=(tuple({"box": box, "confidence": 0.9, "role": "player"} for box in CAIXAS),),
        ),
        "tracking": ModuleOutput(
            kind="tracking",
            frames=(index,),
            records=(
                tracking
                if tracking is not None
                else ({"detection_index": 0, "track_id": 1}, {"detection_index": 1, "track_id": 2}),
            ),
        ),
    }


def test_acumula_amostras_de_cor_entre_batches_antes_de_classificar() -> None:
    modulo = TeamIdentity(minimum=2)
    modulo.load()

    primeiro = modulo.infer(_batch(0), _inputs(0))
    segundo = modulo.infer(_batch(1), _inputs(1))

    assert all("team" not in registro for registro in primeiro.records[0])
    teams = {cast(int, registro["track_id"]): registro["team"] for registro in segundo.records[0]}
    assert set(teams) == {1, 2}
    assert set(teams.values()) == {"team_a", "team_b"}


def test_close_descarta_evidencia_do_clipe_anterior() -> None:
    modulo = TeamIdentity(minimum=2)
    modulo.load()
    modulo.infer(_batch(0), _inputs(0))
    modulo.infer(_batch(1), _inputs(1))
    modulo.close()
    modulo.load()

    saida = modulo.infer(_batch(0), _inputs(0))

    assert all("team" not in registro for registro in saida.records[0])


def test_ignora_deteccao_sem_track_sem_invalidar_as_outras() -> None:
    modulo = TeamIdentity(minimum=2)
    modulo.load()

    saida = modulo.infer(_batch(0), _inputs(0, tracking=({"detection_index": 1, "track_id": 2},)))

    assert saida.kind == "identity"
    assert saida.frames == (0,)
    assert saida.records == (({"track_id": 2},),)
