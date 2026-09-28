"""A espécie `identity`: equipes estáveis a partir das cores acumuladas por trajetória."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import cast

import numpy as np
import pytest

from edge.frames.preprocess import to_tensor_box
from edge.modules.base import FrameBatch, ModuleOutput, PerceptionModule
from edge.modules.conformance import ModuleConformance
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


class TestTeamIdentity(ModuleConformance):
    @pytest.fixture
    def module(self) -> PerceptionModule:
        return TeamIdentity()


def _lote(
    indices: tuple[int, ...],
    batch_de: Callable[[int], FrameBatch],
    inputs_de: Callable[[int], Mapping[str, ModuleOutput]],
) -> tuple[FrameBatch, dict[str, ModuleOutput]]:
    """Junta os quadros num lote só, que é como o runner entrega o clipe à identidade."""
    batches = [batch_de(indice) for indice in indices]
    entradas = [inputs_de(indice) for indice in indices]
    lote = FrameBatch(
        indices=indices,
        tensor=np.concatenate([batch.tensor for batch in batches]),
        preprocess=PREPROCESS,
        source_size=SOURCE_SIZE,
    )
    ctx = {
        kind: ModuleOutput(
            kind=kind,
            frames=tuple(f for entrada in entradas for f in entrada[kind].frames),
            records=tuple(r for entrada in entradas for r in entrada[kind].records),
        )
        for kind in entradas[0]
    }
    return lote, ctx


def _no_clipe(
    indices: tuple[int, ...],
    batch_de: Callable[[int], FrameBatch],
    inputs_de: Callable[[int], Mapping[str, ModuleOutput]],
) -> dict[int, Mapping[str, object]]:
    """Registros por trajetória no último quadro de um clipe entregue num lote só."""
    modulo = TeamIdentity(minimum=2)
    modulo.load()
    saida = modulo.infer(*_lote(indices, batch_de, inputs_de))
    return {cast(int, registro["track_id"]): registro for registro in saida.records[-1]}


def test_classifica_com_as_amostras_de_todos_os_quadros_do_lote() -> None:
    modulo = TeamIdentity(minimum=2)
    modulo.load()

    saida = modulo.infer(*_lote((0, 1), _batch, _inputs))

    for registros in saida.records:
        teams = {cast(int, registro["track_id"]): registro["team"] for registro in registros}
        assert set(teams) == {1, 2}
        assert set(teams.values()) == {"team_a", "team_b"}


def test_cada_lote_e_julgado_sozinho_sem_herdar_amostras_do_anterior() -> None:
    """O cache guarda a saída pela chave do lote, e ela só vale se a saída depender só dele."""
    modulo = TeamIdentity(minimum=2)
    modulo.load()
    modulo.infer(*_lote((0, 1), _batch, _inputs))

    saida = modulo.infer(_batch(2), _inputs(2))

    assert all("team" not in registro for registro in saida.records[0])


def test_a_mesma_entrada_com_jogadores_devolve_a_mesma_saida() -> None:
    modulo = TeamIdentity(minimum=2)
    modulo.load()
    lote, ctx = _lote((0, 1), _batch, _inputs)

    assert modulo.infer(lote, ctx) == modulo.infer(lote, ctx)


def test_ignora_deteccao_sem_track_sem_invalidar_as_outras() -> None:
    modulo = TeamIdentity(minimum=2)
    modulo.load()

    saida = modulo.infer(_batch(0), _inputs(0, tracking=({"detection_index": 1, "track_id": 2},)))

    assert saida.kind == "identity"
    assert saida.frames == (0,)
    assert saida.records == (({"track_id": 2},),)


CAIXA_DO_ARBITRO = CAIXAS[0]
"""O "árbitro" entra sobre a mesma caixa do primeiro jogador, e portanto com a mesma cor.

A trajetória não semeia o agrupamento, que fica só com jogadores, e como a cor dela é a de um time,
a resolução a trata como jogador que o detector marcou como árbitro, o erro que os rótulos do piloto
mostram em 4 de 15 trajetórias estáveis como árbitro.
"""


def _inputs_com_arbitro(index: int) -> dict[str, ModuleOutput]:
    papeis = ("player", "player", "referee")
    caixas = (*CAIXAS, CAIXA_DO_ARBITRO)
    return {
        "players": ModuleOutput(
            kind="players",
            frames=(index,),
            records=(
                tuple(
                    {"box": caixa, "confidence": 0.9, "role": papel}
                    for caixa, papel in zip(caixas, papeis, strict=True)
                ),
            ),
        ),
        "tracking": ModuleOutput(
            kind="tracking",
            frames=(index,),
            records=(
                tuple(
                    {"detection_index": posicao, "track_id": posicao + 1, "role": papel}
                    for posicao, papel in enumerate(papeis)
                ),
            ),
        ),
    }


def test_arbitro_com_a_cor_de_um_time_vira_jogador_desse_time() -> None:
    por_track = _no_clipe((0, 1), _batch, _inputs_com_arbitro)

    assert {por_track[1]["team"], por_track[2]["team"]} == {"team_a", "team_b"}
    assert por_track[3]["role"] == "player"
    assert por_track[3]["team"] == por_track[1]["team"]


CAIXAS_COM_ARBITRO = ((5.0, 10.0, 35.0, 90.0), (35.0, 10.0, 65.0, 90.0), (65.0, 10.0, 95.0, 90.0))
CORES_COM_ARBITRO = ((0, 0, 255), (255, 0, 0), (128, 128, 128))
"""Dois times, vermelho e azul, e a arbitragem de cinza, fora dos dois."""


def _batch_com_arbitro(index: int) -> FrameBatch:
    frame = np.zeros((PREPROCESS.height, PREPROCESS.width, 3), dtype=np.uint8)
    for caixa, cor in zip(CAIXAS_COM_ARBITRO, CORES_COM_ARBITRO, strict=True):
        _pinta_torso(frame, caixa, cor)
    return FrameBatch(
        indices=(index,),
        tensor=frame[np.newaxis, ...],
        preprocess=PREPROCESS,
        source_size=SOURCE_SIZE,
    )


def _inputs_arbitro_marcado_como_jogador(index: int) -> dict[str, ModuleOutput]:
    """O detector marca o árbitro como `player` em quase todo quadro, e como `referee` só no 1."""
    papeis = ("player", "player", "referee" if index == 1 else "player")
    return {
        "players": ModuleOutput(
            kind="players",
            frames=(index,),
            records=(
                tuple(
                    {"box": caixa, "confidence": 0.9, "role": papel}
                    for caixa, papel in zip(CAIXAS_COM_ARBITRO, papeis, strict=True)
                ),
            ),
        ),
        "tracking": ModuleOutput(
            kind="tracking",
            frames=(index,),
            records=(
                tuple(
                    {"detection_index": posicao, "track_id": posicao + 1, "role": "player"}
                    for posicao in range(3)
                ),
            ),
        ),
    }


def test_um_quadro_de_referee_no_clipe_recupera_o_arbitro() -> None:
    """O voto do tracking elege `player`; a evidência de um quadro e a cor própria corrigem."""
    por_track = _no_clipe((0, 1, 2), _batch_com_arbitro, _inputs_arbitro_marcado_como_jogador)

    assert por_track[3]["role"] == "referee"
    assert "team" not in por_track[3]
    assert {por_track[1]["team"], por_track[2]["team"]} == {"team_a", "team_b"}


ATE_O_GOL = (1.0, 0.0, 30.0, 0.0, 1.0, 90.0, 0.0, 0.0, 1.0)
"""Homografia de campo para imagem que leva o pé da terceira caixa, (80, 90), a (50, 0) no campo."""


def _inputs_com_goleiro(index: int, *, com_campo: bool) -> dict[str, ModuleOutput]:
    """O detector marca a terceira pessoa como goleiro só no quadro 1."""
    entradas = _inputs_arbitro_marcado_como_jogador(index)
    papeis = ("player", "player", "goalkeeper" if index == 1 else "player")
    entradas["players"] = ModuleOutput(
        kind="players",
        frames=(index,),
        records=(
            tuple(
                {"box": caixa, "confidence": 0.9, "role": papel}
                for caixa, papel in zip(CAIXAS_COM_ARBITRO, papeis, strict=True)
            ),
        ),
    )
    if com_campo:
        entradas["field"] = ModuleOutput(
            kind="field", frames=(index,), records=(({"homography": ATE_O_GOL},),)
        )
    return entradas


def test_com_campo_quem_tem_quadro_de_goleiro_na_pequena_area_vira_goleiro() -> None:
    sem_campo = _no_clipe(
        (0, 1, 2), _batch_com_arbitro, lambda i: _inputs_com_goleiro(i, com_campo=False)
    )
    com_campo = _no_clipe(
        (0, 1, 2), _batch_com_arbitro, lambda i: _inputs_com_goleiro(i, com_campo=True)
    )

    assert sem_campo[3]["role"] == "player"
    assert com_campo[3]["role"] == "goalkeeper"


ATRAS_DO_GOL = (1.0, 0.0, 20.0, 0.0, 1.0, 90.0, 0.0, 0.0, 1.0)
"""Leva o pé da terceira caixa a (60, 0), 7,5 m atrás da linha de fundo, e os outros ao campo."""


def _inputs_atras_do_gol(index: int) -> dict[str, ModuleOutput]:
    entradas = _inputs_com_goleiro(index, com_campo=True)
    entradas["field"] = ModuleOutput(
        kind="field", frames=(index,), records=(({"homography": ATRAS_DO_GOL},),)
    )
    return entradas


def test_quem_esta_fora_do_campo_sai_como_fora_do_jogo_sem_time() -> None:
    por_track = _no_clipe((0, 1, 2), _batch_com_arbitro, _inputs_atras_do_gol)

    assert por_track[3]["role"] == "off_pitch"
    assert "team" not in por_track[3]
    assert {por_track[1]["team"], por_track[2]["team"]} == {"team_a", "team_b"}


def test_dois_uniformes_parecidos_demais_nao_recebem_time() -> None:
    """Sem contraste entre os grupos, a trajetória fica sem time em vez de ganhar um errado."""
    frame = np.zeros((PREPROCESS.height, PREPROCESS.width, 3), dtype=np.uint8)
    for caixa, cor in zip(CAIXAS, ((0, 0, 255), (20, 20, 235)), strict=True):
        _pinta_torso(frame, caixa, cor)

    def batch_parecido(indice: int) -> FrameBatch:
        return FrameBatch(
            indices=(indice,),
            tensor=frame[np.newaxis, ...],
            preprocess=PREPROCESS,
            source_size=SOURCE_SIZE,
        )

    por_track = _no_clipe((0, 1), batch_parecido, _inputs)

    assert all("team" not in registro for registro in por_track.values())
