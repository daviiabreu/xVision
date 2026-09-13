"""A especie `tracking`: o mesmo numero para o mesmo jogador, e numero novo para jogador novo.

O modo de falha que estes testes fecham e trocar dois jogadores de identidade entre quadros. Ele
nao levanta excecao: a pose vai para o corpo errado, e uma feature que meca orientacao do
finalizador passa a medir a do marcador, com a mesma cara de uma medida boa.
"""

from __future__ import annotations

import numpy as np
import pytest

from contracts import Box
from edge.modules.base import FrameBatch, ModuleNotLoaded, ModuleOutput, PerceptionModule
from edge.modules.conformance import ModuleConformance
from edge.modules.records import check_records
from edge.modules.tracking import PREPROCESS, GreedyTracking


def _lote(indices: tuple[int, ...]) -> FrameBatch:
    tensor = np.zeros((len(indices), PREPROCESS.height, PREPROCESS.width, 1), dtype=np.uint8)
    return FrameBatch(indices=indices, tensor=tensor, preprocess=PREPROCESS)


def _players(por_quadro: dict[int, list[tuple[float, float]]], lado: float = 40.0) -> ModuleOutput:
    """Um jogador por canto declarado, com caixa de `lado` px centrada nele."""
    quadros = tuple(sorted(por_quadro))
    return ModuleOutput(
        kind="players",
        frames=quadros,
        records=tuple(
            tuple(
                {"box": (x, y, x + lado, y + lado), "confidence": 0.9}
                for x, y in por_quadro[quadro]
            )
            for quadro in quadros
        ),
    )


def _players_com_papeis(
    por_quadro: dict[int, list[tuple[float, float, str, float]]], lado: float = 40.0
) -> ModuleOutput:
    quadros = tuple(sorted(por_quadro))
    return ModuleOutput(
        kind="players",
        frames=quadros,
        records=tuple(
            tuple(
                {
                    "box": (x, y, x + lado, y + lado),
                    "confidence": confidence,
                    "role": role,
                }
                for x, y, role, confidence in por_quadro[quadro]
            )
            for quadro in quadros
        ),
    )


def _ids(saida: ModuleOutput) -> list[list[int]]:
    return [[int(r["track_id"]) for r in registros] for registros in saida.records]  # type: ignore[arg-type]


class TestGreedyTracking(ModuleConformance):
    @pytest.fixture
    def module(self) -> PerceptionModule:
        return GreedyTracking()


def test_jogador_que_anda_devagar_mantem_o_numero() -> None:
    modulo = GreedyTracking()
    modulo.load()
    caminho = {q: [(100.0 + q * 5.0, 200.0)] for q in (0, 1, 2, 3)}

    saida = modulo.infer(_lote((0, 1, 2, 3)), {"players": _players(caminho)})

    assert _ids(saida) == [[1], [1], [1], [1]]


def test_jogador_novo_ganha_numero_novo() -> None:
    modulo = GreedyTracking()
    modulo.load()
    caminho = {0: [(100.0, 200.0)], 1: [(100.0, 200.0), (600.0, 200.0)]}

    saida = modulo.infer(_lote((0, 1)), {"players": _players(caminho)})

    assert _ids(saida) == [[1], [1, 2]]


def test_dois_jogadores_nao_trocam_de_identidade() -> None:
    """Sem a guarda de uma trajetoria por deteccao, o guloso daria o mesmo numero aos dois."""
    modulo = GreedyTracking()
    modulo.load()
    caminho = {
        0: [(100.0, 200.0), (300.0, 200.0)],
        1: [(110.0, 200.0), (310.0, 200.0)],
        2: [(120.0, 200.0), (320.0, 200.0)],
    }

    saida = modulo.infer(_lote((0, 1, 2)), {"players": _players(caminho)})

    assert _ids(saida) == [[1, 2], [1, 2], [1, 2]]


def test_salto_grande_demais_nao_e_o_mesmo_jogador() -> None:
    modulo = GreedyTracking()
    modulo.load()
    caminho = {0: [(100.0, 200.0)], 1: [(900.0, 700.0)]}

    saida = modulo.infer(_lote((0, 1)), {"players": _players(caminho)})

    assert _ids(saida) == [[1], [2]]


def test_a_previsao_de_movimento_segura_o_jogador_que_acelera() -> None:
    """Sem prever pela velocidade, um jogador rapido perde o numero na terceira caixa."""
    modulo = GreedyTracking()
    modulo.load()
    caminho = {0: [(100.0, 200.0)], 1: [(130.0, 200.0)], 2: [(160.0, 200.0)]}

    saida = modulo.infer(_lote((0, 1, 2)), {"players": _players(caminho)})

    assert _ids(saida) == [[1], [1], [1]]


def test_oclusao_curta_devolve_o_mesmo_numero() -> None:
    """Jogador que some por poucos quadros e volta perto continua sendo o mesmo."""
    modulo = GreedyTracking()
    modulo.load()
    caminho = {0: [(100.0, 200.0)], 1: [], 2: [], 3: [(105.0, 200.0)]}

    saida = modulo.infer(_lote((0, 1, 2, 3)), {"players": _players(caminho)})

    assert _ids(saida) == [[1], [], [], [1]]


def test_oclusao_longa_demais_encerra_a_trajetoria() -> None:
    modulo = GreedyTracking(max_gap=2)
    modulo.load()
    caminho = {0: [(100.0, 200.0)], 1: [], 2: [], 3: [], 4: [(105.0, 200.0)]}

    saida = modulo.infer(_lote((0, 1, 2, 3, 4)), {"players": _players(caminho)})

    assert _ids(saida) == [[1], [], [], [], [2]]


def test_sem_players_nao_ha_o_que_rastrear() -> None:
    modulo = GreedyTracking()
    modulo.load()

    assert modulo.infer(_lote((0, 1)), {}).records == ((), ())


def test_o_detection_index_aponta_para_a_posicao_na_lista_de_players() -> None:
    """E a chave que o `assemble` usa: errar aqui poe o `track_id` noutro jogador."""
    modulo = GreedyTracking()
    modulo.load()
    caminho = {0: [(100.0, 200.0), (300.0, 200.0), (500.0, 200.0)]}

    saida = modulo.infer(_lote((0,)), {"players": _players(caminho)})

    assert [r["detection_index"] for r in saida.records[0]] == [0, 1, 2]


def test_o_papel_do_track_e_estabilizado_por_toda_a_janela() -> None:
    modulo = GreedyTracking()
    modulo.load()
    entrada = _players_com_papeis(
        {
            0: [(100.0, 200.0, "player", 0.55)],
            1: [(105.0, 200.0, "referee", 0.90)],
            2: [(110.0, 200.0, "referee", 0.85)],
        }
    )

    saida = modulo.infer(_lote((0, 1, 2)), {"players": entrada})

    assert _ids(saida) == [[1], [1], [1]]
    assert [[r["role"] for r in quadro] for quadro in saida.records] == [
        ["referee"],
        ["referee"],
        ["referee"],
    ]


def test_papel_ausente_em_players_continua_ausente_no_tracking() -> None:
    modulo = GreedyTracking()
    modulo.load()

    saida = modulo.infer(_lote((0,)), {"players": _players({0: [(100.0, 200.0)]})})

    assert "role" not in saida.records[0][0]


def test_o_registro_cumpre_o_que_a_especie_promete() -> None:
    modulo = GreedyTracking()
    modulo.load()

    check_records(modulo.infer(_lote((0,)), {"players": _players({0: [(100.0, 200.0)]})}))


def test_duas_chamadas_com_a_mesma_entrada_devolvem_o_mesmo() -> None:
    """A razao de nao usar `model.track` do Ultralytics: ele guarda estado entre chamadas."""
    modulo = GreedyTracking()
    modulo.load()
    caminho = {q: [(100.0 + q * 5.0, 200.0), (300.0, 200.0)] for q in (0, 1, 2)}
    lote = _lote((0, 1, 2))
    entrada = {"players": _players(caminho)}

    assert modulo.infer(lote, entrada) == modulo.infer(lote, entrada)


def test_infer_sem_load_recusa() -> None:
    with pytest.raises(ModuleNotLoaded):
        GreedyTracking().infer(_lote((0,)), {})


def test_caixa_de_players_sem_box_nao_derruba_a_etapa() -> None:
    saida_sem_caixa = ModuleOutput(kind="players", frames=(0,), records=(({"confidence": 0.9},),))
    modulo = GreedyTracking()
    modulo.load()

    assert modulo.infer(_lote((0,)), {"players": saida_sem_caixa}).records == ((),)


def test_o_box_do_contrato_nao_e_lido_como_tupla_solta() -> None:
    """Guarda de regressao do casamento: caixa identica em dois quadros sobrepoe em 1,0."""
    modulo = GreedyTracking()
    modulo.load()
    caminho = {0: [(100.0, 200.0)], 1: [(100.0, 200.0)]}

    saida = modulo.infer(_lote((0, 1)), {"players": _players(caminho)})

    assert _ids(saida) == [[1], [1]]
    assert isinstance(Box(1.0, 2.0, 3.0, 4.0).x1, float)


def test_aafinidade_usa_a_forma_quando_as_caixas_se_tocam() -> None:
    """Mesmo centro e tamanhos distintos sao o jogador perto e o do fundo, e a forma separa."""
    from edge.modules.tracking.greedy import afinidade, iou

    perto = Box(100.0, 100.0, 200.0, 340.0)
    fundo = Box(140.0, 200.0, 160.0, 240.0)

    assert iou(perto, fundo) < 0.2
    assert afinidade(perto, perto) == 1.0


def test_aafinidade_usa_a_distancia_quando_as_caixas_nao_se_tocam() -> None:
    """Sem esse ramo, quem anda mais que a propria caixa nunca casa e nunca ganha velocidade."""
    from edge.modules.tracking.greedy import afinidade, iou

    antes = Box(100.0, 100.0, 140.0, 140.0)
    depois = Box(150.0, 100.0, 190.0, 140.0)

    assert iou(antes, depois) == 0.0
    assert afinidade(antes, depois) > 0.2


def test_caixa_longe_demais_temafinidade_zero() -> None:
    from edge.modules.tracking.greedy import afinidade

    assert afinidade(Box(0.0, 0.0, 40.0, 40.0), Box(900.0, 700.0, 940.0, 740.0)) == 0.0
