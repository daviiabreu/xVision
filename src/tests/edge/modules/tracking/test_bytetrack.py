from __future__ import annotations

from collections.abc import Callable, Sequence

import cv2
import numpy as np
import pytest

from contracts import Box
from edge.modules.base import FrameBatch, ModuleNotLoaded, ModuleOutput, PerceptionModule
from edge.modules.conformance import ModuleConformance
from edge.modules.records import check_records
from edge.modules.tracking import ByteTrackTracking
from edge.modules.tracking.bytetrack import PREPROCESS, Amostra, cor_da_camisa, costurar
from edge.participants.identity import LabColor


def _lote(indices: tuple[int, ...]) -> FrameBatch:
    tensor = np.zeros((len(indices), PREPROCESS.height, PREPROCESS.width, 3), dtype=np.uint8)
    return FrameBatch(indices=indices, tensor=tensor, preprocess=PREPROCESS)


def _players(
    por_quadro: dict[int, list[tuple[float, float, str]]], lado: float = 40.0
) -> ModuleOutput:
    quadros = tuple(sorted(por_quadro))
    return ModuleOutput(
        kind="players",
        frames=quadros,
        records=tuple(
            tuple(
                {
                    "box": (x, y, x + lado, y + lado),
                    "confidence": 0.9,
                    "role": role,
                }
                for x, y, role in por_quadro[quadro]
            )
            for quadro in quadros
        ),
    )


def _ids(saida: ModuleOutput) -> list[list[int]]:
    return [[int(r["track_id"]) for r in registros] for registros in saida.records]  # type: ignore[arg-type]


class TestByteTrackTracking(ModuleConformance):
    @pytest.fixture
    def module(self) -> PerceptionModule:
        return ByteTrackTracking()


def test_jogador_em_movimento_mantem_o_numero() -> None:
    modulo = ByteTrackTracking()
    modulo.load()
    caminho = {q: [(100.0 + q * 5.0, 200.0, "player")] for q in (0, 1, 2, 3)}

    saida = modulo.infer(_lote((0, 1, 2, 3)), {"players": _players(caminho)})

    assert _ids(saida) == [[1], [1], [1], [1]]


def test_duas_chamadas_com_a_mesma_entrada_sao_deterministas() -> None:
    modulo = ByteTrackTracking()
    modulo.load()
    caminho = {q: [(100.0 + q * 5.0, 200.0, "player")] for q in (0, 1, 2)}
    lote = _lote((0, 1, 2))
    entrada = {"players": _players(caminho)}

    assert modulo.infer(lote, entrada) == modulo.infer(lote, entrada)


def test_detection_index_continua_apontando_para_players_quando_ha_registro_invalido() -> None:
    players = ModuleOutput(
        kind="players",
        frames=(0,),
        records=(
            (
                {"confidence": 0.9},
                {"box": (100.0, 100.0, 140.0, 140.0), "confidence": 0.9},
            ),
        ),
    )
    modulo = ByteTrackTracking()
    modulo.load()

    saida = modulo.infer(_lote((0,)), {"players": players})

    assert saida.records[0][0]["detection_index"] == 1


def test_papel_do_track_e_estabilizado_por_toda_a_janela() -> None:
    modulo = ByteTrackTracking()
    modulo.load()
    entrada = _players(
        {
            0: [(100.0, 200.0, "player")],
            1: [(105.0, 200.0, "referee")],
            2: [(110.0, 200.0, "referee")],
        }
    )

    saida = modulo.infer(_lote((0, 1, 2)), {"players": entrada})

    assert [[r["role"] for r in quadro] for quadro in saida.records] == [
        ["referee"],
        ["referee"],
        ["referee"],
    ]


def test_sem_players_nao_ha_o_que_rastrear() -> None:
    modulo = ByteTrackTracking()
    modulo.load()

    assert modulo.infer(_lote((0, 1)), {}).records == ((), ())


def test_o_registro_cumpre_o_contrato() -> None:
    modulo = ByteTrackTracking()
    modulo.load()
    players = _players({0: [(100.0, 200.0, "player")]})

    check_records(modulo.infer(_lote((0,)), {"players": players}))


def test_infer_sem_load_recusa() -> None:
    with pytest.raises(ModuleNotLoaded):
        ByteTrackTracking().infer(_lote((0,)), {})


def test_cor_da_camisa_leva_a_caixa_do_quadro_original_ao_tensor() -> None:
    """A caixa 300..360 x 300..420 de um quadro 1920x1080 cai em 200..240 x 208..288 no tensor."""
    tensor = np.full((1, PREPROCESS.height, PREPROCESS.width, 3), (0, 160, 0), dtype=np.uint8)
    tensor[0, 226:250, 212:228] = (0, 0, 220)
    lote = FrameBatch(indices=(0,), tensor=tensor, preprocess=PREPROCESS, source_size=(1920, 1080))

    cor = cor_da_camisa(lote, 0, Box(300.0, 300.0, 360.0, 420.0))

    vermelho = cv2.cvtColor(np.array([[[0, 0, 220]]], dtype=np.uint8), cv2.COLOR_BGR2LAB)[0, 0]
    assert cor == pytest.approx(tuple(float(v) for v in vermelho), abs=1)


def _trajetoria(posicoes: range, x: float, lado: float = 40.0) -> list[Amostra]:
    return [(p, Box(x, 200.0, x + lado, 200.0 + 2 * lado)) for p in posicoes]


def _cor_fixa(
    cores: dict[float, LabColor | None],
) -> Callable[[Sequence[Amostra]], LabColor | None]:
    """A cor sai do x da primeira caixa, para o teste escolher a cor de cada trajetoria."""
    return lambda amostras: cores[amostras[0][1].x1]


VERDE = (120.0, 90.0, 150.0)
AMARELO = (200.0, 120.0, 190.0)


def test_fragmento_que_nasce_perto_do_fim_com_a_mesma_cor_herda_o_id() -> None:
    trajetorias = {1: _trajetoria(range(0, 5), 100.0), 2: _trajetoria(range(8, 12), 130.0)}

    destino = costurar(
        trajetorias, {1: "player", 2: "player"}, _cor_fixa({100.0: VERDE, 130.0: VERDE})
    )

    assert destino == {2: 1}


@pytest.mark.parametrize(
    ("inicio_de_b", "x_de_b", "papel_de_b", "cor_de_b"),
    [
        (10, 130.0, "player", VERDE),
        (8, 180.0, "player", VERDE),
        (8, 130.0, "referee", VERDE),
        (8, 130.0, "player", AMARELO),
        (8, 130.0, "player", None),
        (8, 130.0, "player", (20.0, 90.0, 150.0)),
    ],
    ids=[
        "lacuna-acima-de-4",
        "longe-demais",
        "outro-papel",
        "outra-cor",
        "sem-cor",
        "outro-brilho",
    ],
)
def test_sem_as_condicoes_nao_ha_costura(
    inicio_de_b: int, x_de_b: float, papel_de_b: str, cor_de_b: LabColor | None
) -> None:
    trajetorias = {
        1: _trajetoria(range(0, 5), 100.0),
        2: _trajetoria(range(inicio_de_b, inicio_de_b + 4), x_de_b),
    }

    destino = costurar(
        trajetorias, {1: "player", 2: papel_de_b}, _cor_fixa({100.0: VERDE, x_de_b: cor_de_b})
    )

    assert destino == {}


def test_a_candidata_mais_proxima_fica_com_o_fragmento() -> None:
    trajetorias = {
        1: _trajetoria(range(0, 5), 100.0),
        2: _trajetoria(range(0, 5), 150.0),
        3: _trajetoria(range(7, 10), 140.0),
    }
    cores = _cor_fixa({100.0: VERDE, 150.0: VERDE, 140.0: VERDE})

    assert costurar(trajetorias, {1: "player", 2: "player", 3: "player"}, cores) == {3: 2}


def test_a_costura_encadeia_e_todos_herdam_o_primeiro_id() -> None:
    trajetorias = {
        1: _trajetoria(range(0, 4), 100.0),
        2: _trajetoria(range(6, 9), 110.0),
        3: _trajetoria(range(11, 14), 120.0),
    }
    cores = _cor_fixa({100.0: VERDE, 110.0: VERDE, 120.0: VERDE})

    assert costurar(trajetorias, {1: "player", 2: "player", 3: "player"}, cores) == {2: 1, 3: 1}


def test_o_jogador_que_some_e_volta_ao_lado_sai_com_um_id_so() -> None:
    """Sem sobreposicao na volta o ByteTrack abre outro ID, e a camisa igual devolve o primeiro."""
    posicoes = {0: 100.0, 1: 100.0, 2: 100.0, 5: 135.0, 6: 135.0, 7: 135.0}
    tensor = np.full((8, PREPROCESS.height, PREPROCESS.width, 3), (0, 160, 0), dtype=np.uint8)
    for quadro, x in posicoes.items():
        tensor[quadro, 211:220, int(x) + 11 : int(x) + 29] = (0, 0, 220)
    lote = FrameBatch(indices=tuple(range(8)), tensor=tensor, preprocess=PREPROCESS)
    players = _players({q: [(x, 200.0, "player")] for q, x in posicoes.items()})
    modulo = ByteTrackTracking()
    modulo.load()

    saida = modulo.infer(lote, {"players": players})

    assert {i for quadro in _ids(saida) for i in quadro} == {1}


def test_costura_desligada_pela_configuracao_deixa_os_fragmentos() -> None:
    """E o modo de rotular: o par so pode ser julgado se as duas trajetorias sairem separadas."""
    posicoes = {0: 100.0, 1: 100.0, 2: 100.0, 5: 135.0, 6: 135.0, 7: 135.0}
    tensor = np.full((8, PREPROCESS.height, PREPROCESS.width, 3), (0, 160, 0), dtype=np.uint8)
    for quadro, x in posicoes.items():
        tensor[quadro, 211:220, int(x) + 11 : int(x) + 29] = (0, 0, 220)
    lote = FrameBatch(indices=tuple(range(8)), tensor=tensor, preprocess=PREPROCESS)
    players = _players({q: [(x, 200.0, "player")] for q, x in posicoes.items()})
    modulo = ByteTrackTracking()
    modulo.configure({"costura": False})
    modulo.load()

    saida = modulo.infer(lote, {"players": players})

    assert {i for quadro in _ids(saida) for i in quadro} == {1, 2}


def test_a_costura_entra_na_impressao_do_cache() -> None:
    sem_costura = ByteTrackTracking()
    sem_costura.configure({"costura": False})

    assert sem_costura.fingerprint() != ByteTrackTracking().fingerprint()


@pytest.mark.parametrize("opcoes", [{"costura": "nao"}, {"lacuna": 4}])
def test_ajuste_que_o_tracking_nao_conhece_e_recusado(opcoes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ByteTrackTracking().configure(opcoes)
