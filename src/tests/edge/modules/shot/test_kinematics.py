"""O modulo `shot`: herda a suite de conformidade e prova o que o `process_clip` vai cobrar.

A suite cobra o contrato de modulo. Estes testes cobram o contrato do consumidor, que e outro: o
`process_clip` recusa saida sem instante, com mais de um instante distinto, ou com instante fora do
clipe, e cada recusa e uma excecao propria. Modulo que passa na conformidade e falha nessas tres
continua derrubando todo clipe.
"""

from __future__ import annotations

import numpy as np
import pytest

from edge.geometry.contact import Observation
from edge.modules.base import FrameBatch, ModuleNotLoaded, ModuleOutput, PerceptionModule
from edge.modules.conformance import ModuleConformance
from edge.modules.records import check_records
from edge.modules.shot import BallBreakShot
from edge.modules.shot.kinematics import ESTIMADO, MEDIDO, PREPROCESS, observacoes

PASSO = 12
QUEBRA = 48


def _lote(indices: tuple[int, ...]) -> FrameBatch:
    tensor = np.zeros((len(indices), PREPROCESS.height, PREPROCESS.width, 1), dtype=np.uint8)
    return FrameBatch(indices=indices, tensor=tensor, preprocess=PREPROCESS)


def _bola(pontos: list[tuple[int, float, float]]) -> ModuleOutput:
    """Uma caixa de 10 px centrada em cada ponto, que e o formato que a especie `ball` promete."""
    return ModuleOutput(
        kind="ball",
        frames=tuple(p[0] for p in pontos),
        records=tuple(
            ({"box": (x - 5.0, y - 5.0, x + 5.0, y + 5.0), "confidence": 0.9},)
            for _, x, y in pontos
        ),
    )


def _trajetoria_com_quebra() -> list[tuple[int, float, float]]:
    antes = [(i * PASSO, 100.0 + 3.0 * i * PASSO, 500.0) for i in range(QUEBRA // PASSO + 1)]
    _, x, y = antes[-1]
    depois = [(QUEBRA + d, x - 6.0 * d, y - 4.0 * d) for d in (PASSO, 2 * PASSO, 3 * PASSO)]
    return antes + depois


def _indices(pontos: list[tuple[int, float, float]]) -> tuple[int, ...]:
    return tuple(p[0] for p in pontos)


def _registros(saida: ModuleOutput) -> list[dict[str, object]]:
    return [dict(r) for registros in saida.records for r in registros]


class TestBallBreakShot(ModuleConformance):
    @pytest.fixture
    def module(self) -> PerceptionModule:
        return BallBreakShot()


def test_aponta_a_quebra_da_trajetoria() -> None:
    pontos = _trajetoria_com_quebra()
    modulo = BallBreakShot()
    modulo.load()

    saida = modulo.infer(_lote(_indices(pontos)), {"ball": _bola(pontos)})

    registros = _registros(saida)
    assert len(registros) == 1
    assert registros[0]["source"] == MEDIDO
    assert abs(int(registros[0]["frame_index"]) - QUEBRA) <= 6  # type: ignore[call-overload]


def test_um_instante_so_por_clipe() -> None:
    """`process_clip` levanta `AmbiguousContact` com mais de um instante distinto."""
    pontos = _trajetoria_com_quebra()
    modulo = BallBreakShot()
    modulo.load()

    saida = modulo.infer(_lote(_indices(pontos)), {"ball": _bola(pontos)})

    assert len({r["frame_index"] for r in _registros(saida)}) == 1


def test_instante_dentro_do_clipe() -> None:
    """`process_clip` levanta `ContactOutOfRange` fora de `0 <= valor < frame_count`."""
    pontos = _trajetoria_com_quebra()
    modulo = BallBreakShot()
    modulo.load()

    saida = modulo.infer(_lote(_indices(pontos)), {"ball": _bola(pontos)})

    quadro = _registros(saida)[0]["frame_index"]
    assert isinstance(quadro, int) and not isinstance(quadro, bool)
    assert 0 <= quadro <= max(_indices(pontos))


def test_sem_bola_cai_para_o_ponto_medio_declarado() -> None:
    """Nao emitir nada custaria o clipe inteiro, entao ele estima e diz que estimou."""
    indices = tuple(range(0, 120, PASSO))
    modulo = BallBreakShot()
    modulo.load()

    saida = modulo.infer(_lote(indices), {})

    registros = _registros(saida)
    assert len(registros) == 1
    assert registros[0]["source"] == ESTIMADO
    assert registros[0]["reason"]
    assert registros[0]["frame_index"] in indices


def test_a_estimativa_sai_com_confianca_baixa() -> None:
    """Sem isso, medido e estimado chegam ao consumidor com a mesma cara."""
    modulo = BallBreakShot()
    modulo.load()

    estimado = _registros(modulo.infer(_lote(tuple(range(0, 120, PASSO))), {}))[0]
    pontos = _trajetoria_com_quebra()
    medido = _registros(modulo.infer(_lote(_indices(pontos)), {"ball": _bola(pontos)}))[0]

    assert float(medido["confidence"]) > float(estimado["confidence"])  # type: ignore[arg-type]


def test_estimativa_pode_ser_desligada() -> None:
    modulo = BallBreakShot(estimar_quando_nao_medir=False)
    modulo.load()

    saida = modulo.infer(_lote(tuple(range(0, 120, PASSO))), {})

    assert _registros(saida) == []


def test_bola_parada_nao_vira_finalizacao_medida() -> None:
    parada = [(i * PASSO, 100.0, 500.0) for i in range(8)]
    modulo = BallBreakShot()
    modulo.load()

    saida = modulo.infer(_lote(_indices(parada)), {"ball": _bola(parada)})

    assert _registros(saida)[0]["source"] == ESTIMADO


def test_o_registro_cumpre_o_que_a_especie_promete() -> None:
    pontos = _trajetoria_com_quebra()
    modulo = BallBreakShot()
    modulo.load()

    check_records(modulo.infer(_lote(_indices(pontos)), {"ball": _bola(pontos)}))


def test_infer_sem_load_recusa() -> None:
    with pytest.raises(ModuleNotLoaded):
        BallBreakShot().infer(_lote((0, 12)), {})


def test_lote_vazio_nao_estoura() -> None:
    modulo = BallBreakShot()
    modulo.load()

    saida = modulo.infer(_lote(()), {})

    assert saida.frames == ()
    assert saida.records == ()


def test_registro_de_bola_sem_caixa_e_ignorado_sem_derrubar_a_etapa() -> None:
    """`check_records` roda depois do `infer`, e ela e quem deve nomear o que faltou."""
    saida = ModuleOutput(kind="ball", frames=(0, 12), records=(({"confidence": 0.9},), ()))

    assert observacoes(saida) == ()


def test_o_portador_do_registro_nao_precisa_ser_o_instante() -> None:
    """O refino cai entre amostras, e amarrar o registro a uma delas desfaria o refino."""
    pontos = _trajetoria_com_quebra()
    modulo = BallBreakShot()
    modulo.load()

    saida = modulo.infer(_lote(_indices(pontos)), {"ball": _bola(pontos)})

    portador = next(q for q, r in zip(saida.frames, saida.records, strict=True) if r)
    quadro = _registros(saida)[0]["frame_index"]
    assert 0 <= abs(int(quadro) - portador) <= PASSO  # type: ignore[call-overload]


def test_observacoes_usam_o_centro_da_caixa() -> None:
    saida = ModuleOutput(
        kind="ball",
        frames=(4,),
        records=(({"box": (10.0, 20.0, 30.0, 60.0), "confidence": 0.9},),),
    )

    assert observacoes(saida) == (Observation(4, 20.0, 40.0),)
