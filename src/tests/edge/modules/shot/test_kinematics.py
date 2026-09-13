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


def _bola_com_fonte(pontos: list[tuple[int, float, float, str | None]]) -> ModuleOutput:
    """Como `_bola`, mas declarando a procedencia que a especie `ball` publica em `source`."""
    return ModuleOutput(
        kind="ball",
        frames=tuple(p[0] for p in pontos),
        records=tuple(
            (
                {
                    "box": (x - 5.0, y - 5.0, x + 5.0, y + 5.0),
                    "confidence": 0.9,
                    **({} if fonte is None else {"source": fonte}),
                },
            )
            for _, x, y, fonte in pontos
        ),
    )


def _com_fonte_no_contato(fonte: str | None) -> list[tuple[int, float, float, str | None]]:
    """A mesma trajetoria dos outros testes, declarando a procedencia so no quadro da quebra."""
    return [
        (indice, x, y, fonte if indice == QUEBRA else "detected")
        for indice, x, y in _trajetoria_com_quebra()
    ]


class TestBallSupport:
    """A procedencia da bola no quadro escolhido, que o modulo jogava fora.

    O rastreador preenche lacuna curta por interpolacao linear, e o instante e exatamente onde a
    trajetoria vira: uma reta por cima da virada passa longe do pe. Sem estes campos, um instante
    decidido sobre posicao inventada sai com a mesma cara de um decidido sobre medida.
    """

    def _registro(self, pontos: list[tuple[int, float, float, str | None]]) -> dict[str, object]:
        modulo = BallBreakShot()
        modulo.load()
        indices = tuple(p[0] for p in pontos)
        saida = modulo.infer(_lote(indices), {"ball": _bola_com_fonte(pontos)})
        return _registros(saida)[0]

    def test_contato_sobre_posicao_medida_declara_detected(self) -> None:
        registro = self._registro(_com_fonte_no_contato("detected"))

        assert registro["source"] == MEDIDO
        assert registro["ball_support"] == "detected"
        assert registro["ball_interpolated_frames"] == 0

    def test_contato_sobre_posicao_inventada_declara_interpolated(self) -> None:
        """O mesmo instante, a mesma confianca, e agora o consumidor consegue distinguir."""
        registro = self._registro(_com_fonte_no_contato("interpolated"))

        assert registro["source"] == MEDIDO
        assert registro["ball_support"] == "interpolated"
        assert registro["ball_interpolated_frames"] == 1

    def test_a_previsao_nao_muda_com_a_procedencia(self) -> None:
        """A spec manda preservar a procedencia antes de deixar ela decidir qualquer coisa."""
        medido = self._registro(_com_fonte_no_contato("detected"))
        inventado = self._registro(_com_fonte_no_contato("interpolated"))

        assert medido["frame_index"] == inventado["frame_index"]
        assert medido["confidence"] == inventado["confidence"]

    def test_bola_sem_source_entra_como_unknown(self) -> None:
        """Detector que nao declara nao pode ser lido como se tivesse declarado medida."""
        registro = self._registro(_com_fonte_no_contato(None))

        assert registro["ball_support"] == "unknown"
        assert registro["ball_interpolated_frames"] == 0

    def test_o_caminho_do_ponto_medio_declara_o_que_havia_naquele_quadro(self) -> None:
        """O meio do trecho nao precisa ter bola, e `absent` e a resposta honesta."""
        modulo = BallBreakShot()
        modulo.load()
        indices = tuple(range(0, 120, PASSO))
        saida = modulo.infer(
            _lote(indices), {"ball": _bola_com_fonte([(0, 10.0, 10.0, "detected")])}
        )

        registro = _registros(saida)[0]
        assert registro["source"] == ESTIMADO
        assert registro["ball_support"] == "absent"
        assert registro["ball_detected_frames"] == 1

    def test_sem_bola_nenhuma_as_contagens_ficam_em_zero(self) -> None:
        modulo = BallBreakShot()
        modulo.load()

        saida = modulo.infer(_lote(tuple(range(0, 120, PASSO))), {})

        registro = _registros(saida)[0]
        assert registro["ball_support"] == "absent"
        assert registro["ball_detected_frames"] == 0
        assert registro["ball_interpolated_frames"] == 0

    def test_o_registro_continua_cumprindo_o_contrato_da_especie(self) -> None:
        modulo = BallBreakShot()
        modulo.load()
        pontos = _com_fonte_no_contato("interpolated")

        check_records(
            modulo.infer(_lote(tuple(p[0] for p in pontos)), {"ball": _bola_com_fonte(pontos)})
        )


class TestExigirBolaMedida:
    """A regra que usa a procedencia para decidir, testada com o criterio da spec 03 fixado antes.

    O rastreador da bola preenche lacuna curta com uma reta, e o instante e onde a trajetoria vira.
    Uma reta por cima da virada nao tem virada nenhuma: incluir aquele ponto no calculo da quebra
    convida o modulo a apontar o quadro errado com confianca alta.
    """

    def _pontos(self, fonte_no_contato: str) -> list[tuple[int, float, float, str | None]]:
        return _com_fonte_no_contato(fonte_no_contato)

    def _pico_interpolado(self) -> list[tuple[int, float, float, str | None]]:
        """Reta com um unico ponto interpolado fora dela, que e a falsa quebra a evitar.

        A interpolacao do rastreador e linear entre as duas deteccoes que fecham a lacuna, entao um
        ponto interpolado nunca deveria sair da reta. Quando ele sai, e porque a deteccao de um dos
        lados estava errada, e a quebra que ele cria e artefato e nao chute.
        """
        pontos: list[tuple[int, float, float, str | None]] = []
        for i in range(10):
            indice = i * PASSO
            fora = i == 5
            pontos.append(
                (
                    indice,
                    100.0 + 3.0 * indice,
                    500.0 - 200.0 * fora,
                    "interpolated" if fora else "detected",
                )
            )
        return pontos

    def test_ligado_nao_deixa_pico_interpolado_virar_quebra(self) -> None:
        pontos = self._pico_interpolado()
        ligado = BallBreakShot(exigir_bola_medida=True)
        desligado = BallBreakShot()
        ligado.load()
        desligado.load()
        lote = _lote(_indices_de(pontos))
        bola = {"ball": _bola_com_fonte(pontos)}

        com = _registros(ligado.infer(lote, bola))[0]
        sem = _registros(desligado.infer(lote, bola))[0]

        assert sem["source"] == MEDIDO
        assert sem["frame_index"] == 5 * PASSO
        assert com["source"] == ESTIMADO
        assert com["ball_interpolated_frames"] == 1

    def test_desligado_e_o_padrao_e_nao_muda_a_previsao(self) -> None:
        pontos = self._pontos("interpolated")
        padrao = BallBreakShot()
        padrao.load()

        saida = padrao.infer(_lote(_indices_de(pontos)), {"ball": _bola_com_fonte(pontos)})

        assert _registros(saida)[0]["ball_support"] == "interpolated"

    def test_sem_nenhuma_medida_o_modulo_cai_no_ponto_medio(self) -> None:
        """Descartar tudo nao pode virar instante inventado com cara de medido."""
        pontos: list[tuple[int, float, float, str | None]] = [
            (i * PASSO, 100.0 + i, 500.0, "interpolated") for i in range(10)
        ]
        modulo = BallBreakShot(exigir_bola_medida=True)
        modulo.load()

        saida = modulo.infer(_lote(_indices_de(pontos)), {"ball": _bola_com_fonte(pontos)})

        registro = _registros(saida)[0]
        assert registro["source"] == ESTIMADO
        assert registro["ball_interpolated_frames"] == 10

    def test_configure_recusa_nome_fora_do_vocabulario(self) -> None:
        with pytest.raises(ValueError, match="shot.exigir_bola_detectada nao existe"):
            BallBreakShot().configure({"exigir_bola_detectada": True})

    def test_configure_recusa_valor_que_nao_e_booleano(self) -> None:
        with pytest.raises(ValueError, match="espera booleano"):
            BallBreakShot().configure({"exigir_bola_medida": "sim"})

    def test_configure_chega_ao_comportamento(self) -> None:
        pontos = self._pico_interpolado()
        modulo = BallBreakShot()
        modulo.configure({"exigir_bola_medida": True})
        modulo.load()

        saida = modulo.infer(_lote(_indices_de(pontos)), {"ball": _bola_com_fonte(pontos)})

        assert _registros(saida)[0]["source"] == ESTIMADO


def _indices_de(pontos: list[tuple[int, float, float, str | None]]) -> tuple[int, ...]:
    return tuple(p[0] for p in pontos)
