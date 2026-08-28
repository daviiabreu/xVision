"""A feature `type_of_play`: a arvore de tres ramos sobre a trajetoria da bola antes do contato.

O ramo `individual` e o limite declarado da feature: o enum do parceiro colapsa padrao de jogo e
assistido contra individual, e nenhuma regra sobre a bola separa lance individual de jogada aberta
sem assistencia. Os testes fixam o que a regra promete, e nao mais que isso.
"""

from __future__ import annotations

from typing import cast

from contracts import BallDetection, Box, FramePerception, Perception
from edge.features.type_of_play import ABERTA, INDIVIDUAL, PARADA, TypeOfPlay

CONTATO = 120


class _Shot:
    def __init__(self, frame_index: int = CONTATO) -> None:
        self.frame_index = frame_index
        self.shooter_track_id = None


def _quadro(indice: int, centro: tuple[float, float]) -> FramePerception:
    x, y = centro
    return FramePerception(
        frame_index=indice,
        timestamp_s=indice / 59.94,
        ball=BallDetection(box=Box(x - 5, y - 5, x + 5, y + 5), confidence=0.9),
    )


def _trajetoria(passos: list[tuple[int, float, float]]) -> Perception:
    return Perception(clip_id="c", frames=tuple(_quadro(i, (x, y)) for i, x, y in passos))


def _parada_ate(contato: int, quantos: int, passo: int = 12) -> list[tuple[int, float, float]]:
    return [(contato - i * passo, 500.0, 300.0) for i in range(quantos)][::-1]


def test_bola_imovel_por_um_trecho_longo_e_bola_parada() -> None:
    percepcao = _trajetoria(_parada_ate(CONTATO, 5))

    valor = TypeOfPlay().extract(percepcao, _Shot())

    assert valor.status == "ok"
    assert valor.value == PARADA


def test_bola_chegando_rapido_de_outro_lugar_e_jogada_aberta() -> None:
    passos = [(CONTATO - i * 12, 500.0 - i * 200.0, 300.0) for i in range(5)][::-1]

    assert TypeOfPlay().extract(_trajetoria(passos), _Shot()).value == ABERTA


def test_bola_vindo_devagar_junto_do_finalizador_e_lance_individual() -> None:
    passos = [(CONTATO - i * 12, 500.0 - i * 36.0, 300.0) for i in range(5)][::-1]

    assert TypeOfPlay().extract(_trajetoria(passos), _Shot()).value == INDIVIDUAL


def test_sem_bola_suficiente_na_janela_sai_unavailable() -> None:
    percepcao = _trajetoria([(CONTATO, 500.0, 300.0)])

    valor = TypeOfPlay().extract(percepcao, _Shot())

    assert valor.status == "unavailable"
    assert valor.value is None
    assert valor.reason


def test_bola_fora_da_janela_anterior_nao_conta() -> None:
    """A janela e a mesma de 1,5 s que o frameset declara: antes disso e outra jogada."""
    passos = [(CONTATO - 400 + i * 12, 500.0, 300.0) for i in range(5)]

    assert TypeOfPlay().extract(_trajetoria(passos), _Shot()).status == "unavailable"


def test_bola_que_parou_andou_e_parou_nao_e_bola_parada() -> None:
    """Somar os dois trechos parados diria que sim, e a contagem para no primeiro movimento."""
    passos = [
        (CONTATO - 96, 500.0, 300.0),
        (CONTATO - 84, 500.0, 300.0),
        (CONTATO - 72, 500.0, 300.0),
        (CONTATO - 60, 100.0, 300.0),
        (CONTATO - 48, 100.0, 300.0),
        (CONTATO - 36, 100.0, 300.0),
    ]

    valor = TypeOfPlay().extract(_trajetoria(passos), _Shot())

    assert int(cast("int", valor.evidence["quadros_imoveis"])) < 48


def test_a_velocidade_sai_por_quadro_e_nao_por_amostra() -> None:
    """A especie ball roda a 5 fps: dividir pela posicao na lista daria doze vezes a velocidade."""
    espacado = [(CONTATO - i * 12, 500.0 - i * 36.0, 300.0) for i in range(5)][::-1]

    valor = TypeOfPlay().extract(_trajetoria(espacado), _Shot())

    assert valor.evidence["velocidade_de_chegada_px_por_quadro"] == 3.0


def test_a_evidencia_diz_como_a_leitura_foi_feita() -> None:
    valor = TypeOfPlay().extract(_trajetoria(_parada_ate(CONTATO, 5)), _Shot())

    assert valor.evidence["quadros_lidos"] == 5
    assert "imovel" in str(valor.evidence["leitura"])


def test_o_valor_cabe_no_dominio_que_o_rotulo_declara() -> None:
    valor = TypeOfPlay().extract(_trajetoria(_parada_ate(CONTATO, 5)), _Shot())

    assert valor.value in {ABERTA, PARADA, INDIVIDUAL}
