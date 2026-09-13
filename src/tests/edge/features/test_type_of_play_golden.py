"""Golden da regra atual de `type_of_play`, escrito antes de refatorar.

Este arquivo nao verifica se a classificacao esta certa: ele fixa o que a regra devolve hoje,
para que a refatoracao falhe alto se mudar alguma predicao. Os literais foram copiados da saida
real de `TypeOfPlay.extract`, executada sobre as sete percepcoes montadas aqui, e nao derivados
de um calculo paralelo. Reimplementar a arvore no teste tiraria dele exatamente a capacidade de
detectar que a arvore mudou.

A evidencia e conferida por subconjunto, de proposito: a refatoracao pode acrescentar chaves sem
invalidar o golden, e mudar o valor de uma das quatro chaves atuais nao pode passar. Os floats
que saem de divisao ou raiz vem com `pytest.approx`; os que sao exatos em binario (0.0, 3.0,
25.0) ficam como literal.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from contracts import BallDetection, Box, FramePerception, Perception, PlayerDetection
from edge.features.type_of_play import TypeOfPlay

CONTACT_FRAME = 120
FPS = 59.94
PLAYER_BOX_HEIGHT = 180.0


class _Shot:
    def __init__(self, frame_index: int = CONTACT_FRAME) -> None:
        self.frame_index = frame_index
        self.shooter_track_id: int | None = None


def _frame(
    index: int,
    center: tuple[float, float],
    *,
    source: str | None = None,
    gap_frames: int | None = None,
    players: tuple[PlayerDetection, ...] = (),
) -> FramePerception:
    x, y = center
    return FramePerception(
        frame_index=index,
        timestamp_s=index / FPS,
        players=players,
        ball=BallDetection(
            box=Box(x - 5.0, y - 5.0, x + 5.0, y + 5.0),
            confidence=0.0 if source == "interpolated" else 0.9,
            source=source,
            gap_frames=gap_frames,
        ),
    )


def _perception(steps: list[tuple[int, float, float]]) -> Perception:
    return Perception(clip_id="golden", frames=tuple(_frame(i, (x, y)) for i, x, y in steps))


def _players(offset: float) -> tuple[PlayerDetection, ...]:
    return tuple(
        PlayerDetection(
            box=Box(x, 200.0, x + 60.0, 200.0 + PLAYER_BOX_HEIGHT),
            confidence=0.8,
            track_id=index,
        )
        for index, x in enumerate((300.0 + offset, 700.0 - offset))
    )


def _assert_evidence_subset(evidence: Mapping[str, object], pinned: Mapping[str, object]) -> None:
    for key, expected in pinned.items():
        assert key in evidence, f"a evidencia perdeu a chave {key!r}"
        assert evidence[key] == expected, f"a chave {key!r} mudou de valor"


def test_bola_imovel_por_um_trecho_longo_sai_como_bola_parada() -> None:
    steps = [(CONTACT_FRAME - i * 12, 500.0, 300.0) for i in range(5)][::-1]

    valor = TypeOfPlay().extract(_perception(steps), _Shot())

    assert valor.status == "ok"
    assert valor.value == "set piece"
    _assert_evidence_subset(
        valor.evidence,
        {
            "quadros_lidos": 5,
            "quadros_imoveis": 48,
            "velocidade_de_chegada_px_por_quadro": 0.0,
            "leitura": "bola imovel por 48 quadros antes do contato",
        },
    )


def test_bola_chegando_rapido_sai_como_jogada_aberta() -> None:
    steps = [(CONTACT_FRAME - i * 12, 500.0 - i * 200.0, 300.0) for i in range(5)][::-1]

    valor = TypeOfPlay().extract(_perception(steps), _Shot())

    assert valor.status == "ok"
    assert valor.value == "open play"
    _assert_evidence_subset(
        valor.evidence,
        {
            "quadros_lidos": 5,
            "quadros_imoveis": 0,
            "velocidade_de_chegada_px_por_quadro": pytest.approx(16.67),
            "leitura": "bola chegou a 16.7 px por quadro",
        },
    )


def test_bola_vindo_devagar_junto_do_finalizador_sai_como_lance_individual() -> None:
    steps = [(CONTACT_FRAME - i * 12, 500.0 - i * 36.0, 300.0) for i in range(5)][::-1]

    valor = TypeOfPlay().extract(_perception(steps), _Shot())

    assert valor.status == "ok"
    assert valor.value == "individual"
    _assert_evidence_subset(
        valor.evidence,
        {
            "quadros_lidos": 5,
            "quadros_imoveis": 0,
            "velocidade_de_chegada_px_por_quadro": 3.0,
            "leitura": "bola chegou a 3.0 px por quadro",
        },
    )


def test_bola_que_parou_andou_e_parou_de_novo_conta_so_o_trecho_final() -> None:
    """O trecho parado do inicio nao entra na conta, e sobram 12 quadros dos 48 imoveis."""
    steps = [
        (40, 500.0, 300.0),
        (52, 500.0, 300.0),
        (64, 500.0, 300.0),
        (76, 500.0, 300.0),
        (88, 200.0, 300.0),
        (100, 200.0, 300.0),
    ]

    valor = TypeOfPlay().extract(_perception(steps), _Shot())

    assert valor.status == "ok"
    assert valor.value == "open play"
    _assert_evidence_subset(
        valor.evidence,
        {
            "quadros_lidos": 6,
            "quadros_imoveis": 12,
            "velocidade_de_chegada_px_por_quadro": 25.0,
            "leitura": "bola chegou a 25.0 px por quadro",
        },
    )


def test_cadencia_de_producao_com_passo_4_e_22_amostras() -> None:
    """Espelha um evento real cujo `quadros_lidos` foi 22, com a bola amostrada a cada 4 quadros."""
    steps = [(36 + 4 * i, 200.0 + 25.0 * i, 100.0 + 10.0 * i) for i in range(22)]

    valor = TypeOfPlay().extract(_perception(steps), _Shot())

    assert valor.status == "ok"
    assert valor.value == "open play"
    _assert_evidence_subset(
        valor.evidence,
        {
            "quadros_lidos": 22,
            "quadros_imoveis": 0,
            "velocidade_de_chegada_px_por_quadro": pytest.approx(6.73),
            "leitura": "bola chegou a 6.7 px por quadro",
        },
    )


def test_pontos_interpolados_entram_na_leitura_como_os_detectados() -> None:
    """A regra nao le `source`, e o golden fixa isso: interpolado pesa igual a detectado."""
    steps: list[tuple[int, float, float, str, int | None]] = [
        (72, 500.0, 300.0, "detected", 0),
        (84, 501.0, 300.0, "interpolated", 12),
        (96, 500.0, 301.0, "detected", 0),
        (108, 502.0, 300.0, "interpolated", 12),
        (120, 500.0, 300.0, "detected", 0),
    ]
    perception = Perception(
        clip_id="golden",
        frames=tuple(
            _frame(i, (x, y), source=source, gap_frames=gap) for i, x, y, source, gap in steps
        ),
    )

    valor = TypeOfPlay().extract(perception, _Shot())

    assert valor.status == "ok"
    assert valor.value == "set piece"
    _assert_evidence_subset(
        valor.evidence,
        {
            "quadros_lidos": 5,
            "quadros_imoveis": 48,
            "velocidade_de_chegada_px_por_quadro": pytest.approx(0.19),
            "leitura": "bola imovel por 48 quadros antes do contato",
        },
    )


def test_percepcao_com_jogadores_nao_muda_a_leitura_da_bola() -> None:
    """`requires` fica so em `ball`, e o golden fixa que caixa de jogador nao entra na conta."""
    perception = Perception(
        clip_id="golden",
        frames=tuple(
            _frame(
                CONTACT_FRAME - i * 12,
                (500.0 - i * 100.0, 300.0),
                players=_players(float(i) * 10.0),
            )
            for i in range(5)
        )[::-1],
    )

    valor = TypeOfPlay().extract(perception, _Shot())

    assert valor.status == "ok"
    assert valor.value == "open play"
    _assert_evidence_subset(
        valor.evidence,
        {
            "quadros_lidos": 5,
            "quadros_imoveis": 0,
            "velocidade_de_chegada_px_por_quadro": pytest.approx(8.33),
            "leitura": "bola chegou a 8.3 px por quadro",
        },
    )
