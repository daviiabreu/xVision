"""A regua que mede se a bola do contato e a bola do lance.

O criterio nao e estatistico e sim fisico: numa finalizacao a bola encosta no finalizador, entao
deteccao longe de toda caixa de jogador nao descreve o lance. Os testes fecham as tres formas de a
regua mentir, que sao piores que ela ser grosseira.

A pior das tres e a taxa dividida pelos clipes que tem bola. Um detector que aprendesse a nao
detectar subiria a taxa ate 1.0 detectando uma vez so e acertando, e a metrica aplaudiria.
"""

from __future__ import annotations

from contracts import BallDetection, Box, FramePerception, PlayerDetection
from mlops.ball_quality import (
    ENCOSTADA_PX,
    Contato,
    contato_de,
    distancia_ate_jogador,
    measure,
    quadro_do_contato,
)

JOGADOR = Box(100.0, 200.0, 140.0, 280.0)


def _quadro(indice: int, bola: Box | None, jogadores: tuple[Box, ...] = (JOGADOR,)):
    return FramePerception(
        frame_index=indice,
        timestamp_s=indice / 60.0,
        players=tuple(PlayerDetection(box=b, confidence=0.9) for b in jogadores),
        ball=None if bola is None else BallDetection(box=bola, confidence=0.8),
    )


def test_bola_dentro_da_caixa_da_distancia_zero() -> None:
    assert distancia_ate_jogador((120.0, 240.0), [JOGADOR]) == 0.0


def test_bola_ao_lado_da_caixa_da_a_distancia_ate_a_borda() -> None:
    assert distancia_ate_jogador((160.0, 240.0), [JOGADOR]) == 20.0


def test_sem_jogador_nenhum_a_distancia_e_indefinida_e_nao_zero() -> None:
    """Zero significaria encostada, e quadro sem jogador nao prova contato nenhum."""
    assert distancia_ate_jogador((120.0, 240.0), []) is None


def test_a_distancia_e_ate_o_jogador_mais_proximo() -> None:
    longe = Box(900.0, 200.0, 940.0, 280.0)
    assert distancia_ate_jogador((160.0, 240.0), [longe, JOGADOR]) == 20.0


def test_quadro_sem_bola_conta_como_falha_e_nao_e_descartado() -> None:
    """Clipe sem bola no contato falha do mesmo jeito que clipe com bola no lugar errado."""
    contato = contato_de("c1", _quadro(200, None), altura=1080)

    assert contato.tem_bola is False
    assert contato.encostada is False
    assert contato.perdida is True


def test_bola_encostada_no_jogador_passa() -> None:
    contato = contato_de("c1", _quadro(200, Box(118.0, 238.0, 130.0, 250.0)), altura=1080)

    assert contato.distancia_px == 0.0
    assert contato.encostada is True


def test_bola_na_arquibancada_e_marcada_como_fora_do_gramado() -> None:
    """Arquibancada e `y` pequeno, porque `y` menor e mais alto na imagem."""
    contato = contato_de("c1", _quadro(200, Box(700.0, 170.0, 712.0, 182.0)), altura=1080)

    assert contato.fora_do_gramado is True
    assert contato.encostada is False


def test_a_folga_de_contato_e_inclusiva() -> None:
    """Exatamente no limiar conta como encostada: raio da bola mais erro de caixa do detector."""
    no_limiar = Contato("c1", tem_bola=True, distancia_px=ENCOSTADA_PX, y_relativo=0.6)
    passou = Contato("c1", tem_bola=True, distancia_px=ENCOSTADA_PX + 0.1, y_relativo=0.6)

    assert no_limiar.encostada is True
    assert passou.encostada is False


def test_a_taxa_divide_pelos_clipes_e_nao_pelos_que_tem_bola() -> None:
    """Dividir pelos que tem bola premiaria um detector que aprendesse a nao detectar."""
    contatos = [
        Contato("a", tem_bola=True, distancia_px=0.0, y_relativo=0.6),
        Contato("b", tem_bola=False, distancia_px=None, y_relativo=None),
        Contato("c", tem_bola=False, distancia_px=None, y_relativo=None),
        Contato("d", tem_bola=False, distancia_px=None, y_relativo=None),
    ]

    relatorio = measure(contatos)

    assert relatorio.com_bola == 1
    assert relatorio.encostada == 1
    assert relatorio.taxa == 0.25


def test_relatorio_vazio_nao_divide_por_zero() -> None:
    assert measure([]).taxa == 0.0


def test_o_quadro_do_contato_e_o_amostrado_mais_proximo() -> None:
    quadros = (_quadro(180, None), _quadro(204, None), _quadro(228, None))

    achado = quadro_do_contato(quadros, 207)

    assert achado is not None
    assert achado.frame_index == 204


def test_amostragem_que_nao_alcanca_o_contato_devolve_nada() -> None:
    """Sem o limite, clipe cuja amostra para 200 quadros antes contaria como se tivesse bola la."""
    quadros = (_quadro(0, None), _quadro(60, None))

    assert quadro_do_contato(quadros, 400) is None


def test_sem_quadro_nenhum_devolve_nada() -> None:
    assert quadro_do_contato((), 200) is None


def test_o_relatorio_carrega_a_escala_junto_com_a_taxa() -> None:
    """Taxa sem acaso nao tem escala: 0,29 parece "um terco certo" e e 2,2 vezes o sorteio."""
    from mlops.ball_quality import ACASO

    relatorio = measure(
        [Contato(str(i), tem_bola=True, distancia_px=0.0, y_relativo=0.6) for i in range(4)]
    )

    documento = relatorio.to_dict()
    assert documento["acaso"] == ACASO
    assert documento["vezes_o_acaso"] == round(1.0 / ACASO, 2)


def test_posicao_interpolada_nao_conta_como_percepcao() -> None:
    """A interpolacao e reta e o contato e onde a trajetoria vira: a reta passa por cima da virada.

    Contar o acerto dela premia o rastreador por inventar e esconde o quanto o detector realmente
    ve. Medido: na linha de base de 38 clipes, cerca de metade dos 28 com bola era inventada.
    """
    quadro = FramePerception(
        frame_index=200,
        timestamp_s=3.3,
        players=(PlayerDetection(box=JOGADOR, confidence=0.9),),
        ball=BallDetection(
            box=Box(118.0, 238.0, 130.0, 250.0), confidence=0.0, source="interpolated"
        ),
    )

    contato = contato_de("c1", quadro, altura=1080)

    assert contato.distancia_px == 0.0
    assert contato.encostada is False
    assert contato.encostada_incluindo_inventada is True


def test_a_taxa_frouxa_sai_ao_lado_da_estrita() -> None:
    """Publicar so a estrita esconderia de quanto foi a queda e por que ela aconteceu."""
    inventada = Contato("a", tem_bola=True, distancia_px=0.0, y_relativo=0.6, detectada=False)
    medida = Contato("b", tem_bola=True, distancia_px=0.0, y_relativo=0.6, detectada=True)

    relatorio = measure([inventada, medida])

    assert relatorio.encostada == 1
    assert relatorio.encostada_incluindo_inventada == 2
    assert relatorio.com_bola == 2
    assert relatorio.com_bola_detectada == 1


def test_o_limiar_de_gramado_e_o_horizonte_medido_e_nao_um_palpite() -> None:
    """0,45 sinalizava quase tudo: `y` menor e mais alto, entao arquibancada e `y` PEQUENO."""
    from mlops.ball_quality import FORA_DO_GRAMADO_Y

    sobre_grama = Contato("a", tem_bola=True, distancia_px=0.0, y_relativo=0.366)
    na_arquibancada = Contato("b", tem_bola=True, distancia_px=200.0, y_relativo=0.172)
    # 4997505 esta a 0,366 e e acerto de 0 px sobre grama; 4996643 esta a 0,172 na arquibancada

    assert FORA_DO_GRAMADO_Y == 0.28
    assert sobre_grama.fora_do_gramado is False
    assert na_arquibancada.fora_do_gramado is True
