"""A geometria do contato: acha a quebra, e se cala quando nao ha quebra.

Cada teste replanta um defeito que a implementacao ingenua tem. Trajetoria constante e bola parada
existem porque um `argmax` sem piso sempre devolve alguma coisa, e alguma coisa e o modo de falha
que o projeto inteiro combate: numero inventado com a mesma cara de numero medido.
"""

from __future__ import annotations

import math

from edge.geometry.contact import Observation, best_contact, find_contact, rank_contacts

PASSO = 12
"""O `coarse` amostra a 5 fps, e a 59,94 fps isso e um quadro a cada 12."""

FRAME_COUNT = 400


def _reta(
    inicio: int, quantos: int, x0: float, y0: float, vx: float, vy: float
) -> list[Observation]:
    return [
        Observation(inicio + i * PASSO, x0 + vx * i * PASSO, y0 + vy * i * PASSO)
        for i in range(quantos)
    ]


def _quebra_em(quadro: int) -> list[Observation]:
    """Passe para a direita ate `quadro`, chute para a esquerda e para cima depois dele."""
    antes = _reta(0, quadro // PASSO + 1, 100.0, 500.0, 3.0, 0.0)
    ponto = antes[-1]
    depois = _reta(ponto.frame_index, 4, ponto.x, ponto.y, -6.0, -4.0)[1:]
    return antes + depois


def test_acha_a_quebra_limpa_no_quadro_certo() -> None:
    contato = find_contact(_quebra_em(48), frame_count=FRAME_COUNT)

    assert contato is not None
    assert contato.frame_index == 48
    assert contato.turn_deg > 90.0
    assert contato.speed_ratio > 1.0


def test_bola_parada_nao_produz_contato() -> None:
    parada = [Observation(i * PASSO, 100.0, 500.0) for i in range(8)]

    assert find_contact(parada, frame_count=FRAME_COUNT) is None


def test_trajetoria_constante_nao_produz_contato() -> None:
    assert find_contact(_reta(0, 8, 100.0, 500.0, 3.0, 1.0), frame_count=FRAME_COUNT) is None


def test_trajetoria_curta_demais_nao_produz_contato() -> None:
    assert find_contact(_reta(0, 3, 100.0, 500.0, 3.0, 1.0), frame_count=FRAME_COUNT) is None


def test_sem_deteccao_nenhuma_nao_produz_contato() -> None:
    assert find_contact([], frame_count=FRAME_COUNT) is None


def test_lacuna_de_deteccao_nao_vira_aceleracao_falsa() -> None:
    """A bola sumir e reaparecer no dobro da distancia e a mesma velocidade, nao o dobro dela.

    Dividir pela posicao na lista em vez de pelo indice de quadro faria esta trajetoria, que e
    retilinea e uniforme, pontuar como quebra.
    """
    reta = _reta(0, 8, 100.0, 500.0, 3.0, 1.0)
    com_lacuna = reta[:3] + reta[5:]

    assert find_contact(com_lacuna, frame_count=FRAME_COUNT) is None


def test_quadro_repetido_nao_divide_por_zero() -> None:
    """Detector que devolve dois candidatos no mesmo quadro daria delta zero na velocidade."""
    pontos = _quebra_em(48)
    duplicado = pontos + [Observation(48, 999.0, 999.0)]

    contato = find_contact(duplicado, frame_count=FRAME_COUNT)

    assert contato is not None
    assert contato.frame_index == 48


def test_observacao_fora_do_clipe_e_descartada() -> None:
    pontos = _quebra_em(48) + [Observation(FRAME_COUNT + 10, 0.0, 0.0), Observation(-5, 0.0, 0.0)]

    contato = find_contact(pontos, frame_count=FRAME_COUNT)

    assert contato is not None
    assert 0 <= contato.frame_index < FRAME_COUNT


def test_candidatos_restringem_a_resposta() -> None:
    """Quem decide onde a bola esta perto de um jogador e o modulo, e a geometria obedece."""
    pontos = _quebra_em(48)
    longe_da_quebra = {p.frame_index for p in pontos if p.frame_index not in (48,)}

    contato = find_contact(pontos, frame_count=FRAME_COUNT, candidates=longe_da_quebra)

    assert contato is None or contato.frame_index != 48


def test_candidato_vazio_nao_produz_contato() -> None:
    assert find_contact(_quebra_em(48), frame_count=FRAME_COUNT, candidates=set()) is None


def test_refino_devolve_instante_entre_amostras() -> None:
    """Sem o refino, o erro fica cravado em ate meio passo de amostragem por construcao.

    A quebra real esta no quadro 54, que nao e amostrado: as amostras vizinhas sao 48 e 60. O
    cruzamento das duas retas precisa cair mais perto de 54 do que de qualquer uma das amostras.
    """
    antes = _reta(0, 5, 100.0, 500.0, 3.0, 0.0)
    virada = Observation(54, 100.0 + 3.0 * 54, 500.0)
    depois = [Observation(54 + d, virada.x - 6.0 * d, virada.y - 4.0 * d) for d in (6, 18, 30)]
    amostrado = [o for o in antes + depois if o.frame_index % PASSO == 0 or o.frame_index == 60]

    contato = find_contact(amostrado, frame_count=FRAME_COUNT)

    assert contato is not None
    assert abs(contato.frame_index - 54) < abs(48 - 54)


def test_confianca_cai_quando_toda_amostra_e_quebra() -> None:
    """Zigue-zague e ruido de deteccao, nao finalizacao, e a confianca precisa dizer isso."""
    zigue = [
        Observation(i * PASSO, 100.0 + i * PASSO * 3.0, 500.0 + (40.0 if i % 2 else -40.0))
        for i in range(10)
    ]

    contato = find_contact(zigue, frame_count=FRAME_COUNT)

    assert contato is None or contato.confidence < 0.5


def test_confianca_alta_quando_ha_uma_quebra_so() -> None:
    contato = find_contact(_quebra_em(48), frame_count=FRAME_COUNT)

    assert contato is not None
    assert contato.confidence > 0.5


def test_bola_parada_de_um_lado_nao_vira_virada_de_noventa_graus() -> None:
    """Vetor nulo nao tem direcao, e chamar isso de virada seria inventar angulo."""
    parada = [Observation(i * PASSO, 100.0, 500.0) for i in range(4)]
    saida = _reta(parada[-1].frame_index, 4, 100.0, 500.0, 8.0, 0.0)[1:]

    contato = find_contact(parada + saida, frame_count=FRAME_COUNT)

    assert contato is not None
    assert math.isclose(contato.turn_deg, 0.0, abs_tol=1e-6)


def test_rank_contacts_ordena_da_maior_quebra_para_a_menor() -> None:
    """Sem ordem declarada, "o contato certo estava em segundo lugar" nao quer dizer nada."""
    ordenados = rank_contacts(_quebra_em(48), frame_count=FRAME_COUNT)

    assert len(ordenados) > 1
    assert [c.score for c in ordenados] == sorted((c.score for c in ordenados), reverse=True)


def test_a_escolha_e_o_primeiro_da_lista_ordenada() -> None:
    """`find_contact` e `rank_contacts` precisam decidir pelo mesmo score.

    O defeito que isto impede e o pior tipo de defeito de diagnostico: duas implementacoes do
    mesmo score divergindo, e a divergencia sendo lida como conclusao sobre o modelo.
    """
    bola = _quebra_em(48)

    contato = find_contact(bola, frame_count=FRAME_COUNT)
    ordenados = rank_contacts(bola, frame_count=FRAME_COUNT)

    assert contato is not None
    assert contato.frame_index == ordenados[0].frame_index
    assert contato.turn_deg == ordenados[0].turn_deg


def test_rank_contacts_respeita_o_filtro_de_candidatos() -> None:
    bola = _quebra_em(48)
    permitidos = {48}

    ordenados = rank_contacts(bola, frame_count=FRAME_COUNT, candidates=permitidos)

    assert {c.sample_index for c in ordenados} == permitidos


def test_rank_contacts_devolve_vazio_quando_a_trajetoria_e_curta() -> None:
    """Trajetoria curta e clipe com bola mal detectada, e nao uso errado da funcao."""
    assert rank_contacts(_reta(0, 3, 0.0, 0.0, 1.0, 0.0), frame_count=FRAME_COUNT) == ()
    assert best_contact(()) is None


def test_a_lista_guarda_o_quadro_refinado_e_o_amostrado() -> None:
    """O refino cai entre amostras, e sem o bruto nao da para saber de qual amostra ele veio."""
    ordenados = rank_contacts(_quebra_em(48), frame_count=FRAME_COUNT)

    assert all(c.sample_index % PASSO == 0 for c in ordenados)
    escolhido = next(c for c in ordenados if c.sample_index == 48)
    assert abs(escolhido.frame_index - escolhido.sample_index) <= PASSO
