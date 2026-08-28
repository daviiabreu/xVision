"""A feature `body_part`: qual perna balancou no contato, ou ausencia declarada.

Cada teste replanta um defeito que a versao anterior tem. O que mais custa nao levanta nada:
decidir a perna por distancia da bola ao tornozelo, quando os dois tornozelos ficam a 10,7 px um
do outro (mediana sobre os 38 clipes anotados) e a bola tem cerca de 24 px de diametro, faz a bola
encostar nos dois ao mesmo tempo, e o relatorio conta o empate como erro do modelo.
"""

from __future__ import annotations

from contracts import BallDetection, Box, FramePerception, Keypoint, Perception, PlayerDetection
from edge.features.body_part import PERNA_DIREITA, PERNA_ESQUERDA, BodyPart

CONTATO = 60
FPS = 60.0
PASSO = 4
"""Quadros nativos entre amostras de pose: o conjunto `dense` roda a 15 fps sobre 59,94."""


class _Shot:
    def __init__(self, frame_index: int = CONTATO) -> None:
        self.frame_index = frame_index
        self.shooter_track_id = None


def _keypoints(
    *,
    esquerdo: tuple[float, float],
    direito: tuple[float, float],
    cabeca: tuple[float, float] = (110.0, 95.0),
    confianca: float = 0.9,
) -> tuple[Keypoint, ...]:
    return (
        Keypoint("nose", cabeca[0], cabeca[1], confianca),
        Keypoint("ankle_left", esquerdo[0], esquerdo[1], confianca),
        Keypoint("ankle_right", direito[0], direito[1], confianca),
    )


def _jogador(keypoints: tuple[Keypoint, ...], track_id: int | None = 7) -> PlayerDetection:
    return PlayerDetection(
        box=Box(90.0, 90.0, 130.0, 220.0),
        confidence=0.9,
        track_id=track_id,
        keypoints=keypoints,
    )


def _quadro(
    indice: int,
    *,
    keypoints: tuple[Keypoint, ...] = (),
    bola: tuple[float, float] | None = None,
    jogadores: tuple[PlayerDetection, ...] | None = None,
) -> FramePerception:
    if jogadores is None:
        jogadores = (_jogador(keypoints),) if keypoints else ()
    detectada = (
        None
        if bola is None
        else BallDetection(
            box=Box(bola[0] - 5, bola[1] - 5, bola[0] + 5, bola[1] + 5), confidence=0.9
        )
    )
    return FramePerception(
        frame_index=indice, timestamp_s=indice / FPS, players=jogadores, ball=detectada
    )


def _perception(*quadros: FramePerception) -> Perception:
    return Perception(clip_id="c", frames=quadros)


def _balanco(
    *,
    quem: str,
    parado: tuple[float, float] = (100.0, 215.0),
    bola: tuple[float, float] | None = None,
    amplitude: float = 9.0,
) -> Perception:
    """Um tornozelo desloca `amplitude` px por amostra e o outro fica no lugar.

    E a assinatura fisica que a regra nova le: a perna que chuta balanca e a de apoio sustenta o
    peso. Sobre os clipes de perna da dobra de ajuste isso decide 9 de 13, contra 4 de 15 da regra
    por distancia.
    """
    quadros: list[FramePerception] = []
    for passo, indice in enumerate(range(CONTATO - 2 * PASSO, CONTATO + 2 * PASSO, PASSO)):
        movel = (parado[0] + 30.0 + amplitude * passo, parado[1])
        lados = (
            {"esquerdo": movel, "direito": parado}
            if quem == PERNA_ESQUERDA
            else {"esquerdo": parado, "direito": movel}
        )
        quadros.append(_quadro(indice, keypoints=_keypoints(**lados)))  # type: ignore[arg-type]
    if bola is not None:
        quadros.append(_quadro(CONTATO, bola=bola))
    return _perception(*quadros)


def test_a_perna_que_balanca_ganha_da_que_esta_mais_perto_da_bola() -> None:
    """O defeito central: com 10,7 px entre os tornozelos, a bola encosta nos dois.

    Aqui a bola esta colada no tornozelo esquerdo, que e o de apoio, e quem balanca e o direito.
    A regra por distancia responde a perna errada.
    """
    percepcao = _balanco(quem=PERNA_DIREITA, bola=(99.0, 216.0))

    valor = BodyPart().extract(percepcao, _Shot())

    assert valor.status == "ok"
    assert valor.value == PERNA_DIREITA


def test_a_perna_esquerda_tambem_e_decidida_pelo_balanco() -> None:
    percepcao = _balanco(quem=PERNA_ESQUERDA, bola=(121.0, 216.0))

    assert BodyPart().extract(percepcao, _Shot()).value == PERNA_ESQUERDA


def test_sem_bola_no_contato_a_feature_ainda_decide() -> None:
    """A velocidade nao depende da bola, e 7 dos 38 clipes anotados nao tem amostra perto.

    Antes disso a feature saia `unavailable` neles, e a cobertura publicada era 31 de 38.
    """
    valor = BodyPart().extract(_balanco(quem=PERNA_DIREITA), _Shot())

    assert valor.status == "ok"
    assert valor.value == PERNA_DIREITA


def test_a_bola_na_altura_da_cabeca_nao_decide_a_parte_do_corpo() -> None:
    """A classe CABECA nao e produzida, e a medida que fecha isso e do teto e nao de um limiar.

    A porta foi fechada, reaberta e fechada de novo, e a ultima medida e a que decide. A razao entre
    a distancia da bola ate a cabeca e ate o tornozelo mais proximo, no quadro do contato anotado e
    com o detector de bola ja corrigido, tem mediana 9,32 nas cabecadas e 4,51 nos chutes de perna:
    ela esta INVERTIDA em relacao a fisica. Varrendo todos os cortes possiveis e escolhendo o melhor
    olhando a resposta, o acerto e 24 de 32, exatamente igual a responder sempre PERNA.

    Medido no pipeline, a porta com corte em 1,0 disparou em 9 clipes, todos de perna, e em nenhuma
    das 8 cabecadas verdadeiras. Na dobra de ajuste, so velocidade da 9 de 20 e velocidade com porta
    da 4 de 20.
    """
    percepcao = _balanco(quem=PERNA_DIREITA, bola=(110.0, 92.0))

    valor = BodyPart().extract(percepcao, _Shot())

    assert valor.value == PERNA_DIREITA


def test_sem_keypoint_na_janela_sai_unavailable_com_motivo() -> None:
    """A regra que nao pode ser quebrada: nunca zero e nunca silencio."""
    valor = BodyPart().extract(_perception(_quadro(CONTATO, bola=(110.0, 200.0))), _Shot())

    assert valor.status == "unavailable"
    assert valor.value is None
    assert valor.reason


def test_keypoint_de_baixa_confianca_nao_vira_medida() -> None:
    """Tornozelo ocluso e ausencia, e nao perna: o defeito nao levanta nada sem esta guarda."""
    quadros = [
        _quadro(
            indice,
            keypoints=_keypoints(esquerdo=(100.0, 215.0), direito=(120.0, 215.0), confianca=0.05),
        )
        for indice in range(CONTATO - PASSO, CONTATO + 2 * PASSO, PASSO)
    ]
    valor = BodyPart().extract(_perception(*quadros), _Shot())

    assert valor.status == "unavailable"
    assert "confianca" in (valor.reason or "")


def test_um_tornozelo_ocluso_impede_a_comparacao() -> None:
    """Responder o tornozelo que sobrou seria responder o que o estimador enxergou melhor.

    E o vies aponta para a resposta errada: quem chuta borra e sai com confianca mais baixa, entao
    o unico ponto que resta tende a ser o pe de apoio.
    """
    quadros = [
        _quadro(
            indice,
            keypoints=(
                Keypoint("ankle_left", 100.0, 215.0, 0.05),
                Keypoint("ankle_right", 120.0 + indice - CONTATO, 215.0, 0.9),
            ),
        )
        for indice in range(CONTATO - PASSO, CONTATO + 2 * PASSO, PASSO)
    ]

    valor = BodyPart().extract(_perception(*quadros), _Shot())

    assert valor.status == "unavailable"
    assert "tornozelos" in (valor.reason or "")


def test_empate_de_velocidade_sai_com_confianca_baixa() -> None:
    """Os dois tornozelos com a mesma velocidade e um empate, e reportar separacao seria mentir."""
    quadros = [
        _quadro(
            indice,
            keypoints=_keypoints(
                esquerdo=(100.0 + (indice - CONTATO), 215.0),
                direito=(120.0 + (indice - CONTATO), 215.0),
            ),
        )
        for indice in range(CONTATO - PASSO, CONTATO + 2 * PASSO, PASSO)
    ]

    valor = BodyPart().extract(_perception(*quadros), _Shot())

    assert valor.status == "ok"
    assert valor.confidence is not None and valor.confidence < 0.05


def test_separacao_clara_de_velocidade_sai_com_confianca_alta() -> None:
    valor = BodyPart().extract(_balanco(quem=PERNA_DIREITA), _Shot())

    assert valor.confidence is not None and valor.confidence > 0.5


def test_o_valor_cabe_no_dominio_que_o_rotulo_declara() -> None:
    valor = BodyPart().extract(_balanco(quem=PERNA_ESQUERDA), _Shot())

    assert valor.value in {PERNA_ESQUERDA, PERNA_DIREITA}


def test_a_evidencia_diz_de_que_quadros_a_medida_saiu() -> None:
    """Sem isso, ninguem distingue uma medida no contato de uma feita meio segundo depois."""
    valor = BodyPart().extract(_balanco(quem=PERNA_DIREITA), _Shot())

    assert valor.evidence["frame_do_corpo"] == CONTATO
    assert valor.evidence["quadros_na_janela"] == 4
    assert set(valor.evidence["velocidade_px_s"]) == {  # type: ignore[arg-type]
        PERNA_ESQUERDA,
        PERNA_DIREITA,
    }


def test_a_janela_nao_atravessa_o_clipe_inteiro() -> None:
    """Meio segundo depois do contato quem balanca e a corrida, e a medida vira ruido.

    Aqui o tornozelo esquerdo so se move um segundo antes do contato. Uma janela sem limite o
    elegeria; a de 0,25 s nao enxerga aquele trecho.
    """
    longe = [
        _quadro(
            indice,
            keypoints=_keypoints(esquerdo=(100.0 + 40 * passo, 215.0), direito=(120.0, 215.0)),
        )
        for passo, indice in enumerate(range(CONTATO - 60, CONTATO - 48, PASSO))
    ]
    perto = [
        _quadro(
            indice,
            keypoints=_keypoints(esquerdo=(100.0, 215.0), direito=(120.0 + 9.0 * passo, 215.0)),
        )
        for passo, indice in enumerate(range(CONTATO - PASSO, CONTATO + 2 * PASSO, PASSO))
    ]

    valor = BodyPart().extract(_perception(*longe, *perto), _Shot())

    assert valor.value == PERNA_DIREITA


def test_entre_dois_corpos_mede_o_que_esta_mais_perto_da_bola() -> None:
    """A escolha do finalizador e feita aqui, e nao herdada da especie pose.

    O modulo de pose deste repositorio mede so o finalizador, mas isso e escolha dele e nao
    promessa do contrato. Sem esta guarda, trocar o modulo passaria a medir o marcador.
    """
    quadros: list[FramePerception] = []
    for passo, indice in enumerate(range(CONTATO - PASSO, CONTATO + 2 * PASSO, PASSO)):
        finalizador = _jogador(
            _keypoints(esquerdo=(100.0, 215.0), direito=(120.0 + 9.0 * passo, 215.0)), track_id=1
        )
        marcador = PlayerDetection(
            box=Box(600.0, 90.0, 640.0, 220.0),
            confidence=0.9,
            track_id=2,
            keypoints=_keypoints(
                esquerdo=(610.0 - 9.0 * passo, 215.0), direito=(630.0, 215.0), cabeca=(620.0, 95.0)
            ),
        )
        quadros.append(_quadro(indice, jogadores=(marcador, finalizador)))
    quadros.append(_quadro(CONTATO, bola=(122.0, 216.0)))

    valor = BodyPart().extract(_perception(*quadros), _Shot())

    assert valor.status == "ok"
    assert valor.value == PERNA_DIREITA


def test_com_dois_corpos_e_sem_bola_a_feature_recusa_escolher() -> None:
    """Sem bola nao ha de quem medir a distancia, e chutar o mais central poe a medida no corpo
    errado sem que nada rio abaixo distinga isso de um acerto."""
    quadros: list[FramePerception] = []
    for passo, indice in enumerate(range(CONTATO - PASSO, CONTATO + 2 * PASSO, PASSO)):
        um = _jogador(
            _keypoints(esquerdo=(100.0, 215.0), direito=(120.0 + 9.0 * passo, 215.0)), track_id=1
        )
        outro = PlayerDetection(
            box=Box(600.0, 90.0, 640.0, 220.0),
            confidence=0.9,
            track_id=2,
            keypoints=_keypoints(
                esquerdo=(610.0 - 9.0 * passo, 215.0), direito=(630.0, 215.0), cabeca=(620.0, 95.0)
            ),
        )
        quadros.append(_quadro(indice, jogadores=(um, outro)))

    valor = BodyPart().extract(_perception(*quadros), _Shot())

    assert valor.status == "unavailable"
    assert "ball" in (valor.reason or "")


def test_uma_amostra_de_pose_so_nao_da_velocidade() -> None:
    """Velocidade precisa de dois quadros, e inventar zero faria o empate virar resposta."""
    percepcao = _perception(
        _quadro(CONTATO, keypoints=_keypoints(esquerdo=(100.0, 215.0), direito=(120.0, 215.0))),
        _quadro(CONTATO, bola=(122.0, 216.0)),
    )

    valor = BodyPart().extract(percepcao, _Shot())

    assert valor.status == "unavailable"
    assert valor.reason


def test_sem_track_id_o_corpo_e_reencontrado_pela_caixa() -> None:
    """A trilha nao pode depender do rastreador: sem `track_id` ela casa pela caixa mais proxima.

    Sem isso, cada quadro entregaria o corpo que estivesse primeiro na lista, e a velocidade sairia
    da diferenca entre o tornozelo de um jogador e o de outro, que e ruido com cara de balanco.
    """
    quadros: list[FramePerception] = []
    for passo, indice in enumerate(range(CONTATO - PASSO, CONTATO + 2 * PASSO, PASSO)):
        finalizador = _jogador(
            _keypoints(esquerdo=(100.0, 215.0), direito=(120.0 + 9.0 * passo, 215.0)), track_id=None
        )
        vizinho = PlayerDetection(
            box=Box(300.0, 90.0, 340.0, 220.0),
            confidence=0.9,
            track_id=None,
            keypoints=_keypoints(
                esquerdo=(310.0 - 30.0 * passo, 215.0), direito=(330.0, 215.0), cabeca=(320.0, 95.0)
            ),
        )
        ordem = (finalizador, vizinho) if passo % 2 else (vizinho, finalizador)
        quadros.append(_quadro(indice, jogadores=ordem))
    quadros.append(_quadro(CONTATO, bola=(122.0, 216.0)))

    valor = BodyPart().extract(_perception(*quadros), _Shot())

    assert valor.status == "ok"
    assert valor.value == PERNA_DIREITA


def test_finalizador_parado_sai_com_confianca_zero() -> None:
    """Dois tornozelos a zero px/s nao elegem perna nenhuma, e a confianca precisa dizer isso."""
    quadros = [
        _quadro(indice, keypoints=_keypoints(esquerdo=(100.0, 215.0), direito=(120.0, 215.0)))
        for indice in range(CONTATO - PASSO, CONTATO + 2 * PASSO, PASSO)
    ]

    valor = BodyPart().extract(_perception(*quadros), _Shot())

    assert valor.status == "ok"
    assert valor.confidence == 0.0


def test_quadros_com_o_mesmo_instante_nao_viram_divisao_por_zero() -> None:
    """A percepcao vem de quem chama, e dois quadros com o mesmo `timestamp_s` sao entrada valida.

    Sem a guarda, o intervalo zero levantaria dentro da feature e o clipe inteiro sairia
    `unavailable` com o nome de uma excecao no lugar do motivo.
    """
    keypoints = _keypoints(esquerdo=(100.0, 215.0), direito=(120.0, 215.0))
    quadros = [
        FramePerception(
            frame_index=indice,
            timestamp_s=CONTATO / FPS,
            players=(_jogador(keypoints),),
        )
        for indice in (CONTATO, CONTATO + PASSO)
    ]

    valor = BodyPart().extract(_perception(*quadros), _Shot())

    assert valor.status == "unavailable"
    assert valor.reason


def test_um_corpo_so_dispensa_a_bola() -> None:
    """Declarar `ball` em requires barraria a feature antes de a regra do corpo unico rodar.

    Sao 7 dos 38 clipes anotados que nao tem amostra de bola a menos de 12 quadros do contato, e
    todos os 7 tem exatamente um corpo com keypoint, ou seja a bola nao acrescenta nada ali.
    """
    from edge.features.body_part import BodyPart

    assert "ball" not in BodyPart().spec.requires
    assert "pose" in BodyPart().spec.requires
