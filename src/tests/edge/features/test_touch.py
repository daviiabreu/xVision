"""Quem tocou a bola, sobre a percepcao ja montada.

O defeito que estes testes replantam nao levanta excecao: uma associacao errada devolve um
`track_id` plausivel, e a comparacao com o finalizador sai estavel e errada. Estavel e errado e
pior do que ausente, porque ninguem desconfia de um numero que nao varia.

A primeira versao deste modulo tinha exatamente esse defeito no centro, e a suite passava: o toque
anterior era o jogador mais proximo no quadro anterior mais recente, e numa percepcao densa isso e
o proprio finalizador. Os testes de entao usavam tres quadros bem separados, cadencia que nenhum
conjunto de producao tem, entao a suite nunca viu o caso. `TestPasseDenso` existe por causa disso.
"""

from __future__ import annotations

from contracts import BallDetection, Box, FramePerception, Perception, PlayerDetection
from edge.features.touch import (
    MARGEM_DE_EMPATE,
    previous_touch,
    shooter_at_contact,
    telemetry,
)
from edge.features.touch import _empatado as _empatado  # pyright: ignore[reportPrivateUsage]
from edge.features.touch import _passo as _passo  # pyright: ignore[reportPrivateUsage]


def _bola(x: float, y: float, source: str | None = "detected") -> BallDetection:
    """O contrato ja exige `confidence=0.0` e `gap_frames` positivo em posicao interpolada."""
    interpolada = source == "interpolated"
    return BallDetection(
        Box(x - 5, y - 5, x + 5, y + 5),
        0.0 if interpolada else 0.9,
        source=source,
        gap_frames=1 if interpolada else None,
    )


def _jogador(x: float, y: float, track: int | None, *, team: str | None = None) -> PlayerDetection:
    """Caixa de 40 por 100, que e a ordem de grandeza de um jogador em quadro de 1080.

    Com altura cem, distancia normalizada e distancia em pixel dividida por cem, entao `0,10` de
    distancia normalizada sao dez pixels da borda da caixa ao centro da bola. Os numeros dos testes
    de margem sao escolhidos nessa escala de proposito, para a conta ficar legivel na leitura.
    """
    return PlayerDetection(Box(x - 20, y - 50, x + 20, y + 50), 0.8, track_id=track, team=team)


def _quadro(
    indice: int, bola: BallDetection | None, *jogadores: PlayerDetection
) -> FramePerception:
    return FramePerception(
        frame_index=indice, timestamp_s=indice / 59.94, players=jogadores, ball=bola
    )


def _percepcao(*quadros: FramePerception) -> Perception:
    return Perception(clip_id="c", frames=quadros)


def _passe_denso() -> Perception:
    """Passe do track 2 para o track 7 em onze quadros consecutivos, na cadencia de producao.

    A bola fica no pe do passador nos quadros 0 a 3, voa nos quadros 4 a 6 fora do raio dos dois, e
    no quadro 7 ja entrou no raio do finalizador ainda em voo. E esse quadro 7 que derrubava a
    versao por proximidade.
    """
    rota = {0: 100.0, 1: 100.0, 2: 100.0, 3: 100.0, 4: 250.0, 5: 400.0, 6: 550.0, 7: 640.0}
    return _percepcao(
        *(
            _quadro(
                i,
                _bola(rota.get(i, 700.0), 100),
                _jogador(100, 100, 2, team="A"),
                _jogador(700, 100, 7, team="B"),
            )
            for i in range(11)
        )
    )


class TestPasseDenso:
    """O caso que derrubou a primeira versao, na cadencia real de um conjunto de producao.

    Passe do track 2 para o track 7 ao longo de onze quadros consecutivos. No quadro 7 a bola em
    voo ja entrou no raio do finalizador, entao proximidade no quadro anterior devolve o proprio
    finalizador e a jogada individual come o passe.
    """

    def test_o_toque_anterior_e_o_passador_e_nao_o_finalizador(self) -> None:
        percepcao = _passe_denso()

        finalizador = shooter_at_contact(percepcao, 10)
        toque = previous_touch(percepcao, 10)

        assert finalizador is not None and finalizador.player_track_id == 7
        assert toque is not None and toque.player_track_id == 2
        assert toque.player_track_id != finalizador.player_track_id
        assert toque.team == "A"

    def test_o_toque_anterior_termina_antes_de_comecar_o_episodio_do_contato(self) -> None:
        """Quadro 3 e o ultimo em que a bola estava no pe do passador; 4 a 6 ela esta em voo."""
        toque = previous_touch(_passe_denso(), 10)

        assert toque is not None and toque.frame_index == 3

    def test_o_contato_refinado_depois_da_ultima_amostra_ainda_da_o_passador(self) -> None:
        """O caso normal em producao, e o que distingue "antes do episodio" de "antes do quadro".

        `find_contact` refina a quebra pela intersecao das retas e devolve um quadro que nao e
        amostra, com frequencia depois da ultima amostra do episodio do contato. Filtrar por
        `fim < quadro_do_contato` em vez de `fim < inicio_do_episodio` deixa o proprio episodio do
        contato entrar na lista, e o toque anterior volta a ser o finalizador.
        """
        percepcao = _passe_denso()

        finalizador = shooter_at_contact(percepcao, 12)
        toque = previous_touch(percepcao, 12)

        assert finalizador is not None and finalizador.player_track_id == 7
        assert toque is not None and toque.player_track_id == 2 and toque.frame_index == 3

    def test_a_bola_em_voo_ja_no_raio_do_finalizador_nao_conta_como_toque_dele(self) -> None:
        """No quadro 7 a bola esta a 0,40 altura da caixa do 7, dentro do raio, e ainda em voo.

        Ela pertence ao episodio do contato, e nao a um toque anterior. Se ela abrisse episodio
        proprio, o toque anterior seria o finalizador e o passe desapareceria.
        """
        percepcao = _passe_denso()

        assert telemetry(percepcao, 10).episodios == 2


class TestConducao:
    """Conducao da varios episodios do mesmo dono, e isso e a resposta certa, nao um defeito."""

    def _percepcao(self) -> Perception:
        posicoes = {0: (100.0, 100.0), 1: (100.0, 100.0), 2: (300.0, 500.0), 3: (300.0, 500.0)}
        return _percepcao(
            *(
                _quadro(i, _bola(x_bola, 100), _jogador(x_jogador, 100, 5, team="A"))
                for i, (x_jogador, x_bola) in {
                    **posicoes,
                    4: (480.0, 500.0),
                    5: (480.0, 500.0),
                }.items()
            )
        )

    def test_o_toque_anterior_e_do_mesmo_jogador(self) -> None:
        percepcao = self._percepcao()

        finalizador = shooter_at_contact(percepcao, 5)
        toque = previous_touch(percepcao, 5)

        assert finalizador is not None and finalizador.player_track_id == 5
        assert toque is not None and toque.player_track_id == 5
        assert toque.frame_index == 1

    def test_a_bola_colada_o_clipe_inteiro_nao_tem_toque_anterior(self) -> None:
        """O limite conhecido do desenho: um dominio que nunca sai do raio e um episodio so.

        Declarar ausente aqui e correto, e e o que o raio permite hoje. Se `RAIO_DO_TOQUE` fosse
        calibrado para baixo, esta mesma cena viraria varios episodios.
        """
        percepcao = _percepcao(
            *(_quadro(i, _bola(100, 100), _jogador(100, 100, 5)) for i in range(6))
        )

        assert telemetry(percepcao, 5).episodios == 1
        assert previous_touch(percepcao, 5) is None
        assert telemetry(percepcao, 5).motivo_do_toque_anterior == "absent"


class TestShooterAtContact:
    def test_devolve_o_jogador_mais_proximo_da_bola(self) -> None:
        percepcao = _percepcao(
            _quadro(10, _bola(100, 100), _jogador(100, 100, 7), _jogador(900, 100, 8))
        )

        evidencia = shooter_at_contact(percepcao, 10)

        assert evidencia is not None
        assert evidencia.player_track_id == 7
        assert evidencia.source == "detected"

    def test_a_bola_fora_da_caixa_e_associacao_e_nao_deteccao(self) -> None:
        percepcao = _percepcao(_quadro(10, _bola(100, 100), _jogador(170, 100, 7)))

        evidencia = shooter_at_contact(percepcao, 10)

        assert evidencia is not None and evidencia.source == "associated"

    def test_bola_ausente_devolve_nada(self) -> None:
        percepcao = _percepcao(_quadro(10, None, _jogador(100, 100, 7)))

        assert shooter_at_contact(percepcao, 10) is None

    def test_bola_interpolada_nao_sustenta_toque(self) -> None:
        """A interpolacao e uma reta entre duas deteccoes, e o toque e onde a reta deixa de ser
        reta. Associar um jogador a uma posicao inventada produz um toque que a imagem nao
        mostra."""
        percepcao = _percepcao(_quadro(10, _bola(100, 100, "interpolated"), _jogador(100, 100, 7)))

        assert shooter_at_contact(percepcao, 10) is None

    def test_bola_sem_procedencia_declarada_tambem_nao(self) -> None:
        percepcao = _percepcao(_quadro(10, _bola(100, 100, None), _jogador(100, 100, 7)))

        assert shooter_at_contact(percepcao, 10) is None

    def test_jogador_sem_track_nao_e_candidato(self) -> None:
        """Sem `track_id` nao ha como comparar com o toque anterior, que e o proposito do seam."""
        percepcao = _percepcao(_quadro(10, _bola(100, 100), _jogador(100, 100, None)))

        assert shooter_at_contact(percepcao, 10) is None

    def test_ninguem_dentro_do_raio_devolve_nada(self) -> None:
        percepcao = _percepcao(_quadro(10, _bola(100, 100), _jogador(2000, 900, 7)))

        assert shooter_at_contact(percepcao, 10) is None

    def test_empate_devolve_nada_em_vez_de_escolher_o_menor_track(self) -> None:
        """Escolher pelo numero produziria resultado estavel e errado, que ninguem desconfia."""
        percepcao = _percepcao(
            _quadro(10, _bola(100, 100), _jogador(60, 100, 3), _jogador(140, 100, 9))
        )

        assert shooter_at_contact(percepcao, 10) is None

    def test_confianca_cai_com_a_distancia(self) -> None:
        perto = shooter_at_contact(
            _percepcao(_quadro(10, _bola(100, 100), _jogador(100, 100, 7))), 10
        )
        distante = shooter_at_contact(
            _percepcao(_quadro(10, _bola(100, 100), _jogador(100, 190, 7))), 10
        )

        assert perto is not None and distante is not None
        assert perto.confidence > distante.confidence

    def test_caixa_de_altura_zero_nao_divide_por_zero(self) -> None:
        degenerado = PlayerDetection(Box(90, 100, 110, 100), 0.8, track_id=7)
        percepcao = _percepcao(_quadro(10, _bola(100, 100), degenerado))

        assert shooter_at_contact(percepcao, 10) is None


class TestMargemDeEmpate:
    """A margem so tem efeito com os dois candidatos dentro do raio, e e la que ela e fixada.

    A suite anterior media a margem com o segundo candidato a 1,32 altura, ou seja fora do raio e
    filtrado antes da comparacao. Com isso `MARGEM_DE_EMPATE = 1000` e `= 1e-12` passavam as duas.
    Os pares aqui cercam o valor por cima e por baixo, entao mudar a constante quebra um teste.
    """

    def _disputa(self, distancia_do_segundo: float) -> Perception:
        """Primeiro a 0,10 altura da bola; segundo a `distancia_do_segundo`, do outro lado."""
        return _percepcao(
            _quadro(
                10,
                _bola(100, 100),
                _jogador(130, 100, 3),
                _jogador(100 - 20 - distancia_do_segundo * 100, 100, 9),
            )
        )

    def test_diferenca_de_quatro_centesimos_e_empate(self) -> None:
        assert shooter_at_contact(self._disputa(0.14), 10) is None

    def test_diferenca_de_seis_centesimos_nao_e_empate(self) -> None:
        evidencia = shooter_at_contact(self._disputa(0.16), 10)

        assert evidencia is not None and evidencia.player_track_id == 3

    def test_dois_dentro_do_raio_com_vencedor_claro_nao_e_empate(self) -> None:
        """Meia altura de diferenca, os dois bem dentro do raio: se isto vira empate a margem esta
        grande demais para servir de qualquer coisa."""
        percepcao = _percepcao(
            _quadro(10, _bola(100, 100), _jogador(100, 100, 3), _jogador(170, 100, 9))
        )

        evidencia = shooter_at_contact(percepcao, 10)

        assert evidencia is not None and evidencia.player_track_id == 3

    def test_distancia_zero_nos_dois_e_empate(self) -> None:
        """Bola dentro das duas caixas: a margem relativa nao conseguia dividir por zero aqui."""
        percepcao = _percepcao(
            _quadro(10, _bola(100, 100), _jogador(100, 100, 3), _jogador(110, 100, 9))
        )

        assert shooter_at_contact(percepcao, 10) is None

    def test_um_so_com_distancia_zero_nao_e_empate(self) -> None:
        percepcao = _percepcao(
            _quadro(10, _bola(100, 100), _jogador(100, 100, 3), _jogador(400, 100, 9))
        )

        evidencia = shooter_at_contact(percepcao, 10)

        assert evidencia is not None and evidencia.player_track_id == 3


class TestEscalaDaJanela:
    def test_caixa_cortada_nao_perde_por_ser_mais_baixa(self) -> None:
        """Normalizar pela altura do candidato penalizava quem esta mais difícil de detectar.

        Os dois jogadores estao a dez pixels da bola. Pela mediana da janela as distancias
        normalizadas sao iguais e a disputa e empate. Pela altura do candidato o de caixa cortada
        ficaria ao dobro da distancia e perderia, mesmo estando igualmente perto na imagem.
        """
        cortado = PlayerDetection(Box(50, 75, 90, 125), 0.8, track_id=9)
        percepcao = _percepcao(_quadro(10, _bola(100, 100), _jogador(130, 100, 3), cortado))

        assert shooter_at_contact(percepcao, 10) is None

    def test_a_escala_vem_de_toda_a_janela_e_nao_do_quadro(self) -> None:
        """Um quadro sem jogador nenhum ainda tem escala, porque a mediana e da janela."""
        percepcao = _percepcao(
            _quadro(0, _bola(100, 100), _jogador(100, 100, 7)),
            _quadro(1, _bola(100, 100)),
            _quadro(2, _bola(100, 100), _jogador(100, 100, 7)),
        )

        assert telemetry(percepcao, 2).frames_com_jogador_perto == 2


class TestToleranciaDoContato:
    """O contato refinado cai entre amostras, e aceitar o episodio mais proximo e necessario.

    Sem limite, porem, o mais proximo responde sempre: uma posse que acabou muito antes do contato
    devolveria aquele jogador com cara de finalizador. A tolerancia sai do passo de amostragem
    observado, entao ela vale dois quadros no passo um e oito quadros no passo quatro.
    """

    def _passo(self, passo: int) -> Perception:
        return _percepcao(
            *(_quadro(i * passo, _bola(100, 100), _jogador(100, 100, 7)) for i in range(4))
        )

    def test_o_contato_entre_amostras_resolve_pelo_episodio(self) -> None:
        evidencia = shooter_at_contact(self._passo(4), 10)

        assert evidencia is not None and evidencia.player_track_id == 7

    def test_no_limite_da_tolerancia_ainda_responde(self) -> None:
        assert shooter_at_contact(self._passo(4), 20) is not None
        assert shooter_at_contact(self._passo(1), 5) is not None

    def test_alem_da_tolerancia_devolve_ausente(self) -> None:
        assert shooter_at_contact(self._passo(4), 21) is None
        assert shooter_at_contact(self._passo(1), 6) is None

    def test_a_tolerancia_acompanha_o_passo_e_nao_e_constante(self) -> None:
        """O quadro 6 esta dentro da tolerancia no passo quatro e fora dela no passo um."""
        assert shooter_at_contact(self._passo(4), 6) is not None
        assert shooter_at_contact(self._passo(1), 6) is None

    def test_uma_posse_muito_antiga_nao_vira_finalizador(self) -> None:
        percepcao = _percepcao(
            _quadro(0, _bola(100, 100), _jogador(100, 100, 7)),
            _quadro(1, _bola(100, 100), _jogador(100, 100, 7)),
            _quadro(600, _bola(5000, 100), _jogador(100, 100, 7)),
        )

        assert shooter_at_contact(percepcao, 600) is None


class TestPreviousTouch:
    def test_o_proprio_episodio_do_contato_nao_conta(self) -> None:
        """Se contasse, a comparacao com o finalizador seria verdadeira por construcao."""
        percepcao = _percepcao(_quadro(10, _bola(100, 100), _jogador(100, 100, 7)))

        assert previous_touch(percepcao, 10) is None

    def test_amostra_interpolada_no_meio_nao_parte_o_episodio(self) -> None:
        """Bola interpolada nao diz que a posse acabou, diz que ninguem mediu.

        A primeira versao dos episodios tratava isso como fim de episodio, e a suite abencoava:
        o teste anterior punha o mesmo dono nos tres quadros, entao a divisao em dois episodios
        era invisivel. Com dono unico nao ha toque anterior a declarar.
        """
        percepcao = _percepcao(
            _quadro(0, _bola(100, 100), _jogador(100, 100, 2, team="B")),
            _quadro(1, _bola(100, 100, "interpolated"), _jogador(100, 100, 2)),
            _quadro(2, _bola(100, 100), _jogador(100, 100, 2)),
        )

        assert telemetry(percepcao, 2).episodios == 1
        assert previous_touch(percepcao, 2) is None

    def test_bola_medida_com_ninguem_perto_encerra_o_episodio(self) -> None:
        """Evidencia de ausencia, e nao ausencia de evidencia: alguem mediu e a bola estava longe
        de todos. Isso e o fim da posse, ao contrario da bola interpolada."""
        percepcao = _percepcao(
            _quadro(0, _bola(100, 100), _jogador(100, 100, 2, team="B")),
            _quadro(1, _bola(2000, 900), _jogador(100, 100, 2)),
            _quadro(2, _bola(100, 100), _jogador(100, 100, 2)),
        )

        toque = previous_touch(percepcao, 2)

        assert telemetry(percepcao, 2).episodios == 2
        assert toque is not None and toque.frame_index == 0 and toque.team == "B"

    def test_lacuna_longa_sem_bola_medida_encerra_o_episodio(self) -> None:
        """Atravessar buraco curto e certo, atravessar buraco longo seria afirmar posse continua
        sobre um trecho em que nem o rastreador da bola afirma continuidade: ele desiste em cinco
        amostras."""
        quadros = [_quadro(0, _bola(100, 100), _jogador(100, 100, 2))]
        quadros += [
            _quadro(i, _bola(100, 100, "interpolated"), _jogador(100, 100, 2)) for i in range(1, 12)
        ]
        quadros.append(_quadro(12, _bola(100, 100), _jogador(100, 100, 2)))
        percepcao = _percepcao(*quadros)

        assert telemetry(percepcao, 12).episodios == 2
        toque = previous_touch(percepcao, 12)
        assert toque is not None and toque.frame_index == 0

    def test_quadro_ambiguo_encerra_o_episodio_em_vez_de_chutar(self) -> None:
        """Um episodio que atravessa uma disputa nao e mais o toque de um jogador so."""
        percepcao = _percepcao(
            _quadro(0, _bola(100, 100), _jogador(100, 100, 2)),
            _quadro(1, _bola(100, 100), _jogador(60, 100, 2), _jogador(140, 100, 3)),
            _quadro(2, _bola(100, 100), _jogador(100, 100, 2)),
        )

        toque = previous_touch(percepcao, 2)

        assert toque is not None and toque.frame_index == 0

    def test_quadro_fora_de_ordem_nao_confunde_a_busca(self) -> None:
        """A percepcao chega ordenada, mas depender disso e uma armadilha silenciosa."""
        embaralhada = _percepcao(
            _quadro(2, _bola(100, 100), _jogador(100, 100, 7)),
            _quadro(0, _bola(100, 100), _jogador(100, 100, 2)),
            _quadro(1, _bola(100, 100)),
        )

        toque = previous_touch(embaralhada, 2)

        assert toque is not None and toque.player_track_id == 2 and toque.frame_index == 0

    def test_sem_episodio_do_contato_nao_ha_toque_anterior(self) -> None:
        percepcao = _percepcao(
            _quadro(0, _bola(100, 100), _jogador(100, 100, 2)),
            _quadro(1, _bola(100, 100), _jogador(100, 100, 2)),
            _quadro(600, _bola(5000, 100), _jogador(100, 100, 2)),
        )

        assert previous_touch(percepcao, 600) is None

    def test_jogador_sem_track_no_meio_encerra_o_episodio(self) -> None:
        percepcao = _percepcao(
            _quadro(0, _bola(100, 100), _jogador(100, 100, 1)),
            _quadro(1, _bola(100, 100), _jogador(100, 100, None)),
            _quadro(2, _bola(100, 100), _jogador(100, 100, 7)),
        )

        toque = previous_touch(percepcao, 2)

        assert toque is not None and toque.frame_index == 0 and toque.player_track_id == 1


class TestTelemetry:
    def test_separa_os_motivos_de_ausencia(self) -> None:
        """Sem isto, ausencia de bola, ausencia de jogador perto e empate chegam iguais."""
        percepcao = _percepcao(
            _quadro(0, None, _jogador(500, 100, 1)),
            _quadro(2, _bola(500, 100, "interpolated"), _jogador(500, 100, 1)),
            _quadro(4, _bola(500, 100), _jogador(2000, 900, 1)),
            _quadro(6, _bola(300, 100), _jogador(260, 100, 2), _jogador(340, 100, 3)),
            _quadro(8, _bola(100, 100), _jogador(100, 100, 7)),
        )

        contagem = telemetry(percepcao, 8)

        assert contagem.frames_com_bola == 4
        assert contagem.frames_com_bola_medida == 3
        assert contagem.frames_com_jogador_perto == 2
        assert contagem.quadros_ambiguos == 1

    def test_empate_no_contato_e_ambiguo_e_nao_ausente(self) -> None:
        percepcao = _percepcao(
            _quadro(10, _bola(100, 100), _jogador(60, 100, 3), _jogador(140, 100, 9))
        )

        assert telemetry(percepcao, 10).motivo_do_contato == "ambiguous"

    def test_ninguem_perto_no_contato_e_ausente(self) -> None:
        percepcao = _percepcao(_quadro(10, _bola(100, 100), _jogador(2000, 900, 7)))

        assert telemetry(percepcao, 10).motivo_do_contato == "absent"

    def test_sem_bola_medida_nenhuma_o_contato_e_ausente(self) -> None:
        percepcao = _percepcao(_quadro(10, None, _jogador(100, 100, 7)))

        assert telemetry(percepcao, 10).motivo_do_contato == "absent"

    def test_disputa_antes_do_contato_e_toque_anterior_ambiguo(self) -> None:
        """Bola disputada antes do chute e clipe sem bola antes do chute nao podem chegar iguais."""
        percepcao = _percepcao(
            _quadro(0, _bola(100, 100), _jogador(60, 100, 3), _jogador(140, 100, 9)),
            _quadro(5, _bola(100, 100), _jogador(100, 100, 7)),
            _quadro(6, _bola(100, 100), _jogador(100, 100, 7)),
        )

        contagem = telemetry(percepcao, 6)

        assert previous_touch(percepcao, 6) is None
        assert contagem.motivo_do_toque_anterior == "ambiguous"

    def test_toque_anterior_encontrado_nao_tem_motivo(self) -> None:
        assert telemetry(_passe_denso(), 10).motivo_do_toque_anterior is None

    def test_conta_o_clipe_inteiro_e_nao_so_o_que_vem_antes_do_contato(self) -> None:
        """A telemetria descreve a mesma janela que os episodios percorrem, que e o clipe todo.

        Contar so o antes esconderia justamente o suprimento que resolve o episodio do contato,
        que pode terminar depois do quadro escolhido.
        """
        percepcao = _percepcao(
            _quadro(0, _bola(100, 100), _jogador(100, 100, 1)),
            _quadro(1, _bola(100, 100), _jogador(100, 100, 1)),
        )

        assert telemetry(percepcao, 0).frames_com_bola == 2


class TestUmaAmostraInterpoladaNaoApagaOPasse:
    """O achado que a revisao adversarial reproduziu depois da reescrita.

    Quadros consecutivos na lista nao sao quadros consecutivos no clipe, e tratar quadro sem bola
    medida como fronteira de episodio devolvia o defeito original numa forma nova: uma unica amostra
    interpolada no meio da recepcao partia o episodio do finalizador em dois, e o toque anterior
    passava a ser ele mesmo meio segundo antes. Passe virava jogada individual, sem excecao e sem
    motivo declarado, porque um episodio anterior existia de verdade.
    """

    def _cena(self, interpolado: int | None) -> Perception:
        rota = {0: 100.0, 4: 100.0, 8: 100.0, 12: 400.0, 16: 700.0}
        rota.update({20: 1000.0, 24: 1000.0, 28: 1000.0, 32: 1000.0})
        return _percepcao(
            *(
                _quadro(
                    i,
                    _bola(x, 100, "interpolated" if i == interpolado else "detected"),
                    _jogador(100, 100, 2, team="A"),
                    _jogador(1000, 100, 7, team="B"),
                )
                for i, x in rota.items()
            )
        )

    def test_a_cena_limpa_da_o_passador(self) -> None:
        toque = previous_touch(self._cena(None), 34)

        assert toque is not None and toque.player_track_id == 2

    def test_a_cena_com_uma_interpolada_da_o_mesmo_passador(self) -> None:
        limpa = previous_touch(self._cena(None), 34)
        marcada = previous_touch(self._cena(28), 34)

        assert limpa is not None and marcada is not None
        assert marcada.player_track_id == limpa.player_track_id == 2
        assert telemetry(self._cena(28), 34).episodios == 2


class TestConjuntosMisturados:
    """`assemble` monta os quadros como uniao dos indices de todas as especies.

    Uma configuracao que ponha `pose` num conjunto mais fino que o da bola, como a `oracle-janela`
    faz, produz tres quadros sem bola entre cada dois com bola. Medir a cadencia na uniao dava um
    onde a cadencia real da bola e quatro, e os episodios viravam um por quadro.
    """

    def _cena(self) -> Perception:
        quadros: list[FramePerception] = []
        for i in range(44):
            tem_bola = i % 4 == 0
            x = 100.0 if i <= 8 else (1000.0 if i >= 20 else 500.0)
            jogadores = (_jogador(100, 100, 2), _jogador(1000, 100, 5)) if tem_bola else ()
            quadros.append(_quadro(i, _bola(x, 100) if tem_bola else None, *jogadores))
        return _percepcao(*quadros)

    def test_o_passo_vem_da_cadencia_da_bola_e_nao_da_uniao(self) -> None:
        cena = self._cena()

        assert _passo(cena) == 4.0

    def test_os_episodios_nao_viram_um_por_quadro(self) -> None:
        assert telemetry(self._cena(), 42).episodios == 2

    def test_o_toque_anterior_continua_sendo_o_passador(self) -> None:
        toque = previous_touch(self._cena(), 42)

        assert toque is not None and toque.player_track_id == 2


class TestDirecaoDoContato:
    """O contato de um chute fica no fim da posse do finalizador, nao no comeco.

    A primeira versao escolhia o episodio mais proximo em modulo, sem sinal, entao o zagueiro que
    apareceu na frente 0,13 s depois do chute era um finalizador admissivel, e a jogada saia como
    passe.
    """

    def test_episodio_que_comeca_depois_do_chute_nao_responde_por_ele(self) -> None:
        percepcao = _percepcao(
            *(_quadro(i, _bola(100, 100), _jogador(100, 100, 3)) for i in range(50, 61, 2)),
            *(_quadro(i, _bola(2000, 100), _jogador(2000, 100, 9)) for i in range(108, 117, 2)),
        )

        assert shooter_at_contact(percepcao, 100) is None

    def test_terminar_antes_ganha_de_comecar_depois_na_mesma_distancia(self) -> None:
        """Dois episodios a dois quadros do contato, um de cada lado, e nenhum cobrindo.

        O quadro 8 tem bola medida longe de todos, entao ele encerra o primeiro episodio e nao
        abre o segundo. Vence o que terminou antes, porque o contato de um chute fica no fim da
        posse de quem chutou.
        """
        percepcao = _percepcao(
            _quadro(6, _bola(100, 100), _jogador(100, 100, 3)),
            _quadro(7, _bola(100, 100), _jogador(100, 100, 3)),
            _quadro(8, _bola(2000, 100), _jogador(100, 100, 3)),
            _quadro(11, _bola(4000, 100), _jogador(4000, 100, 9)),
            _quadro(12, _bola(4000, 100), _jogador(4000, 100, 9)),
        )

        assert telemetry(percepcao, 9).episodios == 2
        evidencia = shooter_at_contact(percepcao, 9)

        assert evidencia is not None and evidencia.player_track_id == 3


class TestTresEpisodios:
    """Com tres episodios, "o ultimo que termina antes" deixa de coincidir com "o primeiro".

    A suite anterior nunca tinha tres, entao trocar o maximo pelo minimo na busca do toque anterior
    passava, apesar de essa escolha ser a propria definicao do que o modulo devolve.
    """

    def _cena(self) -> Perception:
        return _percepcao(
            _quadro(0, _bola(100, 100), _jogador(100, 100, 2), _jogador(1000, 100, 7)),
            _quadro(1, _bola(100, 100), _jogador(100, 100, 2), _jogador(1000, 100, 7)),
            _quadro(2, _bola(2000, 900), _jogador(100, 100, 2), _jogador(1000, 100, 7)),
            _quadro(3, _bola(100, 100), _jogador(100, 100, 4), _jogador(1000, 100, 7)),
            _quadro(4, _bola(100, 100), _jogador(100, 100, 4), _jogador(1000, 100, 7)),
            _quadro(5, _bola(2000, 900), _jogador(100, 100, 4), _jogador(1000, 100, 7)),
            _quadro(6, _bola(1000, 100), _jogador(100, 100, 4), _jogador(1000, 100, 7)),
            _quadro(7, _bola(1000, 100), _jogador(100, 100, 4), _jogador(1000, 100, 7)),
        )

    def test_o_toque_anterior_e_o_ultimo_que_termina_antes(self) -> None:
        toque = previous_touch(self._cena(), 7)

        assert toque is not None
        assert toque.player_track_id == 4, "pegou o primeiro episodio em vez do ultimo"
        assert toque.frame_index == 4

    def test_a_cena_tem_os_tres_episodios(self) -> None:
        assert telemetry(self._cena(), 7).episodios == 3


class TestFronteiraDoEpisodio:
    def test_dois_donos_em_quadros_vizinhos_dao_dois_episodios(self) -> None:
        """Nenhum teste anterior punha donos diferentes em quadros consecutivos: em toda cena havia
        um voo ou um quadro sem trilha separando. O ramo que separa passe curto de posse continuada
        na area, que e onde `type_of_play` mais precisa da distincao, ficava inteiro sem cobertura.
        """
        percepcao = _percepcao(
            _quadro(0, _bola(100, 100), _jogador(100, 100, 2), _jogador(180, 100, 7)),
            _quadro(1, _bola(100, 100), _jogador(100, 100, 2), _jogador(180, 100, 7)),
            _quadro(2, _bola(180, 100), _jogador(100, 100, 2), _jogador(180, 100, 7)),
            _quadro(3, _bola(180, 100), _jogador(100, 100, 2), _jogador(180, 100, 7)),
        )

        assert telemetry(percepcao, 3).episodios == 2
        toque = previous_touch(percepcao, 3)
        assert toque is not None and toque.player_track_id == 2

    def test_a_percepcao_desordenada_e_ordenada_antes_de_agrupar(self) -> None:
        """O teste anterior de ordem passava com a ordenacao removida, porque a cena embaralhada
        dava a mesma resposta de qualquer jeito. Sem ordenar, `inicio` pode passar de `fim` e o
        episodio deixa de cobrir quadro nenhum."""
        quadros = [
            _quadro(0, _bola(100, 100), _jogador(100, 100, 2)),
            _quadro(1, _bola(100, 100), _jogador(100, 100, 2)),
            _quadro(2, _bola(2000, 900), _jogador(100, 100, 2)),
            _quadro(3, _bola(1000, 100), _jogador(1000, 100, 7)),
            _quadro(4, _bola(1000, 100), _jogador(1000, 100, 7)),
        ]
        na_ordem = _percepcao(*quadros)
        embaralhada = _percepcao(*[quadros[i] for i in (3, 0, 4, 2, 1)])

        a = previous_touch(na_ordem, 4)
        b = previous_touch(embaralhada, 4)

        assert a is not None and b is not None
        assert (a.player_track_id, a.frame_index) == (b.player_track_id, b.frame_index) == (2, 1)
        assert shooter_at_contact(embaralhada, 4) is not None


class TestEscalaMedianaEMaxima:
    def test_a_escala_e_a_mediana_e_nao_a_maior_altura(self) -> None:
        """Trocar a mediana pelo maximo sobrevivia a suite inteira. Aqui um jogador de caixa tres
        vezes mais alta esta em cena e longe da bola: pela mediana o candidato fica fora do raio e
        o modulo se abstem, pelo maximo ele entraria e responderia."""
        alto = PlayerDetection(Box(1980, -50, 2020, 250), 0.8, track_id=8)
        percepcao = _percepcao(
            _quadro(10, _bola(100, 100), _jogador(270, 100, 3), _jogador(1500, 100, 5), alto)
        )

        assert shooter_at_contact(percepcao, 10) is None


class TestQuadroRepresentativo:
    def test_o_episodio_se_declara_pelo_quadro_mais_perto_e_nao_pelo_mais_confiante(self) -> None:
        """A procedencia do episodio sai do quadro em que a bola estava mais perto da caixa, e nao
        do quadro em que o detector estava mais confiante: confianca de deteccao nao mede posse."""
        perto_e_incerto = PlayerDetection(Box(80, 50, 120, 150), 0.4, track_id=7)
        longe_e_certo = PlayerDetection(Box(130, 50, 170, 150), 0.99, track_id=7)
        percepcao = _percepcao(
            _quadro(0, _bola(100, 100), perto_e_incerto),
            _quadro(1, _bola(100, 100), longe_e_certo),
        )

        evidencia = shooter_at_contact(percepcao, 0)

        assert evidencia is not None and evidencia.source == "detected"


class TestMotivosComTolerancia:
    def test_disputa_longe_do_contato_nao_explica_o_contato(self) -> None:
        """Uma disputa a dez segundos do contato explicava o contato como `ambiguous`, e a spec 04
        separa problema de cobertura de problema de rastreio. Motivo errado manda a taxonomia de
        erros procurar no lugar errado."""
        percepcao = _percepcao(
            _quadro(0, _bola(100, 100), _jogador(60, 100, 3), _jogador(140, 100, 9)),
            _quadro(600, None, _jogador(100, 100, 7)),
        )

        assert telemetry(percepcao, 600).motivo_do_contato == "absent"

    def test_o_quadro_que_explica_o_contato_desempata_pelo_indice(self) -> None:
        percepcao = _percepcao(
            _quadro(12, _bola(2000, 900), _jogador(100, 100, 7)),
            _quadro(8, _bola(100, 100), _jogador(60, 100, 3), _jogador(140, 100, 9)),
        )

        assert telemetry(percepcao, 10).motivo_do_contato == "ambiguous"

    def test_disputa_depois_do_chute_nao_explica_o_toque_anterior(self) -> None:
        """O motivo cita disputa que aconteceu antes do chute. Uma disputa posterior ao contato
        nao explica a ausencia de um toque que veio antes dele."""
        percepcao = _percepcao(
            _quadro(6, _bola(100, 100), _jogador(2000, 900, 1)),
            _quadro(7, _bola(100, 100), _jogador(60, 100, 3), _jogador(140, 100, 9)),
            _quadro(8, _bola(100, 100), _jogador(100, 100, 7)),
            _quadro(9, _bola(100, 100), _jogador(100, 100, 7)),
        )

        assert telemetry(percepcao, 6).motivo_do_toque_anterior == "absent"


class TestMargemNaFronteira:
    def test_diferenca_igual_a_margem_nao_e_empate(self) -> None:
        """Exercitado no ajudante e nao pela geometria de proposito: a fronteira exata tem medida
        nula em pixel, entao nenhuma cena de caixa e bola cai sobre ela. A convencao ainda precisa
        ficar presa, senao trocar `<` por `<=` passa sem que nada avise.
        """
        um = _jogador(100, 100, 3)
        outro = _jogador(100, 100, 9)

        assert _empatado([(um, 0.0), (outro, MARGEM_DE_EMPATE)]) is False
        assert _empatado([(um, 0.0), (outro, MARGEM_DE_EMPATE / 2)]) is True
