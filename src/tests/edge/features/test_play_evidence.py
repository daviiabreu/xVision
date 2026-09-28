"""A evidencia da jogada, um teste por defeito que nao levanta excecao nenhuma.

Todo defeito coberto aqui devolve numero plausivel. Velocidade dividida pela posicao na lista sai
com a mesma cara de velocidade dividida pelo tempo, imobilidade contada sobre posicao interpolada
sai com a mesma cara de bola parada de verdade, e `False` em vez de `None` sai com a mesma cara de
uma comparacao que alguem fez. Nenhum deles quebra o schema e nenhum aparece no relatorio.

Por isso quase todo teste daqui compara duas cenas que precisam dar a mesma leitura, ou uma cena
que precisa dar leitura diferente da que o defeito daria. Afirmar so que o campo existe deixaria
passar todos eles.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields

import pytest

from contracts import BallDetection, Box, FramePerception, Perception, PlayerDetection
from edge.features.base import ShotOutput
from edge.features.play_evidence import (
    PlayEvidence,
    PlayTelemetry,
    ShotProvenance,
    contact_sample,
    contact_time_s,
    measure,
    window,
)

FPS = 59.94
FPS_METADE = 29.97
"""As duas cadencias do mesmo clipe: o nativo e a metade dele, que e onde a divisao por indice
deixa de coincidir com a divisao por tempo."""

FPS_FINO = 32.0
FPS_DEZ = 10.0
"""Potencia de dois e potencia de dez, para os empates e as bordas da janela caberem exatos em
ponto flutuante. Com 59,94 a diferenca de dois instantes simetricos nao sai igual, e o teste de
empate mediria o arredondamento em vez da regra."""

PASSO_ALTO = 6
PASSO_BAIXO = 3

POSICOES = (0.0, 300.0, 305.0, 310.0, 315.0)
"""Um trecho rapido seguido de tres lentos: 300 px, depois 5 px por amostra.

A ordem importa. Com o trecho rapido no comeco, a velocidade de chegada tem de ler os dois ultimos
trechos e nao o maximo da janela, e a contagem de imobilidade tem de parar no trecho rapido em vez
de somar a janela inteira.
"""

CAMPOS_DA_EVIDENCIA = (
    "shot_source",
    "shot_confidence",
    "shot_ball_support",
    "stationary_s",
    "path_player_heights",
    "arrival_speed_player_heights_s",
    "observed_fraction",
    "previous_touch_is_shooter",
    "previous_touch_age_s",
    "previous_touch_source",
    "camera_cut",
    "previous_touch_relation",
)

CAMPOS_DA_TELEMETRIA = (
    "frames_in_window",
    "ball_samples",
    "ball_detected",
    "median_player_height_px",
    "contact_sample_offset_s",
    "touch",
)


@dataclass(frozen=True)
class _Shot:
    frame_index: int
    shooter_track_id: int | None = None
    """Sempre ausente, como o modulo de chute produz. A evidencia nunca le este campo, e o teste o
    deixa em `None` para que ler passe a falhar em vez de funcionar por acidente."""


PROVENANCE = ShotProvenance(source="ball-break", confidence=0.8, ball_support="detected")


def _ball(
    x: float, y: float = 100.0, *, source: str | None = "detected", radius: float = 5.0
) -> BallDetection:
    """O contrato ja exige `confidence=0.0` e `gap_frames` positivo em posicao interpolada."""
    interpolated = source == "interpolated"
    return BallDetection(
        Box(x - radius, y - radius, x + radius, y + radius),
        0.0 if interpolated else 0.9,
        source=source,
        gap_frames=1 if interpolated else None,
    )


def _player(
    x: float, y: float, track: int | None, *, width: float = 40.0, height: float = 100.0
) -> PlayerDetection:
    """Caixa de 40 por 100 por padrao, que e a ordem de grandeza de um jogador em quadro de 1080.

    Com altura cem, distancia normalizada e distancia em pixel dividida por cem, e velocidade em
    altura por segundo e velocidade em pixel por segundo dividida por cem.
    """
    return PlayerDetection(
        Box(x - width / 2, y - height / 2, x + width / 2, y + height / 2), 0.8, track_id=track
    )


def _frame(
    index: int, fps: float, ball: BallDetection | None, *players: PlayerDetection
) -> FramePerception:
    return FramePerception(frame_index=index, timestamp_s=index / fps, players=players, ball=ball)


def _perception(*frames: FramePerception) -> Perception:
    return Perception(clip_id="c", frames=frames)


def _measure(
    perception: Perception, contact: int, provenance: ShotProvenance = PROVENANCE
) -> tuple[PlayEvidence, PlayTelemetry]:
    shot: ShotOutput = _Shot(contact)
    return measure(perception, shot, provenance)


def _far_players(
    scale: float = 1.0, heights: Sequence[float] = (100.0, 100.0)
) -> tuple[PlayerDetection, ...]:
    """Jogadores rastreados longe da bola, um por altura pedida.

    Longe de proposito: estes testes medem trajetoria, e um jogador dentro do raio da bola abriria
    episodio de posse e traria os campos de toque para dentro da comparacao.
    """
    return tuple(
        _player(
            (600.0 + 200.0 * i) * scale,
            100.0 * scale,
            2 + i,
            width=40.0 * scale,
            height=height * scale,
        )
        for i, height in enumerate(heights)
    )


def _ball_scene(
    positions: Sequence[float] = POSICOES,
    *,
    fps: float = FPS,
    step: int = PASSO_ALTO,
    scale: float = 1.0,
    heights: Sequence[float] = (100.0, 100.0),
) -> Perception:
    """Uma deteccao por amostra ao longo de `positions`, com o contato na ultima delas."""
    players = _far_players(scale, heights)
    return _perception(
        *(
            _frame(i * step, fps, _ball(x * scale, 100.0 * scale, radius=5.0 * scale), *players)
            for i, x in enumerate(positions)
        )
    )


def _dribble_scene(*, tail_source: str | None = "detected") -> Perception:
    """Conducao do track 2: bola no pe, sai do raio, volta ao pe, e o chute sai dali.

    A bola sair e voltar ao raio e o que faz uma conducao render dois episodios de posse do mesmo
    dono, que e a unica forma de existir toque anterior do proprio finalizador.

    `tail_source` troca a procedencia dos quadros 61 a 64, que sao os unicos dentro da tolerancia do
    contato. Com eles interpolados nao existe amostra medida para sustentar finalizador, e o quadro
    60 fica a 0,125 s do instante, fora dos 0,1 s de tolerancia.
    """
    route = {56: 100.0, 57: 100.0, 58: 100.0, 59: 400.0, 60: 400.0}
    route |= {61: 100.0, 62: 100.0, 63: 100.0, 64: 100.0}
    return _perception(
        *(
            _frame(
                index,
                FPS_FINO,
                _ball(x, source="detected" if index <= 60 else tail_source),
                _player(100.0, 100.0, 2),
                _player(900.0, 100.0, 7),
            )
            for index, x in route.items()
        )
    )


def test_velocidade_usa_timestamp_e_nao_indice_nem_posicao() -> None:
    """A mesma trajetoria a 59,94 e a 29,97 fps tem de dar a mesma leitura.

    E o defeito que `type_of_play` carrega hoje: `_velocidades` divide por diferenca de indice, e
    `_imoveis_no_fim` soma diferenca de indice como se fosse tempo. Com os indices pela metade e os
    instantes iguais, dividir por indice dobra a velocidade e somar indice corta a imobilidade pela
    metade, e nenhuma das duas coisas aconteceu no video.
    """
    fina = _ball_scene(fps=FPS, step=PASSO_ALTO)
    grossa = _ball_scene(fps=FPS_METADE, step=PASSO_BAIXO)
    assert tuple(q.timestamp_s for q in fina.frames) == tuple(
        q.timestamp_s for q in grossa.frames
    ), "as duas cenas precisam ter os mesmos instantes para a comparacao valer"

    evidencia_fina, telemetria_fina = _measure(fina, 4 * PASSO_ALTO)
    evidencia_grossa, telemetria_grossa = _measure(grossa, 4 * PASSO_BAIXO)

    assert telemetria_fina.frames_in_window == 5
    assert telemetria_grossa.frames_in_window == 5
    assert evidencia_fina.stationary_s == pytest.approx(3 * PASSO_ALTO / FPS)
    assert evidencia_fina.arrival_speed_player_heights_s == pytest.approx(0.4995)
    assert evidencia_fina.path_player_heights == pytest.approx(3.15)
    assert evidencia_fina == evidencia_grossa


def test_distancia_e_normalizada_pela_altura_mediana() -> None:
    """A leitura nao pode mudar com o zoom da camera, e a escala e a mediana e nao uma caixa.

    Tres comparacoes. Jogador com o dobro da altura na mesma trajetoria significa camera duas vezes
    mais longe, e a bola andou metade das alturas. Cena com toda coordenada dobrada e a mesma
    jogada filmada de mais perto, e a evidencia sai identica. E com alturas 60, 100 e 200 na janela,
    a escala e 100: normalizar pela caixa de um candidato daria 60, 200 ou a media 120.
    """
    normal, telemetria_normal = _measure(_ball_scene(), 4 * PASSO_ALTO)
    altos, telemetria_altos = _measure(_ball_scene(heights=(200.0, 200.0)), 4 * PASSO_ALTO)
    dobrada, telemetria_dobrada = _measure(_ball_scene(scale=2.0), 4 * PASSO_ALTO)
    variadas, telemetria_variadas = _measure(
        _ball_scene(heights=(60.0, 100.0, 200.0)), 4 * PASSO_ALTO
    )
    assert normal.arrival_speed_player_heights_s is not None
    assert normal.path_player_heights is not None

    assert telemetria_normal.median_player_height_px == 100.0
    assert telemetria_altos.median_player_height_px == 200.0
    assert telemetria_dobrada.median_player_height_px == 200.0
    assert telemetria_variadas.median_player_height_px == 100.0
    assert altos.arrival_speed_player_heights_s == pytest.approx(
        normal.arrival_speed_player_heights_s / 2
    )
    assert altos.path_player_heights == pytest.approx(normal.path_player_heights / 2)
    assert dobrada == normal
    assert variadas.path_player_heights == pytest.approx(normal.path_player_heights)
    assert variadas.arrival_speed_player_heights_s == pytest.approx(
        normal.arrival_speed_player_heights_s
    )


def test_ponto_interpolado_nao_prova_imobilidade() -> None:
    """Bola detectada em movimento, e duas posicoes interpoladas paradas no mesmo ponto depois.

    A interpolacao e uma reta entre as duas deteccoes que fecham a lacuna, entao duas amostras
    interpoladas no mesmo lugar sao consequencia da reta e nao bola quieta. Contando os
    interpolados, este clipe sai com 0,2 s de imobilidade antes do chute, que e a leitura de bola
    parada sobre um clipe em que a bola simplesmente nao foi detectada.
    """
    players = _far_players()
    percepcao = _perception(
        _frame(0, FPS, _ball(0.0), *players),
        _frame(6, FPS, _ball(100.0), *players),
        _frame(12, FPS, _ball(100.0, source="interpolated"), *players),
        _frame(18, FPS, _ball(100.0, source="interpolated"), *players),
    )

    evidencia, telemetria = _measure(percepcao, 18)

    assert evidencia.stationary_s == 0.0
    assert evidencia.path_player_heights == pytest.approx(1.0)
    assert telemetria.ball_samples == 4
    assert telemetria.ball_detected == 2
    assert evidencia.observed_fraction == 0.5


def test_lacuna_longa_entre_deteccoes_encerra_o_trecho_imovel() -> None:
    """Duas deteccoes no mesmo ponto separadas por 1,2 s nao somam 1,2 s de bola parada.

    O rastreador fecha lacuna de no maximo cinco amostras a 15 fps, ou seja 0,33 s. Um intervalo de
    1,2 s entre duas deteccoes nao diz que a bola ficou ali: diz que ela foi perdida de vista, e um
    passe inteiro cabe dentro. Somar isso transformaria falha de deteccao em bola parada.

    O trecho de 0,1 s antes da lacuna tambem esta imovel de proposito: a contagem precisa parar na
    lacuna, e nao pular por cima dela para somar o que veio antes.
    """
    players = _far_players()
    percepcao = _perception(
        _frame(0, FPS_DEZ, _ball(0.0), *players),
        _frame(1, FPS_DEZ, _ball(0.0), *players),
        _frame(13, FPS_DEZ, _ball(0.0), *players),
    )

    evidencia, telemetria = _measure(percepcao, 13)

    assert telemetria.ball_detected == 3
    assert evidencia.stationary_s == 0.0
    assert evidencia.path_player_heights == 0.0


def test_duas_amostras_no_mesmo_instante_nao_viram_velocidade_infinita() -> None:
    """Dois quadros com o mesmo `timestamp_s` nao tem tempo decorrido entre si.

    Acontece quando quem monta a percepcao repete o instante, e a divisao por zero levantaria
    dentro de `measure`, o que custaria as outras features do mesmo clipe. A recusa sai `None` e
    nao um numero grande: velocidade infinita se le como bola chegando rapido, que e o ramo de
    jogada aberta, e nada foi medido.

    E o unico caso em que duas deteccoes na janela nao produzem trajetoria, e por isso ele esta
    aqui: a condicao de recusa nao e so a contagem de deteccoes.
    """
    players = _far_players()
    percepcao = _perception(
        _frame(0, FPS, _ball(0.0), *players),
        FramePerception(frame_index=6, timestamp_s=0.0, players=players, ball=_ball(100.0)),
    )

    evidencia, telemetria = _measure(percepcao, 6)

    assert telemetria.ball_detected == 2
    assert evidencia.observed_fraction == 1.0
    assert evidencia.stationary_s is None
    assert evidencia.path_player_heights is None
    assert evidencia.arrival_speed_player_heights_s is None


def test_menos_de_duas_detectadas_deixa_alturas_em_none() -> None:
    """Uma deteccao so nao define trajetoria, e a recusa nao pode sair como zero.

    Zero em `stationary_s` e `path_player_heights` se le como bola parada e sem deslocamento, que e
    exatamente o ramo de bola parada da regra. O clipe sem deteccao sairia identico ao clipe com a
    bola quieta no chao.

    Tres formas de nao ser deteccao aparecem aqui: posicao interpolada, posicao sem procedencia
    declarada e quadro sem bola. `source = None` e a mais silenciosa das tres, e conta como nao
    detectada pela mesma regra de `touch._medida`: ler ausencia de procedencia como medida
    transformaria um detector que nao declara em um que declarou, e este clipe passaria a ter duas
    deteccoes e trajetoria medida.
    """
    players = _far_players()
    percepcao = _perception(
        _frame(0, FPS, _ball(0.0, source="interpolated"), *players),
        _frame(6, FPS, _ball(100.0), *players),
        _frame(12, FPS, _ball(200.0, source=None), *players),
        _frame(18, FPS, None, *players),
    )

    evidencia, telemetria = _measure(percepcao, 18)

    assert evidencia.stationary_s is None
    assert evidencia.path_player_heights is None
    assert evidencia.arrival_speed_player_heights_s is None
    assert evidencia.observed_fraction == 0.25
    assert telemetria.ball_detected == 1
    assert telemetria.median_player_height_px == 100.0


def test_sem_jogador_na_janela_nao_ha_escala() -> None:
    """Sem jogador nao existe altura para normalizar, e a fracao observada continua medida.

    As duas coisas juntas: a distancia em pixel esta la, mas nao ha escala que a torne comparavel
    entre clipes, e publicar o pixel cru daria numeros diferentes para a mesma jogada em zoom
    diferente. A cobertura da bola nao depende de jogador nenhum e continua respondendo.

    Caixa de altura zero nao conta como jogador. Ela aparece quando o detector devolve caixa
    degenerada, e a media dela com as outras baixaria a escala de todo mundo.
    """
    percepcao = _perception(
        _frame(0, FPS, _ball(0.0)),
        _frame(6, FPS, _ball(5.0)),
        _frame(12, FPS, _ball(10.0)),
    )
    degenerada = _perception(
        *(
            _frame(i * PASSO_ALTO, FPS, _ball(float(i * 5)), _player(600.0, 100.0, 2, height=0.0))
            for i in range(3)
        )
    )

    evidencia, telemetria = _measure(percepcao, 12)
    evidencia_degenerada, telemetria_degenerada = _measure(degenerada, 12)

    assert telemetria.median_player_height_px is None
    assert telemetria_degenerada.median_player_height_px is None
    assert evidencia.stationary_s is None
    assert evidencia.path_player_heights is None
    assert evidencia.arrival_speed_player_heights_s is None
    assert evidencia_degenerada.path_player_heights is None
    assert evidencia.observed_fraction == 1.0
    assert telemetria.ball_detected == 3
    assert evidencia.camera_cut is False


def test_fracao_observada_conta_deteccao_sobre_quadros_da_janela() -> None:
    """Vinte e dois quadros, onze detectados, seis interpolados e cinco sem bola dao 0,5.

    Interpolado e sem bola ficam no denominador. O que este numero mede e quanto da janela foi
    observado de verdade, e tirar do denominador o que nao foi observado faria toda janela parecer
    inteiramente observada: um clipe com uma deteccao e vinte e uma lacunas sairia com 1,0.
    """
    players = _far_players()
    quadros: list[FramePerception] = []
    for indice in range(22):
        if indice < 11:
            bola = _ball(float(indice * 10))
        elif indice < 17:
            bola = _ball(110.0, source="interpolated")
        else:
            bola = None
        quadros.append(_frame(indice, FPS, bola, *players))

    evidencia, telemetria = _measure(_perception(*quadros), 21)

    assert telemetria.frames_in_window == 22
    assert telemetria.ball_samples == 17
    assert telemetria.ball_detected == 11
    assert evidencia.observed_fraction == 0.5


def test_janela_em_segundos_inclui_o_contato_e_exclui_o_depois() -> None:
    """A janela e `[t_c - 1,5 s, t_c]`: a amostra em `t_c - 1,5` entra, a anterior nao, e nada
    depois do contato entra.

    Os cinco quadros depois do contato tem a bola detectada parada no mesmo ponto, que e o que
    acontece em todo chute que bate na rede, no goleiro ou no chao. Se eles entrassem, todo chute
    defendido sairia com meio segundo de bola parada e o ramo de bola parada dispararia.
    """
    players = (_player(1500.0, 100.0, 2), _player(1700.0, 100.0, 7))
    quadros = [
        _frame(84, FPS_DEZ, _ball(-300.0), *players),
        _frame(85, FPS_DEZ, _ball(0.0), *players),
        _frame(100, FPS_DEZ, _ball(600.0), *players),
    ]
    quadros += [_frame(indice, FPS_DEZ, _ball(600.0), *players) for indice in range(101, 106)]
    percepcao = _perception(*quadros)

    t_contato = contact_time_s(percepcao, 100)
    assert t_contato is not None and t_contato == 10.0
    janela = window(percepcao, t_contato)
    evidencia, telemetria = _measure(percepcao, 100)

    assert tuple(q.frame_index for q in janela) == (85, 100)
    assert telemetria.frames_in_window == 2
    assert evidencia.path_player_heights == pytest.approx(6.0)
    assert evidencia.stationary_s == 0.0


def test_contato_fora_das_amostras_recebe_tempo_por_interpolacao_do_relogio() -> None:
    """O contato refinado cai entre amostras em tres de cada quatro clipes, e o relogio e afim.

    `find_contact` devolve o quadro da intersecao das duas retas ajustadas, que nao e amostra. Com
    amostras de quatro em quatro, o contato 103 tem de sair em `103 / 59,94`, que e o que o runner
    escreveria se aquele quadro tivesse sido amostrado.

    Com uma amostra so o relogio nao tem inclinacao, e a resposta e `None` em vez de um instante
    obtido inventando fps.
    """
    players = _far_players()
    percepcao = _perception(
        *(_frame(indice, FPS, _ball(float(indice)), *players) for indice in (96, 100, 104, 108))
    )
    sozinha = _perception(_frame(100, FPS, _ball(0.0), *players))

    assert contact_time_s(percepcao, 103) == pytest.approx(103 / FPS)
    assert contact_time_s(percepcao, 104) == 104 / FPS
    assert contact_time_s(sozinha, 103) is None

    evidencia, telemetria = _measure(sozinha, 103)

    assert telemetria.frames_in_window == 0
    assert evidencia.observed_fraction == 0.0
    assert evidencia.stationary_s is None


def test_contato_entre_amostras_usa_a_detectada_mais_proxima_e_prefere_a_anterior() -> None:
    """A amostra do contato e a deteccao mais proxima do instante, e o empate escolhe a anterior.

    Proximidade decide primeiro: com deteccoes a 0,0625 s antes e 0,03125 s depois, ganha a de
    depois. No empate exato ganha a anterior, porque as duas descrevem a bola antes e depois de
    sair do pe e e a anterior que ainda a tem no pe.

    Fora da tolerancia a resposta e `None`. A mais proxima sem limite responde sempre, e a ultima
    deteccao de um clipe sairia como bola no pe do finalizador mesmo segundos antes do chute.
    """
    players = _far_players()

    def cena(*indices: int) -> Perception:
        """Os indices dados com bola detectada, e o quadro 64 do contato sempre interpolado."""
        quadros = [_frame(64, FPS_FINO, _ball(50.0, source="interpolated"), *players)]
        quadros += [_frame(indice, FPS_FINO, _ball(float(indice)), *players) for indice in indices]
        return _perception(*quadros)

    mais_proxima = contact_sample(cena(62, 65), 2.0)
    empate = contact_sample(cena(63, 65), 2.0)
    fora = contact_sample(cena(68), 2.0)
    _, telemetria = _measure(cena(63, 65), 64)

    assert mais_proxima is not None and mais_proxima.frame_index == 65
    assert empate is not None and empate.frame_index == 63
    assert fora is None
    assert telemetria.contact_sample_offset_s == -0.03125


def test_amostra_interpolada_no_contato_nao_sustenta_finalizador() -> None:
    """A mesma conducao, com os quadros do contato medidos e interpolados, da finalizador e `None`.

    Interpolada, a posicao da bola no contato e a reta que o rastreador desenhou entre as duas
    deteccoes que fecham a lacuna, e o contato e exatamente onde a trajetoria deixa de ser reta.
    Associar um jogador aquela posicao devolveria um finalizador que a imagem nao mostra, e ele
    sairia com a mesma cara de um finalizador medido.
    """
    medida, telemetria_medida = _measure(_dribble_scene(), 64)
    interpolada, telemetria_interpolada = _measure(_dribble_scene(tail_source="interpolated"), 64)

    assert telemetria_medida.contact_sample_offset_s == 0.0
    assert medida.previous_touch_is_shooter is True
    assert telemetria_interpolada.contact_sample_offset_s is None
    assert interpolada.previous_touch_is_shooter is None
    assert interpolada.previous_touch_age_s is None
    assert interpolada.previous_touch_source is None


def test_toque_anterior_do_mesmo_track_e_do_finalizador() -> None:
    """Conducao: a bola sai e volta ao pe do track 2, e o toque anterior e dele mesmo."""
    evidencia, telemetria = _measure(_dribble_scene(), 64)

    assert evidencia.previous_touch_is_shooter is True
    assert evidencia.previous_touch_source == "detected"
    assert telemetria.touch.episodios == 2


def test_toque_de_outro_track_nao_e() -> None:
    """Passe: o track 7 toca, a bola voa fora do raio dos dois, e o track 2 finaliza.

    `False` aqui e medida e nao ausencia, e e o unico ramo que separa passe de conducao. Um modulo
    que devolvesse `None` neste caso apagaria a distincao que a feature existe para fazer.
    """
    route = {56: 900.0, 57: 900.0, 58: 900.0, 59: 500.0, 60: 500.0}
    route |= {61: 100.0, 62: 100.0, 63: 100.0, 64: 100.0}
    percepcao = _perception(
        *(
            _frame(
                indice,
                FPS_FINO,
                _ball(x),
                _player(100.0, 100.0, 2),
                _player(900.0, 100.0, 7),
            )
            for indice, x in route.items()
        )
    )

    evidencia, _ = _measure(percepcao, 64)

    assert evidencia.previous_touch_is_shooter is False


def _passe(time_do_passador: str | None, papel_do_passador: str | None = None) -> Perception:
    """O track 7 passa para o track 2, do `team_a`, que finaliza no quadro 64."""
    route = {56: 900.0, 57: 900.0, 58: 900.0, 59: 500.0, 60: 500.0}
    route |= {61: 100.0, 62: 100.0, 63: 100.0, 64: 100.0}
    finalizador = PlayerDetection(Box(80, 50, 120, 150), 0.8, track_id=2, team="team_a")
    passador = PlayerDetection(
        Box(880, 50, 920, 150), 0.8, track_id=7, team=time_do_passador, role=papel_do_passador
    )
    return _perception(
        *(_frame(i, FPS_FINO, _ball(x), finalizador, passador) for i, x in route.items())
    )


@pytest.mark.parametrize(
    ("time_do_passador", "papel", "relacao"),
    [
        ("team_a", None, "companheiro"),
        ("team_b", None, "adversario"),
        (None, "goalkeeper", "goleiro"),
        (None, None, None),
    ],
)
def test_relacao_do_toque_anterior_com_o_finalizador(
    time_do_passador: str | None, papel: str | None, relacao: str | None
) -> None:
    evidencia, _ = _measure(_passe(time_do_passador, papel), 64)

    assert evidencia.previous_touch_relation == relacao


def test_conducao_e_toque_do_proprio_finalizador() -> None:
    evidencia, _ = _measure(_dribble_scene(), 64)

    assert evidencia.previous_touch_relation == "proprio"


def test_empate_no_contato_deixa_none_em_vez_de_escolher() -> None:
    """Bola exatamente entre dois jogadores em toda a janela: nao ha finalizador a declarar.

    Escolher o de menor `track_id` produziria resposta estavel e errada, e a amostra do contato
    existe, entao o `None` vem do empate e nao de falta de bola medida.
    """
    percepcao = _perception(
        *(
            _frame(
                indice,
                FPS_FINO,
                _ball(200.0),
                _player(100.0, 100.0, 2),
                _player(300.0, 100.0, 7),
            )
            for indice in range(56, 65)
        )
    )

    evidencia, telemetria = _measure(percepcao, 64)

    assert telemetria.contact_sample_offset_s == 0.0
    assert telemetria.touch.quadros_ambiguos > 0
    assert evidencia.previous_touch_is_shooter is None
    assert evidencia.previous_touch_age_s is None
    assert evidencia.previous_touch_source is None


def test_sem_finalizador_o_sinal_fica_none_e_nao_false() -> None:
    """Bola longe de todo jogador rastreado: nenhuma comparacao foi feita, e o campo diz isso.

    `False` afirmaria que o toque anterior foi de outro jogador, que e a leitura de passe. Sem
    finalizador identificado nao houve comparacao nenhuma, e a diferenca entre as duas coisas e a
    diferenca entre passe medido e passe suposto.
    """
    percepcao = _perception(
        *(
            _frame(
                indice,
                FPS_FINO,
                _ball(1500.0),
                _player(100.0, 100.0, 2),
                _player(900.0, 100.0, 7),
            )
            for indice in range(56, 65)
        )
    )

    evidencia, telemetria = _measure(percepcao, 64)

    assert telemetria.contact_sample_offset_s == 0.0
    assert evidencia.previous_touch_is_shooter is None, (
        "sem finalizador, False afirmaria que o toque anterior foi de outro jogador"
    )
    assert evidencia.previous_touch_age_s is None
    assert evidencia.previous_touch_source is None


def test_idade_do_toque_anterior_e_em_segundos_do_contato() -> None:
    """A idade do toque e `t_c` menos o instante do toque, em segundos e nao em quadros.

    O toque fecha no quadro 58 e o contato esta em 64, a 32 fps: 0,1875 s. Em quadros isso seriam
    seis, que significa 0,1 s a 59,94 fps e 0,4 s a 15 fps, e a regra que separa passe de conducao
    olha o tempo.
    """
    evidencia, _ = _measure(_dribble_scene(), 64)

    assert evidencia.previous_touch_age_s == pytest.approx(2.0 - 58 / FPS_FINO)
    assert evidencia.previous_touch_age_s == pytest.approx(0.1875)


def test_corte_de_camera_quando_nenhum_track_sobrevive() -> None:
    """Dois quadros vizinhos com `{1, 2}` e `{8, 9}`: o rastreador IoU guloso perdeu tudo.

    Perda total entre quadros consecutivos e o que um corte de replay produz, e ela invalida a
    leitura de trajetoria da janela inteira: as caixas de antes e de depois nao sao da mesma cena.
    """
    antes = (_player(1500.0, 100.0, 1), _player(1700.0, 100.0, 2))
    depois = (_player(1500.0, 100.0, 8), _player(1700.0, 100.0, 9))
    percepcao = _perception(
        *(
            _frame(indice, FPS_FINO, _ball(100.0), *(antes if indice <= 61 else depois))
            for indice in range(60, 65)
        )
    )

    evidencia, telemetria = _measure(percepcao, 64)

    assert telemetria.frames_in_window == 5
    assert evidencia.camera_cut is True


def test_troca_parcial_de_tracks_nao_e_corte() -> None:
    """Trajetoria que sobrevive, e lado com uma trajetoria so, nao sao corte.

    `{1, 2}` seguido de `{2, 9}` e jogador entrando e saindo do quadro, que acontece o tempo todo.
    E com um `track_id` so em cada lado, perder o unico e o caso comum de quem saiu do quadro:
    chamar isso de corte declararia corte em quase todo clipe.
    """

    def cena(antes: tuple[PlayerDetection, ...], depois: tuple[PlayerDetection, ...]) -> Perception:
        return _perception(
            *(
                _frame(indice, FPS_FINO, _ball(100.0), *(antes if indice <= 61 else depois))
                for indice in range(60, 65)
            )
        )

    parcial = cena(
        (_player(1500.0, 100.0, 1), _player(1700.0, 100.0, 2)),
        (_player(1500.0, 100.0, 2), _player(1700.0, 100.0, 9)),
    )
    solitaria = cena((_player(1500.0, 100.0, 1),), (_player(1500.0, 100.0, 9),))

    evidencia_parcial, _ = _measure(parcial, 64)
    evidencia_solitaria, _ = _measure(solitaria, 64)

    assert evidencia_parcial.camera_cut is False
    assert evidencia_solitaria.camera_cut is False


@pytest.mark.parametrize(
    ("declarado", "esperado"),
    [
        ("detected", "detected"),
        ("interpolated", "interpolated"),
        ("mixed", "mixed"),
        ("absent", "none"),
        ("unknown", "none"),
        (None, "none"),
        ("procedencia-nova", "none"),
    ],
)
def test_procedencia_copiada_e_apoio_fora_do_vocabulario_vira_none(
    declarado: str | None, esperado: str
) -> None:
    """A procedencia do chute e copiada, e o apoio fora dos quatro valores vira `none`.

    O vocabulario do modulo de chute tem cinco valores e este campo tem quatro: `absent`, `unknown`
    e valor novo dizem a mesma coisa aqui, que nao existe posicao medida sustentando o instante.
    Passar o quinto adiante deixaria um campo tipado como quatro carregar um valor que quem
    ramificasse nos quatro conhecidos trataria como o ramo `else` sem saber.

    Fonte e confianca ausentes viram `unknown` e `0,0`, e nao somem: um chute sem procedencia
    declarada e informacao sobre o chute, e nao ausencia de campo.
    """
    percepcao = _dribble_scene()

    declarada, _ = _measure(percepcao, 64, ShotProvenance("ball-break", 0.42, declarado))
    vazia, _ = _measure(percepcao, 64, ShotProvenance(None, None, declarado))

    assert declarada.shot_source == "ball-break"
    assert declarada.shot_confidence == 0.42
    assert declarada.shot_ball_support == esperado
    assert vazia.shot_source == "unknown"
    assert vazia.shot_confidence == 0.0
    assert vazia.shot_ball_support == esperado


def test_os_campos_sao_exatamente_os_da_spec() -> None:
    """Campo novo na evidencia sem teste proprio passaria a viajar ate o consumidor em silencio.

    A lista congelada aqui e o contrato combinado com quem liga o `extract`: acrescentar campo
    exige mexer nesta linha, e mexer nela aparece na revisao.
    """
    assert tuple(campo.name for campo in fields(PlayEvidence)) == CAMPOS_DA_EVIDENCIA
    assert tuple(campo.name for campo in fields(PlayTelemetry)) == CAMPOS_DA_TELEMETRIA
