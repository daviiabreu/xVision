"""Quem tocou a bola: o finalizador no contato e o toque anterior, sobre a percepcao ja montada.

O modulo `shot` nao pode responder isso, e o motivo e mais simples do que o historico sugere: ele
decide sobre a trajetoria da bola, e nao sobre caixas de jogador. Houve um tempo em que a
dependencia tambem fechava ciclo, quando `tracking` rodava numa janela ancorada no `shot`, mas a
uniformizacao removeu o ciclo. Medido no config de producao, `shot` dependendo de `tracking` ordena
limpo em `ball, field, identity, players, tracking, shot, pose`, entao quem repetir o argumento do
ciclo esta repetindo historia.

O que sobra e fragilidade, e ela basta: `tracking` ainda declara `window` como frameset no proprio
spec, e `window` continua ancorado no `shot`. Amarrar `shot` a `tracking` faria a validade do grafo
depender de qual frameset o TOML escolhe, que varia por config: `pipeline.oracle-janela.toml` ja poe
`pose` no `window`. O proprio `edge/modules/shot/kinematics.py` aponta o trabalho para ca.

Aqui o grafo ja rodou. A `Perception` montada tem bola, jogadores e `track_id` no mesmo quadro,
entao a associacao e aritmetica sobre dado pronto, sem inferencia nova e sem peso.

Um toque e um episodio, e nao um quadro. A primeira versao deste modulo devolvia o jogador mais
proximo no quadro anterior mais recente, e numa `Perception` densa isso colapsa: a 59,94 fps um
passe a 15 m/s anda 0,25 m por quadro, muito dentro do raio de toque, entao o quadro anterior ao
contato pertence ao proprio finalizador e `previous_touch_is_shooter` saia verdadeiro tambem para
passe. Estavel e errado e pior do que ausente, porque ninguem desconfia.

Agora os quadros sao agrupados em episodios de posse: sequencias de quadros consecutivos em que a
bola esta dentro do raio do mesmo `track_id`. O toque anterior e o ultimo episodio que termina antes
de comecar o episodio do contato. Passe da dois episodios com donos diferentes; conducao da varios
episodios do mesmo dono, porque a bola sai e volta ao raio entre os toques; corrida com a bola
colada da um episodio so, e ai nao existe toque anterior a declarar.

O limite conhecido desse desenho e o raio: com um raio generoso, uma conducao de dominio muito
proximo nunca sai dele e vira um episodio unico. `RAIO_DO_TOQUE` e o parametro a calibrar, e
enquanto ele for preliminar nenhum numero publicado pode depender deste modulo.

Duas coisas que este modulo recusa fazer, e o motivo de cada uma.

Empate nao vira escolha. Dois jogadores a mesma distancia da bola devolvem `None` e a telemetria
declara `ambiguous`, em vez de escolher o de menor `track_id`, que produziria resultado estavel e
errado.

Posicao interpolada nao sustenta toque. A bola interpolada e uma reta entre as duas deteccoes que
fecham a lacuna, e um toque e exatamente onde a trajetoria deixa de ser reta. Associar um jogador a
uma posicao inventada produz um toque que a imagem nao mostra.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import Literal

from contracts import BallDetection, Box, FramePerception, Perception, PlayerDetection

DETECTADO = "detected"
ASSOCIADO = "associated"
AMBIGUO = "ambiguous"
AUSENTE = "absent"

Fonte = Literal["detected", "associated"]
Motivo = Literal["ambiguous", "absent"]

RAIO_DO_TOQUE = 1.2
"""Quantas alturas de jogador de distancia ainda contam como posse.

A distancia e da borda da caixa do jogador ao centro da bola, normalizada pela altura mediana das
caixas de jogador da janela, e nao pela altura da caixa do candidato. Normalizar por candidato
penalizava sistematicamente a caixa cortada: um jogador ocluido pelos joelhos tem altura menor,
entao a mesma distancia em pixel virava distancia normalizada maior, e ele perdia a disputa por ser
o mais dificil de detectar.

`1,2` e alvo preliminar e nao medida. Ele existe para o seam poder ser escrito e testado, e precisa
ser calibrado nas dobras 0 a 2 antes de qualquer numero publicado depender dele. A spec 04 registra
isso como questao aberta, e o raio tambem decide quantos episodios uma conducao produz.
"""

MARGEM_DE_EMPATE = 0.05
"""Diferenca de distancia normalizada abaixo da qual dois jogadores estao a mesma distancia.

Absoluta, em altura de jogador, e nao relativa. A margem relativa degenerava perto de zero: com a
bola dentro das duas caixas, meio pixel de diferenca decidia a disputa, enquanto a propria caixa do
detector oscila varios pixels entre quadros. Cinco centesimos de altura de jogador sao cerca de
cinco pixels num jogador de cem, que e a ordem de grandeza do ruido da caixa.
"""


@dataclass(frozen=True)
class ContactPlayerEvidence:
    """Quem estava com a bola no quadro do contato."""

    player_track_id: int
    confidence: float
    source: Fonte
    team: str | None = None
    role: str | None = None


@dataclass(frozen=True)
class TouchEvidence:
    """O toque anterior ao contato, e de quem foi."""

    frame_index: int
    player_track_id: int
    team: str | None
    confidence: float
    source: Fonte
    role: str | None = None


@dataclass(frozen=True)
class TouchTelemetry:
    """Por que nao houve resposta, quando nao houve, separado por papel.

    Sem os dois motivos, `None` de bola ausente, `None` de ninguem perto e `None` de empate chegam
    iguais ao consumidor, e a diferenca entre os tres e o que diz se o problema e do detector, do
    rastreador ou do limiar. A spec 04 pede exatamente `shooter-at-contact-unavailable` e
    `previous-touch-ambiguous` como modos de falha distintos.
    """

    frames_com_bola: int = 0
    frames_com_bola_medida: int = 0
    frames_com_jogador_perto: int = 0
    quadros_ambiguos: int = 0
    episodios: int = 0
    motivo_do_contato: Motivo | None = None
    motivo_do_toque_anterior: Motivo | None = None


@dataclass(frozen=True)
class _Posse:
    """Um episodio de posse: quadros consecutivos com a bola no raio do mesmo `track_id`."""

    track_id: int
    inicio: int
    fim: int
    team: str | None
    confidence: float
    source: Fonte
    role: str | None = None


def shooter_at_contact(perception: Perception, frame_index: int) -> ContactPlayerEvidence | None:
    """Quem estava com a bola no episodio que cobre o quadro do contato.

    Resolve pelo episodio, e nao pelo quadro exato, porque o instante nao cai numa amostra: o
    detector de quebra refina a posicao pela intersecao das retas e devolve um quadro entre duas
    delas.

    No maximo um episodio cobre um quadro, e isso sai do desenho e nao da sorte: cada quadro entra
    em exatamente um `atual`, `atual` e um trecho contiguo da lista ordenada por indice, e as pontas
    dele sao `inicio` e `fim`. Os intervalos ficam disjuntos e nao existe indice a escolher.
    """
    episodio = _do_contato(
        _episodios(perception), frame_index, TOLERANCIA_EM_PASSOS * _passo(perception)
    )
    if episodio is None:
        return None
    return ContactPlayerEvidence(
        player_track_id=episodio.track_id,
        confidence=episodio.confidence,
        source=episodio.source,
        team=episodio.team,
        role=episodio.role,
    )


def previous_touch(perception: Perception, frame_index: int) -> TouchEvidence | None:
    """O ultimo episodio de posse que termina antes de comecar o episodio do contato.

    Antes do episodio, e nao antes do quadro: excluir so o quadro do contato deixava o proprio
    finalizador ser o toque anterior em qualquer clipe amostrado de perto, e a comparacao com o
    finalizador saia verdadeira tambem para passe.

    O ultimo episodio pode ser do mesmo `track_id` do contato, e isso e correto: conducao e uma
    sequencia de toques do mesmo jogador, com a bola saindo e voltando ao raio entre eles.

    So conta o que cabe em `JANELA_DO_TOQUE_ANTERIOR_S` antes do episodio do contato. Sem esse
    limite o `max` responde sempre que exista qualquer posse anterior no clipe, e um toque de dez
    segundos antes saia com cara de assistencia — o mesmo defeito que `_do_contato` evita com a
    tolerancia dele, e que o docstring do modulo ja descrevia.

    Com relogio indefinido nao ha janela para aplicar, e a resposta e ausente em vez do defeito
    antigo: sem dois quadros com `timestamp_s` nada distingue passe de posse de outra fase.
    """
    episodios = _episodios(perception)
    contato = _do_contato(episodios, frame_index, TOLERANCIA_EM_PASSOS * _passo(perception))
    if contato is None:
        return None
    limite = _instante(perception, contato.inicio)
    if limite is None:
        return None
    anteriores = [
        e
        for e in episodios
        if e.fim < contato.inicio and _dentro_da_janela(perception, e.fim, limite)
    ]
    if not anteriores:
        return None
    ultimo = max(anteriores, key=lambda e: e.fim)
    return TouchEvidence(
        frame_index=ultimo.fim,
        player_track_id=ultimo.track_id,
        team=ultimo.team,
        confidence=ultimo.confidence,
        source=ultimo.source,
        role=ultimo.role,
    )


def telemetry(perception: Perception, frame_index: int) -> TouchTelemetry:
    """Contagens e motivos que explicam a ausencia de resposta, sem expor identificador nenhum."""
    altura = _altura_de_referencia(perception)
    com_bola = com_medida = com_jogador = 0
    ambiguos: list[int] = []
    for quadro in perception.frames:
        if quadro.ball is None:
            continue
        com_bola += 1
        if _medida(quadro.ball) is None:
            continue
        com_medida += 1
        proximos = _ordenados(quadro, altura)
        if not proximos:
            continue
        com_jogador += 1
        if _empatado(proximos):
            ambiguos.append(quadro.frame_index)
    episodios = _episodios(perception)
    contato = _do_contato(episodios, frame_index, TOLERANCIA_EM_PASSOS * _passo(perception))
    return TouchTelemetry(
        frames_com_bola=com_bola,
        frames_com_bola_medida=com_medida,
        frames_com_jogador_perto=com_jogador,
        quadros_ambiguos=len(ambiguos),
        episodios=len(episodios),
        motivo_do_contato=_motivo_do_contato(
            perception, altura, frame_index, contato, TOLERANCIA_EM_PASSOS * _passo(perception)
        ),
        motivo_do_toque_anterior=_motivo_do_toque_anterior(
            episodios, contato, ambiguos, frame_index
        ),
    )


def _motivo_do_toque_anterior(
    episodios: tuple[_Posse, ...],
    contato: _Posse | None,
    ambiguos: list[int],
    frame_index: int,
) -> Motivo | None:
    """`ambiguous` quando o que precedeu o contato foi disputa, `absent` quando nao houve nada.

    Sem essa separacao, bola disputada antes do chute e clipe sem bola nenhuma antes do chute
    chegam identicos ao consumidor, e a spec 04 pede `previous-touch-ambiguous` como modo proprio.
    """
    if contato is None or any(e.fim < contato.inicio for e in episodios):
        return None
    limite = min(frame_index, contato.inicio)
    return AMBIGUO if any(q < limite for q in ambiguos) else AUSENTE


def _motivo_do_contato(
    perception: Perception,
    altura: float,
    frame_index: int,
    contato: _Posse | None,
    tolerancia: float,
) -> Motivo | None:
    """`ambiguous` quando o quadro mais proximo do contato tinha empate, `absent` quando nao havia
    bola medida nem jogador rastreado perto dela."""
    if contato is not None:
        return None
    perto = _quadro_mais_proximo(perception, frame_index, tolerancia)
    if perto is None:
        return AUSENTE
    return AMBIGUO if _empatado(_ordenados(perto, altura)) else AUSENTE


def _quadro_mais_proximo(
    perception: Perception, frame_index: int, tolerancia: float
) -> FramePerception | None:
    """O quadro com bola medida mais proximo do contato, dentro da mesma tolerancia dos episodios.

    Sem a tolerancia, uma disputa a dez segundos do contato explicava o contato como `ambiguous`, e
    a spec 04 separa `shooter-at-contact-unavailable`, que e problema de cobertura, de
    `previous-touch-ambiguous`, que e problema de rastreio ou de limiar. Motivo errado manda a
    taxonomia de erros procurar no lugar errado.
    """
    uteis = [
        q
        for q in perception.frames
        if q.ball is not None
        and _medida(q.ball) is not None
        and abs(q.frame_index - frame_index) <= tolerancia
    ]
    if not uteis:
        return None
    return min(uteis, key=lambda q: (abs(q.frame_index - frame_index), q.frame_index))


def _episodios(perception: Perception) -> tuple[_Posse, ...]:
    """Agrupa os quadros em episodios de posse, na ordem do clipe.

    Quadro sem dono, por bola ausente, bola interpolada, ninguem no raio ou empate, encerra o
    episodio em curso. Encerrar no empate e deliberado: um episodio que atravessa uma disputa nao
    e mais o toque de um jogador so.
    """
    altura = _altura_de_referencia(perception)
    lacuna = LACUNA_MAXIMA_EM_PASSOS * _passo(perception)
    episodios: list[_Posse] = []
    atual: list[tuple[int, PlayerDetection, float]] = []

    def fecha() -> None:
        if not atual:
            return
        melhor = min(atual, key=lambda item: item[2])
        jogador = melhor[1]
        assert jogador.track_id is not None
        episodios.append(
            _Posse(
                track_id=jogador.track_id,
                inicio=atual[0][0],
                fim=atual[-1][0],
                team=jogador.team,
                confidence=_confianca(jogador.confidence, melhor[2]),
                source=DETECTADO if melhor[2] <= 0.0 else ASSOCIADO,
                role=jogador.role,
            )
        )
        atual.clear()

    for quadro in sorted(perception.frames, key=lambda q: q.frame_index):
        situacao = _situacao(quadro, altura)
        if situacao == SEM_EVIDENCIA:
            continue
        if situacao in (LIVRE, DISPUTADO):
            fecha()
            continue
        jogador, distancia = situacao
        if atual and (
            atual[-1][1].track_id != jogador.track_id or quadro.frame_index - atual[-1][0] > lacuna
        ):
            fecha()
        atual.append((quadro.frame_index, jogador, distancia))
    fecha()
    return tuple(episodios)


def _do_contato(
    episodios: tuple[_Posse, ...], frame_index: int, tolerancia: float
) -> _Posse | None:
    """O episodio que cobre o quadro do contato, ou o mais proximo dentro da tolerancia.

    Aceitar o mais proximo e necessario: `find_contact` refina a quebra pela intersecao das duas
    retas e devolve um quadro entre amostras, entao exigir cobertura devolvia ausente em tres de
    cada quatro contatos no passo de producao.

    A tolerancia e obrigatoria porque o mais proximo sem limite responde sempre. Um clipe em que a
    ultima posse acabou dez segundos antes do contato devolveria aquele jogador com cara de
    finalizador, que e o mesmo defeito estavel e errado noutra forma.
    """
    if not episodios:
        return None
    cobrindo = next((e for e in episodios if e.inicio <= frame_index <= e.fim), None)
    if cobrindo is not None:
        return cobrindo
    antes = [e for e in episodios if e.fim < frame_index and frame_index - e.fim <= tolerancia]
    if antes:
        return max(antes, key=lambda e: e.fim)
    depois = [
        e for e in episodios if e.inicio > frame_index and e.inicio - frame_index <= tolerancia
    ]
    if depois:
        return min(depois, key=lambda e: e.inicio)
    return None


def _instante(perception: Perception, frame_index: int) -> float | None:
    """O instante de um indice em segundos, interpolado quando ele nao cai numa amostra.

    O relogio e afim no indice porque o runner monta `timestamp_s = indice / clip.fps`, entao
    interpolar entre as duas amostras mais proximas reconstroi o instante exato e nao uma
    aproximacao dele. `_episodios` guarda indice, nao tempo, e os limites de um episodio caem em
    amostra, mas o inicio do episodio do contato pode ser comparado com quadros de outra especie.

    Sobre todos os quadros, e nao so os com bola: `timestamp_s` e do quadro, existe tenha ou nao
    deteccao, e descartar quadros sem bola so encolheria a base do relogio sem mudar a inclinacao.

    `None` com menos de duas amostras, e nao zero: uma amostra sozinha nao define a inclinacao, e
    completa-la exigiria inventar um fps que a percepcao nao carrega.

    Duplica `play_evidence.contact_time_s` de proposito: `play_evidence` ja importa deste modulo, e
    importar de volta fecharia um ciclo entre as duas features.
    """
    relogio = {quadro.frame_index: quadro.timestamp_s for quadro in perception.frames}
    if frame_index in relogio:
        return relogio[frame_index]
    if len(relogio) < 2:
        return None
    vizinhos = sorted(relogio, key=lambda indice: (abs(indice - frame_index), indice))[:2]
    primeiro, ultimo = min(vizinhos), max(vizinhos)
    inclinacao = (relogio[ultimo] - relogio[primeiro]) / (ultimo - primeiro)
    return relogio[primeiro] + (frame_index - primeiro) * inclinacao


def _dentro_da_janela(perception: Perception, frame_index: int, limite: float) -> bool:
    """Se o fim de um episodio cabe na janela que termina em `limite`.

    Indice sem instante nao entra: um relogio que nao sabe dizer quando aquele quadro aconteceu nao
    pode afirmar que ele esta dentro da janela, e a afirmacao silenciosa e o defeito que a janela
    existe para impedir.
    """
    instante = _instante(perception, frame_index)
    if instante is None:
        return False
    return limite - JANELA_DO_TOQUE_ANTERIOR_S <= instante <= limite


def _passo(perception: Perception) -> float:
    """Passo de amostragem observado, que e a distancia em que o refinamento do contato alcanca.

    Medido na propria percepcao, e nao constante: o frameset `window` roda em passos diferentes por
    conjunto, e um numero fixo de quadros significaria tolerancias diferentes em cada um.

    Medido sobre os quadros que tem bola, e nao sobre todos. `assemble` monta `perception.frames`
    como a uniao dos indices de `players`, `ball` e `field`, entao uma configuracao que ponha
    jogadores ou campo num conjunto mais fino que o da bola produz quadros sem bola entre cada dois
    com bola. O menor salto da uniao seria um, quando a cadencia real da bola e quatro, e a
    tolerancia do contato sairia quatro vezes menor do que devia.

    O menor salto positivo, e nao a mediana. `uniform_indices` amostra em passo constante, entao
    dentro de uma especie qualquer salto maior que o menor e buraco, e a mediana de uma percepcao
    esburacada mede o buraco: com saltos de um e de quinhentos e noventa e nove ela devolve
    trezentos, e uma posse que acabou dez segundos antes do contato passava a responder por ele.

    Com menos de dois quadros com bola devolve um, e nao o menor salto da uniao. Ali a cadencia da
    bola e desconhecida, e cadencia desconhecida tem que virar tolerancia minima: uma percepcao com
    bola num quadro e outro seiscentos quadros depois daria passo seiscentos pela uniao, e uma
    disputa a dez segundos do contato voltaria a explicar o contato.
    """
    indices = sorted(q.frame_index for q in perception.frames if q.ball is not None)
    saltos = [b - a for a, b in zip(indices, indices[1:], strict=False) if b > a]
    return float(min(saltos)) if saltos else 1.0


LACUNA_MAXIMA_EM_PASSOS = 5.0
"""Quantas amostras sem bola medida um episodio de posse atravessa antes de ser encerrado.

Cinco porque e o que `TrackerConfig.max_gap_frames` vale no rastreador da bola: dentro de cinco
amostras o proprio rastreador afirma continuidade e fecha a lacuna por interpolacao, e alem delas
ele desiste. Atravessar mais do que o rastreador atravessa seria afirmar posse contínua sobre um
trecho em que nada afirma continuidade nem da bola.
"""

TOLERANCIA_EM_PASSOS = 2.0
"""Quantos passos de amostragem separam o contato do episodio que ainda pode responder por ele.

Um passo cobre o refinamento entre amostras. O segundo cobre a bola que ja saiu do pe na amostra
imediatamente anterior ao instante, que e o caso normal de um chute.
"""

JANELA_DO_TOQUE_ANTERIOR_S = 3.0
"""Quanto tempo antes do episodio do contato um toque ainda pode ser o toque anterior, em segundos.

Em segundos, e nao em passos de amostragem, porque o que separa um passe do chute e tempo real de
jogo, e o passo varia por frameset: dois passos sao 33 ms no conjunto `dense` e 530 ms no
`oracle-janela`. `TOLERANCIA_EM_PASSOS` nao serve aqui porque mede outra coisa: ela cobre o
refinamento do contato entre amostras, na casa das dezenas de milissegundos, e reusa-la mataria
quase toda assistencia real.

Tres segundos porque e a ordem de grandeza de um passe que ainda pertence a mesma jogada. Acima
disso o toque anterior tende a ser posse de outra fase, e responder por ela seria o mesmo defeito
que `_do_contato` evita com a tolerancia dele.

Preliminar, como `RAIO_DO_TOQUE` e o raio de marcacao. Precisa ser calibrado antes que qualquer
numero publicado dependa dele.
"""


SEM_EVIDENCIA = "sem-evidencia"
LIVRE = "livre"
DISPUTADO = "disputado"

Situacao = Literal["sem-evidencia", "livre", "disputado"]


def _situacao(quadro: FramePerception, altura: float) -> tuple[PlayerDetection, float] | Situacao:
    """O que o quadro diz sobre a posse, separando ausencia de evidencia de evidencia de ausencia.

    Essa separacao e o centro do modulo, e a primeira versao dos episodios nao a fazia. Quadro sem
    bola medida nao diz que a posse acabou, diz que ninguem olhou: a bola interpolada e uma reta
    que o rastreador desenhou, e a bola ausente e o detector nao tendo achado nada. Tratar isso como
    fim de episodio devolve o defeito que os episodios existem para remover, porque uma unica
    amostra interpolada no meio de uma recepcao parte o episodio em dois e o toque anterior passa a
    ser o proprio finalizador meio segundo antes.

    Bola medida com ninguem no raio, ao contrario, e evidencia de que a posse acabou: alguem olhou e
    a bola estava longe de todos. Disputa tambem encerra, porque um episodio que atravessa uma
    disputa nao e mais o toque de um jogador so.
    """
    bola = quadro.ball
    if bola is None or _medida(bola) is None:
        return SEM_EVIDENCIA
    proximos = _ordenados(quadro, altura)
    if not proximos:
        return LIVRE
    if _empatado(proximos):
        return DISPUTADO
    return proximos[0]


def _altura_de_referencia(perception: Perception) -> float:
    """Altura mediana das caixas de jogador da janela, que normaliza toda distancia deste modulo.

    Mediana da janela, e nao altura do candidato: a caixa cortada de um jogador ocluido tem altura
    menor, e normalizar por ela penalizava justamente quem esta mais dificil de detectar.
    """
    alturas = [
        jogador.box.y2 - jogador.box.y1
        for quadro in perception.frames
        for jogador in quadro.players
        if jogador.box.y2 - jogador.box.y1 > 0
    ]
    return median(alturas) if alturas else 0.0


def _medida(bola: BallDetection) -> BallDetection | None:
    """Bola interpolada nao sustenta toque, e bola sem procedencia declarada tambem nao.

    Ler `source` ausente como medida transformaria um detector que nao declara em um que declarou.
    """
    return bola if bola.source == DETECTADO else None


def _ordenados(quadro: FramePerception, altura: float) -> list[tuple[PlayerDetection, float]]:
    """Jogadores rastreados dentro do raio, do mais proximo ao mais distante."""
    bola = quadro.ball
    if bola is None or _medida(bola) is None or altura <= 0:
        return []
    centro = _centro(bola.box)
    candidatos = [
        (jogador, _distancia_da_caixa(jogador.box, centro) / altura)
        for jogador in quadro.players
        if jogador.track_id is not None
    ]
    dentro = [par for par in candidatos if par[1] <= RAIO_DO_TOQUE]
    return sorted(dentro, key=lambda par: par[1])


def _empatado(proximos: list[tuple[PlayerDetection, float]]) -> bool:
    """Empate e diferenca absoluta de distancia normalizada abaixo da margem."""
    if len(proximos) < 2:
        return False
    return proximos[1][1] - proximos[0][1] < MARGEM_DE_EMPATE


def _centro(box: Box) -> tuple[float, float]:
    return ((box.x1 + box.x2) / 2.0, (box.y1 + box.y2) / 2.0)


def _distancia_da_caixa(box: Box, ponto: tuple[float, float]) -> float:
    """Distancia do ponto a borda da caixa, e zero quando o ponto esta dentro dela."""
    x, y = ponto
    dx = max(box.x1 - x, 0.0, x - box.x2)
    dy = max(box.y1 - y, 0.0, y - box.y2)
    return math.hypot(dx, dy)


def _confianca(deteccao: float, distancia: float) -> float:
    """Confianca da associacao: a do detector, reduzida conforme a bola se afasta da caixa.

    Nao e probabilidade calibrada e nao pretende ser. Ela serve para o consumidor ordenar dois
    candidatos; publicar isso como probabilidade seria inventar uma escala que ninguem mediu. Valor
    nao finito do detector vira zero, porque `min(1.0, nan)` devolve `1.0` e um `nan` no detector
    viraria confianca maxima.
    """
    if not math.isfinite(deteccao) or not math.isfinite(distancia):
        return 0.0
    folga = max(0.0, 1.0 - distancia / RAIO_DO_TOQUE)
    return round(max(0.0, min(1.0, deteccao)) * folga, 6)
