"""As medidas cruas da jogada que chega ao chute, em unidade fisica e sem nenhuma decisao dentro.

`type_of_play` hoje le a trajetoria em pixel por quadro e conta imobilidade em quadros, e as duas
escalas mudam de significado quando a cadencia muda. Pixel por quadro depende de fps: a mesma bola
a 15 fps anda quatro vezes mais por quadro do que a 59,94 fps, e o limiar de 1,5 px por quadro
significa 22,5 px/s num conjunto e 90 px/s no outro. Pixel depende tambem do zoom: 1,5 px num
jogador de 30 px de altura e outro movimento que 1,5 px num jogador de 200 px. E `PARADA_MIN_QUADROS
= 24` conta quadros amostrados como se fossem tempo, o que a 5 fps da 4,8 s de bola parada.

Aqui tudo e medido em segundo e em altura de jogador. Segundo vem de `FramePerception.timestamp_s`,
que o runner monta como `indice / clip.fps`, entao a leitura nao muda quando o conjunto de quadros
muda de passo. Altura de jogador e a mediana das caixas da janela, a mesma normalizacao que
`touch.py` usa, entao a leitura nao muda com o zoom da camera.

A separacao entre medir e decidir e o ponto do modulo. Enquanto a regra e o medidor moram na mesma
funcao, calibrar um limiar exige reescrever a leitura, e nao existe como reproduzir o numero antigo
depois. Com a medida separada, a regra recebe evidencia com unidade declarada e a calibracao das
dobras 0 a 2 mexe so nos limiares.

`measure` nunca levanta e nunca decide. Evidencia fraca vira `None` no campo correspondente, e
`PlayTelemetry` diz quanto foi observado, para o consumidor separar "a bola estava parada" de "nao
houve deteccao suficiente para dizer".

Quatro recusas, e o motivo de cada uma.

Posicao interpolada nao prova imobilidade. A interpolacao e uma reta entre as duas deteccoes que
fecham a lacuna, e duas amostras interpoladas no mesmo ponto sao consequencia da reta e nao
observacao de bola quieta. Contar isso como imobilidade transformaria lacuna de deteccao em bola
parada, que e exatamente o ramo que a feature precisa decidir.

Nada depois do contato entra na janela. Bola que para depois do chute nao e sinal de nada: ela para
porque bateu na rede, no goleiro ou no chao, e incluir esses quadros faria todo chute defendido
parecer bola parada.

Empate na amostra do contato prefere a anterior. Duas deteccoes a mesma distancia do instante
descrevem a bola antes e depois de sair do pe, e a anterior e a que ainda tem a bola no pe.

Sem finalizador o sinal fica `None` e nao `False`. `False` afirma que o toque anterior foi de outro
jogador, e sem finalizador identificado nenhuma comparacao foi feita.

Nao decide classe nenhuma, nao mede assistencia e nao le `shot.shooter_track_id`, que sai sempre
ausente do modulo de chute por razao estrutural documentada em `edge/modules/shot/kinematics.py`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median
from typing import Literal

from contracts import Box, FramePerception, Perception
from edge.features.base import ShotOutput, ShotProvenance
from edge.features.touch import (
    ContactPlayerEvidence,
    TouchEvidence,
    TouchTelemetry,
    previous_touch,
    shooter_at_contact,
)
from edge.features.touch import telemetry as touch_telemetry

BallSupport = Literal["detected", "interpolated", "mixed", "none"]

DETECTED = "detected"
UNKNOWN = "unknown"
NO_SUPPORT: BallSupport = "none"

SUPPORT_VOCABULARY: dict[str, BallSupport] = {
    "detected": "detected",
    "interpolated": "interpolated",
    "mixed": "mixed",
}
"""O que `ball_support` do modulo de chute significa aqui, e o que e jogado no mesmo balde.

O vocabulario de la tem cinco valores: `detected`, `interpolated`, `mixed`, `absent` e `unknown`.
Aqui sao quatro, porque os tres ultimos casos dizem a mesma coisa para quem le a evidencia: nao
existe posicao de bola medida sustentando o instante. `absent` e o quadro escolhido sem bola por
perto, `unknown` e posicao sem procedencia declarada, e valor fora do vocabulario e detector que
mudou sem avisar. Distinguir os tres aqui daria a impressao de que a diferenca entre eles muda a
leitura da jogada, e ela nao muda: nenhum dos tres autoriza medir sobre aquela posicao. Quem precisa
da diferenca a encontra intacta no registro do chute.
"""

JANELA_S = 1.5
"""Quanto tempo antes do contato entra na leitura, em segundos.

E o `before_s` do conjunto `window` de `config/pipeline.toml`, e nao um numero escolhido aqui: ler
mais do que a janela que o pipeline recorta pediria quadro que os modulos de percepcao nao rodaram.

Preliminar. Precisa ser calibrado nas dobras 0 a 2 antes de qualquer numero publicado depender
dele, e a calibracao pode querer menos que a janela inteira.
"""

PARADA_ALTURAS_POR_S = 0.96
"""Abaixo desta velocidade a bola conta como imovel, em altura de jogador por segundo.

Conversao do limiar que `type_of_play` ja usa, para a unidade deste modulo: 1,5 px por quadro a
59,94 fps sao 89,9 px/s, e a altura mediana da caixa do finalizador medida em `body_part.py` sobre
os 38 clipes anotados e 93,4 px, o que da 0,96 altura por segundo. O valor e o mesmo movimento; o
que muda e que ele deixa de significar coisas diferentes em cada fps e em cada zoom.

Preliminar. Precisa ser calibrado nas dobras 0 a 2 antes de qualquer numero publicado depender
dele, porque a conversao preserva o limiar antigo e nao o justifica.
"""

LACUNA_MAX_S = 0.4
"""Quanto tempo pode separar duas deteccoes para elas ainda descreverem o mesmo trecho.

O rastreador da bola fecha lacuna de no maximo `max_gap_frames = 5` amostras, e a especie `ball`
roda no conjunto `dense` a 15 fps, o que da 0,33 s. Acima disso o rastreador declarou perda, e duas
deteccoes separadas por mais que isso nao provam o que aconteceu entre elas: somar o intervalo como
imobilidade transformaria bola perdida de vista em bola parada.

Preliminar. Precisa ser calibrado nas dobras 0 a 2 antes de qualquer numero publicado depender
dele, junto com o passo real do conjunto em que a bola roda.
"""

CONTATO_TOLERANCIA_S = 0.1
"""Quanto a amostra de bola pode se afastar do instante e ainda responder por ele, em segundos.

O passo do conjunto `dense` e 1/15 = 0,067 s, entao uma tolerancia de 0,1 s alcanca a amostra
vizinha e nao a seguinte. A tolerancia e obrigatoria porque a amostra mais proxima sem limite
responde sempre: num clipe em que a ultima deteccao ficou dez segundos antes do contato, ela sairia
com cara de bola no pe do finalizador.

Preliminar. Precisa ser calibrado nas dobras 0 a 2 antes de qualquer numero publicado depender
dele.
"""

MIN_TRACKS_PARA_CORTE = 2
"""Quantas trajetorias os dois quadros precisam ter para a perda de todas elas contar como corte.

O rastreador e IoU guloso, entao um corte de camera perde toda trajetoria de uma vez. Com um
`track_id` so em cada lado, perder o unico e o caso comum de jogador que saiu do quadro, e chamar
isso de corte declararia corte em quase todo clipe.

Preliminar. Precisa ser calibrado nas dobras 0 a 2 antes de qualquer numero publicado depender
dele.
"""


@dataclass(frozen=True)
class PlayEvidence:
    """A evidencia da jogada, em segundo e em altura de jogador, sem classe nenhuma decidida.

    Todo campo `None` e recusa e nao zero: significa que a evidencia nao sustentava a medida, e
    `PlayTelemetry` diz o que faltou. Zero significaria o contrario, que a medida foi feita e deu
    zero, e as duas coisas chegam iguais ao consumidor se a ausencia virar numero.
    """

    shot_source: str
    shot_confidence: float
    shot_ball_support: BallSupport
    stationary_s: float | None
    """Quanto tempo a bola passou imovel imediatamente antes do contato, contado de tras para
    frente e interrompido no primeiro trecho em movimento ou na primeira lacuna longa.

    Zero e medida: a bola estava em movimento no ultimo trecho. `None` e recusa: nao houve duas
    deteccoes na janela, ou nao houve jogador para dar escala.
    """

    path_player_heights: float | None
    arrival_speed_player_heights_s: float | None
    observed_fraction: float
    """Fracao dos quadros da janela em que um detector viu a bola.

    Quadro com bola interpolada e quadro sem bola nenhuma ficam no denominador de proposito: o que
    este numero mede e quanto da janela foi observado de verdade, e tirar do denominador o que nao
    foi observado faria toda janela parecer inteiramente observada.
    """

    previous_touch_is_shooter: bool | None
    previous_touch_age_s: float | None
    previous_touch_source: str | None
    camera_cut: bool


@dataclass(frozen=True)
class PlayTelemetry:
    """Quanto foi observado, para separar medida fraca de medida ausente.

    Sem isto, `stationary_s = None` por clipe sem bola detectada e `stationary_s = None` por clipe
    sem jogador chegam iguais ao consumidor, e a diferenca entre os dois e o que diz se o problema
    e do detector de bola ou do de jogador.
    """

    frames_in_window: int
    ball_samples: int
    ball_detected: int
    median_player_height_px: float | None
    contact_sample_offset_s: float | None
    """Quanto a amostra usada no contato esta depois do instante, em segundos, com sinal.

    Com sinal e nao em modulo: negativo e bola ainda no pe, positivo e bola ja saindo, e a
    diferenca entre os dois e o que explica um finalizador identificado errado.
    """

    touch: TouchTelemetry


@dataclass(frozen=True)
class _Segment:
    """Um trecho entre duas deteccoes consecutivas da janela.

    Trecho e sempre de deteccao a deteccao. Um trecho que comecasse ou terminasse em posicao
    interpolada mediria a reta que o rastreador desenhou, e nao a bola.
    """

    distance_px: float
    elapsed_s: float
    speed_heights_s: float


def measure(
    perception: Perception, shot: ShotOutput, provenance: ShotProvenance
) -> tuple[PlayEvidence, PlayTelemetry]:
    """Le a jogada anterior ao contato e devolve evidencia com unidade, mais o que foi observado.

    Nunca levanta e nunca decide. Percepcao vazia, contato fora das amostras, bola nao detectada e
    janela sem jogador saem todos como `None` no campo que a evidencia nao sustenta, porque uma
    feature que estoura custa as outras do mesmo clipe e um numero inventado chega ao consumidor com
    a mesma cara de um medido.
    """
    t_contact = contact_time_s(perception, shot.frame_index)
    frames = () if t_contact is None else window(perception, t_contact)
    scale = player_scale_px(frames)
    detected = _detected(frames)
    segments = _segments(detected, scale)
    stationary, path, arrival = _scaled(detected, segments, scale)

    sample = None if t_contact is None else contact_sample(perception, t_contact)
    offset = None if sample is None or t_contact is None else sample.timestamp_s - t_contact
    anchor = shot.frame_index if sample is None else sample.frame_index
    shooter = None if sample is None else shooter_at_contact(perception, anchor)
    touch = None if sample is None else previous_touch(perception, anchor)

    evidence = PlayEvidence(
        shot_source=provenance.source or UNKNOWN,
        shot_confidence=_confianca_do_chute(provenance.confidence),
        shot_ball_support=_support(provenance.ball_support),
        stationary_s=stationary,
        path_player_heights=path,
        arrival_speed_player_heights_s=arrival,
        observed_fraction=len(detected) / len(frames) if frames else 0.0,
        previous_touch_is_shooter=_is_shooter(shooter, touch),
        previous_touch_age_s=_touch_age_s(perception, touch, t_contact),
        previous_touch_source=None if touch is None else touch.source,
        camera_cut=_camera_cut(frames),
    )
    telemetry = PlayTelemetry(
        frames_in_window=len(frames),
        ball_samples=sum(1 for frame in frames if frame.ball is not None),
        ball_detected=len(detected),
        median_player_height_px=scale,
        contact_sample_offset_s=offset,
        touch=touch_telemetry(perception, anchor),
    )
    return evidence, telemetry


def contact_time_s(perception: Perception, frame_index: int) -> float | None:
    """O instante do contato em segundos, interpolado no relogio quando ele nao cai numa amostra.

    O contato nao e amostra no caso normal. `find_contact` refina a quebra pela intersecao das duas
    retas ajustadas e devolve um quadro entre amostras, e no conjunto `dense`, de passo quatro a
    59,94 fps, tres de cada quatro valores refinados nao coincidem com amostra. Ler
    `timestamp_s` so do quadro exato devolveria ausente em quase todo clipe.

    A interpolacao e afim no indice porque o relogio e afim no indice: o runner monta
    `timestamp_s = indice / clip.fps`. Entao interpolar entre as duas amostras mais proximas
    reconstroi o instante exato, e nao uma aproximacao dele.

    `None` com menos de duas amostras, e nao zero: uma amostra sozinha nao define a inclinacao do
    relogio, e completa-la exigiria inventar um fps que a percepcao nao carrega.
    """
    clock = {frame.frame_index: frame.timestamp_s for frame in perception.frames}
    if frame_index in clock:
        return clock[frame_index]
    if len(clock) < 2:
        return None
    nearest = sorted(clock, key=lambda index: (abs(index - frame_index), index))[:2]
    first, last = min(nearest), max(nearest)
    inclinacao = (clock[last] - clock[first]) / (last - first)
    return clock[first] + (frame_index - first) * inclinacao


def window(perception: Perception, t_contact: float) -> tuple[FramePerception, ...]:
    """Os quadros de `t_contact - JANELA_S` ate o proprio contato, em ordem de tempo.

    Em segundos e nao em quadros, porque contar quadros mede coisas diferentes em cada conjunto: 90
    quadros sao 1,5 s a 59,94 fps e 6 s a 15 fps.

    Nada depois do contato entra. Bola que para depois do chute para porque bateu em algo, e incluir
    esses quadros faria chute defendido parecer bola parada.
    """
    return tuple(
        sorted(
            (
                frame
                for frame in perception.frames
                if t_contact - JANELA_S <= frame.timestamp_s <= t_contact
            ),
            key=lambda frame: (frame.timestamp_s, frame.frame_index),
        )
    )


def player_scale_px(frames: Sequence[FramePerception]) -> float | None:
    """Altura mediana da caixa de jogador na janela, em pixel, que normaliza toda distancia daqui.

    Mediana da janela, e nao altura de um candidato: a caixa cortada de um jogador ocluido tem
    altura menor, e normalizar por ela daria velocidade maior para a mesma bola, ou seja o resultado
    dependeria de quem estava mais dificil de detectar. E a mesma normalizacao de `touch.py`, de
    proposito, para que raio de toque e velocidade de bola fiquem na mesma escala.

    `None` e nao zero quando nao ha jogador nenhum: zero dividiria e estouraria, e substituir por um
    valor plausivel produziria velocidade em altura de um jogador que ninguem viu.
    """
    heights = [
        player.box.y2 - player.box.y1
        for frame in frames
        for player in frame.players
        if player.box.y2 - player.box.y1 > 0.0
    ]
    return median(heights) if heights else None


def contact_sample(perception: Perception, t_contact: float) -> FramePerception | None:
    """O quadro com bola detectada mais proximo do instante, dentro de `CONTATO_TOLERANCIA_S`.

    So bola detectada. A posicao interpolada e a reta que o rastreador desenhou entre as duas
    deteccoes que fecham a lacuna, e o contato e exatamente onde a trajetoria deixa de ser reta:
    associar um jogador aquela posicao produz um finalizador que a imagem nao mostra.

    Empate escolhe a anterior. Duas deteccoes a mesma distancia do instante descrevem a bola antes e
    depois de sair do pe, e a anterior e a que ainda a tem no pe. Escolher a posterior mediria para
    onde a bola foi, e num chute forte ela ja esta longe do finalizador.

    `None` fora da tolerancia, e nao a amostra mais proxima que existir: sem limite, a ultima
    deteccao de um clipe sairia como bola no pe do finalizador mesmo dez segundos antes do chute.
    """
    candidates = [
        frame
        for frame in perception.frames
        if frame.ball is not None
        and frame.ball.source == DETECTED
        and abs(frame.timestamp_s - t_contact) <= CONTATO_TOLERANCIA_S
    ]
    if not candidates:
        return None
    return min(
        candidates, key=lambda frame: (abs(frame.timestamp_s - t_contact), frame.timestamp_s)
    )


def _confianca_do_chute(declarada: float | None) -> float:
    """Zero quando ausente ou nao finita, porque `is None` nao pega `nan`.

    `measure` aceita uma `ShotProvenance` montada a mao, e nao so a que `shot_provenance` filtra,
    entao a guarda existe nos dois lugares de proposito: um `nan` aqui atravessa para o vetor que a
    fatia 5 entrega ao softmax, e um `nan` num logit zera a distribuicao inteira sem levantar.
    """
    if declarada is None or not math.isfinite(declarada):
        return 0.0
    return float(declarada)


def _support(declared: str | None) -> BallSupport:
    """Traduz o `ball_support` do chute para o vocabulario de quatro valores desta evidencia.

    Valor desconhecido cai em `none` em vez de passar adiante. Passar adiante deixaria um campo
    tipado como quatro valores carregar um quinto, e o consumidor que ramificasse nos quatro
    conhecidos trataria o quinto como o ramo `else` sem saber que estava fazendo isso.
    """
    if declared is None:
        return NO_SUPPORT
    return SUPPORT_VOCABULARY.get(declared, NO_SUPPORT)


def _detected(frames: Sequence[FramePerception]) -> tuple[FramePerception, ...]:
    """Os quadros da janela em que um detector viu a bola, na ordem em que a janela veio.

    `source = None` nao conta como detectado, pela mesma regra de `touch._medida`: ler procedencia
    ausente como medida transformaria um detector que nao declara em um que declarou.
    """
    return tuple(
        frame for frame in frames if frame.ball is not None and frame.ball.source == DETECTED
    )


def _segments(detected: Sequence[FramePerception], scale: float | None) -> tuple[_Segment, ...]:
    """Os trechos entre deteccoes consecutivas, com velocidade em altura de jogador por segundo.

    Divide pelo tempo decorrido e nao pela posicao na lista. Duas amostras vizinhas na lista podem
    estar a doze quadros de distancia, porque a especie `ball` roda a 15 fps sobre um clipe de 59,94
    e as deteccoes tem buraco: dividir pela posicao daria velocidade doze vezes maior.

    Trecho sem tempo decorrido e descartado. Dois quadros com o mesmo `timestamp_s` dariam
    velocidade infinita, que nenhum relogio sustenta.
    """
    if scale is None or scale <= 0.0:
        return ()
    segments: list[_Segment] = []
    for before, after in zip(detected, detected[1:], strict=False):
        elapsed = after.timestamp_s - before.timestamp_s
        if elapsed <= 0.0:
            continue
        x1, y1 = _center(before)
        x2, y2 = _center(after)
        distance = math.hypot(x2 - x1, y2 - y1)
        segments.append(
            _Segment(
                distance_px=distance,
                elapsed_s=elapsed,
                speed_heights_s=distance / elapsed / scale,
            )
        )
    return tuple(segments)


def _scaled(
    detected: Sequence[FramePerception], segments: Sequence[_Segment], scale: float | None
) -> tuple[float | None, float | None, float | None]:
    """As tres medidas que dependem da escala do jogador, ou a recusa das tres juntas.

    Recusa quando ha menos de duas deteccoes na janela, quando nao houve jogador para dar escala,
    ou quando nenhum par de deteccoes tem tempo decorrido entre si. Nos tres casos a resposta
    errada disponivel seria zero, que se le como bola parada e sem deslocamento: o clipe sem
    deteccao sairia identico ao clipe com a bola quieta no chao.
    """
    if scale is None or len(detected) < 2 or not segments:
        return (None, None, None)
    return (
        _stationary_s(segments),
        sum(segment.distance_px for segment in segments) / scale,
        max(segment.speed_heights_s for segment in segments[-2:]),
    )


def _stationary_s(segments: Sequence[_Segment]) -> float:
    """Quanto tempo a bola passou imovel imediatamente antes do contato, em segundos.

    Conta de tras para frente e para no primeiro trecho em movimento: uma bola que ficou parada,
    andou e parou de novo nao e bola parada, e somar os dois trechos diria que sim.

    Para tambem na primeira lacuna longa, e essa e a parte que nao aparece contando quadros. Duas
    deteccoes no mesmo ponto separadas por mais de `LACUNA_MAX_S` nao provam que a bola ficou ali no
    meio: elas provam que o rastreador a perdeu de vista, e o intervalo pode ter tido um passe
    inteiro dentro.
    """
    total = 0.0
    for segment in reversed(segments):
        if segment.speed_heights_s > PARADA_ALTURAS_POR_S or segment.elapsed_s > LACUNA_MAX_S:
            break
        total += segment.elapsed_s
    return total


def _is_shooter(shooter: ContactPlayerEvidence | None, touch: TouchEvidence | None) -> bool | None:
    """Se quem tocou a bola antes do chute e o proprio finalizador.

    `None` e nao `False` quando falta um dos dois lados. `False` afirma que o toque anterior foi de
    outro jogador, o que separa passe de conducao, e sem finalizador identificado ou sem toque
    anterior nenhuma comparacao foi feita. Os dois chegam `None` tambem quando nao houve amostra de
    bola detectada no contato, porque e nela que os dois sao ancorados.
    """
    if shooter is None or touch is None:
        return None
    return touch.player_track_id == shooter.player_track_id


def _touch_age_s(
    perception: Perception, touch: TouchEvidence | None, t_contact: float | None
) -> float | None:
    """Quanto tempo separa o toque anterior do contato, em segundos.

    Em segundos e nao em quadros porque e assim que a idade se compara entre clipes: 20 quadros sao
    0,33 s a 59,94 fps e 1,33 s a 15 fps, e a regra que separa passe de conducao olha o tempo.
    """
    if touch is None or t_contact is None:
        return None
    touch_time = contact_time_s(perception, touch.frame_index)
    return None if touch_time is None else t_contact - touch_time


def _camera_cut(frames: Sequence[FramePerception]) -> bool:
    """Se algum par de quadros vizinhos da janela perdeu todas as trajetorias de uma vez.

    O rastreador e IoU guloso entre quadros consecutivos, entao um corte de camera nao sobrevive:
    nenhuma caixa do quadro novo cobre caixa do antigo e toda trajetoria reabre com identificador
    novo. Perda total e o sinal; troca parcial nao e corte, e sim jogador entrando ou saindo do
    quadro, que acontece o tempo todo.

    Os dois lados precisam de `MIN_TRACKS_PARA_CORTE` trajetorias. Com uma so em cada lado, perder a
    unica e o caso comum de quem saiu do quadro, e chamar isso de corte declararia corte em quase
    todo clipe.
    """
    ids = [_track_ids(frame) for frame in frames]
    return any(
        len(before) >= MIN_TRACKS_PARA_CORTE
        and len(after) >= MIN_TRACKS_PARA_CORTE
        and not (before & after)
        for before, after in zip(ids, ids[1:], strict=False)
    )


def _track_ids(frame: FramePerception) -> frozenset[int]:
    return frozenset(player.track_id for player in frame.players if player.track_id is not None)


def _center(frame: FramePerception) -> tuple[float, float]:
    box = frame.ball.box if frame.ball is not None else None
    if box is None:  # pragma: no cover - o chamador so passa quadro com bola detectada
        raise ValueError(f"quadro {frame.frame_index} nao tem bola para medir")
    return _box_center(box)


def _box_center(box: Box) -> tuple[float, float]:
    return ((box.x1 + box.x2) / 2.0, (box.y1 + box.y2) / 2.0)
