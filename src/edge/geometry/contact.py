"""O instante do contato como quebra da trajetoria da bola, sem peso e sem treino.

A finalizacao e o unico momento do clipe em que a bola muda de direcao e ganha velocidade ao mesmo
tempo, junto de um jogador. Anzer e Bauer (2021) resolveram o mesmo problema por essa via depois de
medir offsets de ate 20 s na anotacao manual, e chegaram a 0,23 s de erro medio. Aqui a
formulacao e a mesma ideia com o vocabulario deste projeto.

Nao abre video, nao importa cv2, nao carrega peso e nao decide de quem e a bola. Recebe posicoes e
devolve um quadro. Quem decide quais quadros sao candidatos, ou seja, em quais deles a bola esta
perto de um jogador, e o modulo, porque isso depende da especie `players` e esta funcao existe
justamente para nao depender de nada.

Duas escolhas que valem explicacao. A primeira e que a velocidade sai da diferenca de indice de
quadro, e nao da posicao na lista, entao lacuna de deteccao nao vira aceleracao falsa: a bola
sumir por seis quadros e reaparecer no dobro da distancia e a mesma velocidade, nao o dobro dela.
A segunda e que o refino por intersecao devolve instante entre amostras, o que importa porque o
`coarse` amostra a 5 fps e o contato dura menos que os 12 quadros entre duas amostras.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass

TURN_WEIGHT = 0.6
"""A virada pesa mais que a aceleracao porque ela sobrevive a erro de escala do detector."""

SPEED_WEIGHT = 0.4
MIN_OBSERVATIONS = 4
AJUSTE = 3
"""Pontos de cada lado no ajuste da reta. Tres aguenta um outlier sem alisar a quebra."""

MIN_SCORE = 0.12
"""Abaixo disso a trajetoria nao tem quebra: bola parada ou passe reto pontuam perto de zero."""

EPS = 1e-9


@dataclass(frozen=True)
class Observation:
    frame_index: int
    x: float
    y: float


@dataclass(frozen=True)
class Contact:
    frame_index: int
    confidence: float
    turn_deg: float
    speed_ratio: float


@dataclass(frozen=True)
class Candidate:
    """Uma quebra da trajetoria antes de escolher, com o quadro refinado e o amostrado lado a lado.

    Os dois quadros respondem perguntas diferentes. `frame_index` e o instante que o consumidor
    usa, ja refinado pela intersecao das retas. `sample_index` e a amostra que gerou a quebra, e e
    por ela que se confere se dois candidatos sao a mesma quebra vista duas vezes: o refino move o
    quadro para dentro do intervalo vizinho, entao duas amostras seguidas podem cair perto uma da
    outra depois de refinadas sem serem a mesma coisa.
    """

    frame_index: int
    sample_index: int
    score: float
    turn_deg: float
    speed_ratio: float


def find_contact(
    ball: Sequence[Observation],
    *,
    frame_count: int,
    candidates: Collection[int] | None = None,
) -> Contact | None:
    """O quadro da maior quebra da trajetoria, ou `None` quando nao ha quebra que se sustente.

    `None` e resposta legitima e frequente: bola nunca detectada, trajetoria curta demais, ou
    movimento sem quebra nenhuma. Quem chama decide o que fazer com isso, e a resposta certa nunca
    e inventar um quadro sem dizer que inventou.
    """
    return best_contact(rank_contacts(ball, frame_count=frame_count, candidates=candidates))


def rank_contacts(
    ball: Sequence[Observation],
    *,
    frame_count: int,
    candidates: Collection[int] | None = None,
) -> tuple[Candidate, ...]:
    """Todas as quebras da trajetoria, da maior para a menor, antes do corte de `MIN_SCORE`.

    Existe separada de `find_contact` porque "qual quadro e o contato" e "o contato estava entre
    as quebras descartadas" sao perguntas diferentes sobre o mesmo score. Medir a segunda
    recalculando o score em outro lugar transformaria qualquer divergencia entre as duas
    implementacoes em conclusao sobre o modelo, que e o erro que esta funcao existe para nao
    cometer.

    Devolve vazio em vez de levantar quando a trajetoria e curta demais: bola mal detectada e o
    caso normal de um clipe ruim, nao uso errado da funcao.
    """
    pontos = _ordenados(ball, frame_count)
    if len(pontos) < MIN_OBSERVATIONS:
        return ()
    achados: list[Candidate] = []
    for posicao in range(1, len(pontos) - 1):
        if candidates is not None and pontos[posicao].frame_index not in candidates:
            continue
        score, turn_deg, speed_ratio = _score(pontos, posicao)
        achados.append(
            Candidate(
                frame_index=_refinado(pontos, posicao, frame_count),
                sample_index=pontos[posicao].frame_index,
                score=score,
                turn_deg=turn_deg,
                speed_ratio=speed_ratio,
            )
        )
    return tuple(sorted(achados, key=lambda candidato: -candidato.score))


def best_contact(ordenados: Sequence[Candidate]) -> Contact | None:
    """A quebra escolhida de uma lista ja ordenada, ou `None` quando nenhuma se sustenta.

    Recebe a lista pronta em vez de recalcula-la para quem precisa registrar o que considerou
    pagar o score uma vez so, e para o registro ser exatamente aquilo sobre o que a escolha caiu.

    Empate vai para a quebra mais cedo, porque `rank_contacts` ordena de forma estavel sobre
    posicoes crescentes. E arbitrario, e o importante e ser o mesmo criterio sempre: empate
    resolvido por ordem de dicionario mudaria o instante entre duas execucoes iguais.
    """
    if not ordenados:
        return None
    melhor = ordenados[0]
    if melhor.score < MIN_SCORE:
        return None
    return Contact(
        frame_index=melhor.frame_index,
        confidence=_confianca(ordenados, melhor.score),
        turn_deg=melhor.turn_deg,
        speed_ratio=melhor.speed_ratio,
    )


def _ordenados(ball: Sequence[Observation], frame_count: int) -> tuple[Observation, ...]:
    """Ordena por quadro, descarta o que cai fora do clipe e colapsa quadro repetido.

    Quadro repetido acontece quando o detector devolve dois candidatos a bola no mesmo quadro e o
    chamador nao reduziu. Deixar passar daria velocidade infinita na divisao por delta zero.
    """
    dentro = [o for o in ball if 0 <= o.frame_index < frame_count]
    unicos: dict[int, Observation] = {}
    for o in sorted(dentro, key=lambda o: o.frame_index):
        unicos.setdefault(o.frame_index, o)
    return tuple(unicos.values())


def _score(pontos: tuple[Observation, ...], posicao: int) -> tuple[float, float, float]:
    antes = _velocidade(pontos[posicao - 1], pontos[posicao])
    depois = _velocidade(pontos[posicao], pontos[posicao + 1])
    v_antes, v_depois = _norma(antes), _norma(depois)

    cosseno = _cosseno(antes, depois)
    turn = (1.0 - cosseno) / 2.0
    ganho = (v_depois - v_antes) / (v_depois + v_antes + EPS)
    aceleracao = max(0.0, ganho)

    score = TURN_WEIGHT * turn + SPEED_WEIGHT * aceleracao
    razao = v_depois / v_antes if v_antes > EPS else math.inf
    return score, math.degrees(math.acos(max(-1.0, min(1.0, cosseno)))), razao


def _velocidade(a: Observation, b: Observation) -> tuple[float, float]:
    delta = b.frame_index - a.frame_index
    return ((b.x - a.x) / delta, (b.y - a.y) / delta)


def _norma(v: tuple[float, float]) -> float:
    return math.hypot(v[0], v[1])


def _cosseno(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Bola parada de um dos lados nao tem direcao, e chamar isso de virada de 90 graus seria
    inventar. Um vetor nulo devolve cosseno 1, ou seja, virada zero, e deixa a aceleracao decidir.
    """
    na, nb = _norma(a), _norma(b)
    if na < EPS or nb < EPS:
        return 1.0
    return (a[0] * b[0] + a[1] * b[1]) / (na * nb)


def _refinado(pontos: tuple[Observation, ...], posicao: int, frame_count: int) -> int:
    """Cruza a reta de entrada com a de saida e devolve o quadro do encontro.

    O `coarse` amostra a 5 fps, ou seja, um quadro a cada 12 a 59,94 fps, e o contato dura menos
    que isso. Devolver a amostra crua cravaria o erro em ate 6 quadros por construcao, antes de
    qualquer erro do detector.

    As duas retas sao ajustadas SEM o proprio ponto da quebra. Usar os segmentos que passam por ele
    faria as duas se encontrarem exatamente nele, e o refino seria um `round` disfarcado. E o
    ponto da quebra e justamente o mais contaminado: e nele que a amostragem mistura o antes e o
    depois.
    """
    antes = pontos[max(0, posicao - AJUSTE) : posicao]
    depois = pontos[posicao + 1 : posicao + 1 + AJUSTE]
    if len(antes) < 2 or len(depois) < 2:
        return pontos[posicao].frame_index

    candidatos: list[tuple[float, float]] = []
    for eixo in (0, 1):
        origem_a, inclinacao_a = _ajuste(antes, eixo)
        origem_d, inclinacao_d = _ajuste(depois, eixo)
        separacao = inclinacao_a - inclinacao_d
        if abs(separacao) < EPS:
            continue
        candidatos.append((abs(separacao), (origem_d - origem_a) / separacao))
    if not candidatos:
        return pontos[posicao].frame_index

    _, instante = max(candidatos, key=lambda par: par[0])
    piso = float(pontos[posicao - 1].frame_index)
    teto = float(pontos[posicao + 1].frame_index)
    return max(0, min(frame_count - 1, round(max(piso, min(teto, instante)))))


def _ajuste(pontos: Sequence[Observation], eixo: int) -> tuple[float, float]:
    """Minimos quadrados de uma coordenada contra o indice de quadro, devolvendo origem e taxa."""
    ts = [float(p.frame_index) for p in pontos]
    vs = [(p.x, p.y)[eixo] for p in pontos]
    n = len(ts)
    media_t = sum(ts) / n
    media_v = sum(vs) / n
    variancia = sum((t - media_t) ** 2 for t in ts)
    if variancia < EPS:
        return media_v, 0.0
    inclinacao = sum((t - media_t) * (v - media_v) for t, v in zip(ts, vs, strict=True)) / variancia
    return media_v - inclinacao * media_t, inclinacao


def _confianca(ordenados: Sequence[Candidate], score: float) -> float:
    """O quanto a quebra escolhida se destaca das outras, e nao o quanto ela e grande.

    Score alto num clipe onde todo quadro tem score alto e ruido de deteccao, nao finalizacao. A
    mediana das concorrentes e a referencia porque ela nao se move quando um unico quadro estoura.
    """
    if len(ordenados) < 2:
        return min(1.0, score)
    scores = sorted(candidato.score for candidato in ordenados)
    mediana = scores[len(scores) // 2]
    return max(0.0, min(1.0, (score - mediana) / (score + mediana + EPS)))
