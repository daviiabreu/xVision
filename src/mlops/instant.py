"""O gabarito manual do instante do contato, e a medida contra ele.

Ele existe porque a marcacao do parceiro nao serve para isto, e o numero e definitivo. Para pousar
`match_events.time_play_in_seconds` dentro do clipe e preciso o offset da transmissao, e o residuo
dele em torno da mediana por partida e tempo de jogo tem p50 de 2,0 s e p90 de 7,5 s. Somando o
grao de 1 s da propria marcacao, a incerteza fica entre 180 e 510 quadros, contra clipes de 240 a
420. A incerteza do gabarito e do tamanho do clipe inteiro.

A ferramenta e headless de proposito, e o motivo tem medida: o `opencv-python-headless` que o lock
trava sai com `GUI: NONE` no wheel de Linux, entao `cv2.imshow` nao roda para ninguem do time que
nao esteja num Mac. Em vez de janela, ela escreve uma folha de contato em PNG e recebe a resposta
por linha de comando.

As duas tolerancias saem de `config/pipeline.toml` e nao de gosto. Com `before_s = 1.5` e
`after_s = 0.5` a 59,94 fps, a janela e 90 quadros antes e 30 depois do instante: erro de ate 30
quadros ainda preserva dois tercos do contexto anterior, e erro acima de 120 tira o contato de
dentro da janela.

Nao abre video por conta propria: o decode passa pelo `FrameProvider`, que e o unico do projeto.
Nao guarda quadro do parceiro em lugar versionado, porque `var/` e ignorado. E o relatorio sai
agregado, sem `shot_id`, pela mesma regra que ja vale para `mlops.evaluate`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import comb
from pathlib import Path

from edge.modules.shot.kinematics import ESTIMADO
from edge.paths import var_dir

ARQUIVO = "instants.jsonl"
OBSERVATIONS_FILE = "instant-observations.jsonl"
"""Procedencia do instante por clipe, ao lado dos eventos da execucao.

Arquivo separado de proposito: `ShotEvent` esta em 1.0 e tem consumidor de API, frontend e
persistencia, entao acrescentar campo ali obrigaria versionar o schema publico para resolver um
problema de avaliacao. Ele mora em `var/`, junto do `events.jsonl`, e carrega identificador de
clipe, entao nao entra em documentacao nem em pagina publica.
"""
TOLERANCIA_ALVO = 30
"""Erro em quadros que preserva dois tercos do contexto anterior da janela."""

TOLERANCIA_LIMITE = 90
"""Erro em quadros acima do qual a janela perde todo o contexto anterior."""


class InstantError(ValueError): ...


@dataclass(frozen=True)
class Anotacao:
    clip_id: str
    frame_index: int
    annotator: str

    def to_dict(self) -> dict[str, object]:
        return {
            "clip_id": self.clip_id,
            "frame_index": self.frame_index,
            "annotator": self.annotator,
        }


def labels_dir(root: Path | None = None) -> Path:
    return (var_dir() if root is None else root) / "labels"


def write(anotacao: Anotacao, root: Path | None = None) -> Path:
    """Acrescenta em vez de sobrescrever: duas anotacoes do mesmo clipe medem a concordancia."""
    destino = labels_dir(root)
    destino.mkdir(parents=True, exist_ok=True)
    caminho = destino / ARQUIVO
    with caminho.open("a", encoding="utf-8") as arquivo:
        arquivo.write(json.dumps(anotacao.to_dict(), ensure_ascii=False) + "\n")
    return caminho


def read(root: Path | None = None) -> tuple[Anotacao, ...]:
    caminho = labels_dir(root) / ARQUIVO
    if not caminho.is_file():
        return ()
    anotacoes: list[Anotacao] = []
    for numero, linha in enumerate(caminho.read_text(encoding="utf-8").splitlines(), start=1):
        if not linha.strip():
            continue
        try:
            documento = json.loads(linha)
            anotacoes.append(
                Anotacao(
                    clip_id=str(documento["clip_id"]),
                    frame_index=int(documento["frame_index"]),
                    annotator=str(documento.get("annotator", "")),
                )
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as erro:
            raise InstantError(f"{caminho}:{numero} nao e uma anotacao valida: {erro}") from erro
    return tuple(anotacoes)


def latest(anotacoes: Iterable[Anotacao]) -> tuple[Anotacao, ...]:
    """Uma anotacao por clipe e por pessoa, a ultima que ela gravou.

    O arquivo e um log e continua sendo: quem volta num clipe e marca de novo acrescenta linha em
    vez de reescrever, e o historico de quem mudou de ideia fica. Mas ler o log cru conta a correcao
    como uma segunda opiniao, e ai `agreement` mede a pessoa contra ela mesma e devolve discordancia
    que nao existe. Quem corrige nao e um segundo anotador.
    """
    ultima: dict[tuple[str, str], Anotacao] = {}
    for anotacao in anotacoes:
        ultima[(anotacao.clip_id, anotacao.annotator)] = anotacao
    return tuple(ultima.values())


def consensus(anotacoes: Iterable[Anotacao]) -> dict[str, int]:
    """A mediana por clipe, que resiste a um anotador destoante sem descartar o trabalho dele."""
    por_clipe: dict[str, list[int]] = {}
    for anotacao in latest(anotacoes):
        por_clipe.setdefault(anotacao.clip_id, []).append(anotacao.frame_index)
    return {clip: sorted(quadros)[len(quadros) // 2] for clip, quadros in por_clipe.items()}


def agreement(anotacoes: Iterable[Anotacao]) -> dict[str, float] | None:
    """A distancia entre anotadores nos clipes anotados mais de uma vez.

    Sem este numero o gabarito nao tem barra de erro propria, e todo erro medido contra ele fica
    sem saber quanto e do modelo e quanto e de quem anotou.
    """
    por_clipe: dict[str, list[int]] = {}
    for anotacao in latest(anotacoes):
        por_clipe.setdefault(anotacao.clip_id, []).append(anotacao.frame_index)
    duplicados = [sorted(q) for q in por_clipe.values() if len(q) > 1]
    if not duplicados:
        return None
    espalhamentos = [q[-1] - q[0] for q in duplicados]
    ordenado = sorted(espalhamentos)
    return {
        "n_clipes": len(duplicados),
        "mediana_quadros": float(ordenado[len(ordenado) // 2]),
        "max_quadros": float(ordenado[-1]),
    }


@dataclass(frozen=True)
class InstantReport:
    n: int
    mediana_erro: float
    p95_erro: float
    dentro_da_tolerancia: float
    dentro_do_limite: float
    fontes: Mapping[str, int]
    concordancia: Mapping[str, float] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "tolerancia_quadros": TOLERANCIA_ALVO,
            "limite_quadros": TOLERANCIA_LIMITE,
            "mediana_erro_quadros": self.mediana_erro,
            "p95_erro_quadros": self.p95_erro,
            "dentro_da_tolerancia": self.dentro_da_tolerancia,
            "dentro_do_limite": self.dentro_do_limite,
            "fontes": dict(self.fontes),
            "concordancia_entre_anotadores": (
                None if self.concordancia is None else dict(self.concordancia)
            ),
        }


def measure(
    previsto: Mapping[str, int],
    gabarito: Mapping[str, int],
    *,
    fontes: Mapping[str, str] | None = None,
    concordancia: Mapping[str, float] | None = None,
) -> InstantReport:
    """O erro em quadros contra o gabarito manual, agregado e sem identificador de finalizacao."""
    comuns = sorted(set(previsto) & set(gabarito))
    erros = [abs(previsto[clip] - gabarito[clip]) for clip in comuns]
    if not erros:
        return InstantReport(0, 0.0, 0.0, 0.0, 0.0, {}, concordancia)
    ordenado = sorted(erros)
    origem: dict[str, int] = {}
    for clip in comuns:
        chave = (fontes or {}).get(clip, "desconhecida")
        origem[chave] = origem.get(chave, 0) + 1
    return InstantReport(
        n=len(erros),
        mediana_erro=float(ordenado[len(ordenado) // 2]),
        p95_erro=float(ordenado[min(len(ordenado) - 1, int(0.95 * (len(ordenado) - 1)))]),
        dentro_da_tolerancia=sum(1 for e in erros if e <= TOLERANCIA_ALVO) / len(erros),
        dentro_do_limite=sum(1 for e in erros if e <= TOLERANCIA_LIMITE) / len(erros),
        fontes=origem,
        concordancia=concordancia,
    )


ACERTO = "acerto"
"""Erro dentro da tolerancia. Nao e causa de nada: entra na tabela so para o resto ter
denominador."""

ESCOLHA = "escolha"
"""A quebra certa existia na lista e perdeu no score.

E o unico tipo em que reordenar candidatos resolve. Quantos clipes caem aqui e o teto de qualquer
refinamento que mexa so no criterio de escolha, e medir esse teto antes de escrever o criterio e o
que separa refinamento de tentativa.
"""

TRAJETORIA = "trajetoria"
"""Nenhuma das quebras registradas caiu perto do gabarito.

Aqui reordenar nao muda nada: a quebra do contato nao chegou a existir, seja porque a bola nao foi
detectada naquele trecho, seja porque a trajetoria interpolada passou reto por cima da virada. O
trabalho e na deteccao da bola, e nao no instante.
"""

ESTIMATIVA = "estimado"
"""O instante e o ponto medio, porque nenhuma quebra passou de `MIN_SCORE`.

Separado de `trajetoria` de proposito: aqui pode haver candidato bom abaixo do limiar, e o ajuste
e de limiar. La nao ha candidato bom nenhum.
"""

NAO_DECLARADO = "nao-declarado"
"""A execucao nao registrou candidatos para este clipe, entao a causa nao e conhecida.

Execucao anterior a este campo cai toda aqui, e e por isso que `cobertura` existe: uma taxonomia
medida sobre execucao velha diria `trajetoria` para tudo se a ausencia fosse lida como lista vazia.
"""


@dataclass(frozen=True)
class Diagnostico:
    """O erro de um clipe com a causa nomeada, e o que a lista de candidatos sabia sobre ele."""

    clip_id: str
    erro: int
    tipo: str
    melhor_rank: int | None
    """Posicao do primeiro candidato dentro da tolerancia, contada a partir de 1, ou `None`."""
    melhor_erro: int | None
    """Erro do candidato mais proximo do gabarito, mesmo que nenhum esteja dentro da tolerancia."""


def _quadro(documento: Mapping[str, object]) -> int | None:
    valor = documento.get("frame_index")
    if isinstance(valor, bool) or not isinstance(valor, int | float):
        return None
    return int(valor)


def diagnose(
    previsto: Mapping[str, int],
    gabarito: Mapping[str, int],
    *,
    fontes: Mapping[str, str] | None = None,
    candidatos: Mapping[str, Sequence[Mapping[str, object]]] | None = None,
    tolerancia: int = TOLERANCIA_ALVO,
) -> tuple[Diagnostico, ...]:
    """A causa do erro de cada clipe, lida da lista de quebras que a execucao registrou.

    Sem isto o relatorio do instante para na mediana, e mediana nao diz o que fazer: 60 quadros de
    erro pedem trabalho diferente se a quebra certa estava em segundo lugar ou se ela nunca
    existiu. A taxonomia separa exatamente essas duas populacoes.

    Clipe sem candidato registrado sai `nao-declarado` em vez de entrar em `trajetoria`, porque
    ausencia de registro nao e ausencia de quebra.
    """
    comuns = sorted(set(previsto) & set(gabarito))
    achados: list[Diagnostico] = []
    for clip in comuns:
        verdade = gabarito[clip]
        erro = abs(previsto[clip] - verdade)
        lista = list((candidatos or {}).get(clip) or ())
        medidos = [
            (posicao, abs(quadro - verdade))
            for posicao, documento in enumerate(lista, start=1)
            if (quadro := _quadro(documento)) is not None
        ]
        dentro = [posicao for posicao, distancia in medidos if distancia <= tolerancia]
        achados.append(
            Diagnostico(
                clip_id=clip,
                erro=erro,
                tipo=_tipo(erro, (fontes or {}).get(clip), medidos, dentro, tolerancia),
                melhor_rank=dentro[0] if dentro else None,
                melhor_erro=min(d for _, d in medidos) if medidos else None,
            )
        )
    return tuple(achados)


def _tipo(
    erro: int,
    fonte: str | None,
    medidos: Sequence[tuple[int, int]],
    dentro: Sequence[int],
    tolerancia: int,
) -> str:
    if erro <= tolerancia:
        return ACERTO
    if fonte == ESTIMADO:
        return ESTIMATIVA
    if not medidos:
        return NAO_DECLARADO
    return ESCOLHA if dentro else TRAJETORIA


@dataclass(frozen=True)
class TaxonomyReport:
    n: int
    erros: int
    tipos: Mapping[str, int]
    cobertura: float
    recall: Mapping[str, float]
    acaso: Mapping[str, float]
    n_com_conjunto: int

    def to_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "erros": self.erros,
            "tipos": dict(self.tipos),
            "cobertura_das_causas": self.cobertura,
            "recall_em_k": dict(self.recall),
            "recall_ao_acaso": dict(self.acaso),
            "clipes_com_conjunto_de_quebras": self.n_com_conjunto,
        }


def _acaso_do_clipe(quadros: Sequence[int], verdade: int, k: int, tolerancia: int) -> float:
    """A chance de `k` quebras sorteadas cairem dentro da tolerancia, sorteadas sem reposicao.

    Exata, e nao simulada: sao `n` quebras das quais `dentro` servem, entao a chance de nenhuma
    das `k` servir e `C(n - dentro, k) / C(n, k)`. Simular daria o mesmo numero com ruido, e uma
    semente para alguem discutir.
    """
    n = len(quadros)
    if n == 0 or k <= 0:
        return 0.0
    dentro = sum(1 for quadro in quadros if abs(quadro - verdade) <= tolerancia)
    if dentro == 0:
        return 0.0
    if k >= n:
        return 1.0
    return 1.0 - comb(n - dentro, k) / comb(n, k)


def taxonomy(
    diagnosticos: Sequence[Diagnostico],
    *,
    profundidade: int = 5,
    gabarito: Mapping[str, int] | None = None,
    pools: Mapping[str, Sequence[int]] | None = None,
    tolerancia: int = TOLERANCIA_ALVO,
) -> TaxonomyReport:
    """Agrega os diagnosticos, mede o teto de reordenar e poe o acaso ao lado dele.

    `recall_em_k` e a fracao de clipes em que algum dos `k` primeiros candidatos esta dentro da
    tolerancia. Em `k = 1` ele e o proprio acerto do modelo, com uma excecao: clipe que caiu no
    ponto medio tem candidato de primeira posicao que foi rejeitado pelo limiar, entao ele pode
    contar aqui sem ter contado na medida.

    `recall_ao_acaso` e o mesmo numero para `k` quebras sorteadas do conjunto inteiro daquele
    clipe, e sem ele o teto engana. Com 54 quebras por clipe na mediana e tolerancia de 30 quadros,
    sortear quatro ja cai dentro em mais da metade dos clipes. O que um criterio de escolha tem
    para ganhar e a diferenca entre os dois, e nao o `recall_em_k` sozinho.

    `cobertura` e a fracao dos erros com causa nomeada, e e ela que responde ao criterio de
    classificar ao menos 80% deles. Acerto nao entra no denominador porque acerto nao tem causa
    para nomear.
    """
    contagem: dict[str, int] = {}
    for diagnostico in diagnosticos:
        contagem[diagnostico.tipo] = contagem.get(diagnostico.tipo, 0) + 1
    erros = sum(quantos for tipo, quantos in contagem.items() if tipo != ACERTO)
    nomeados = erros - contagem.get(NAO_DECLARADO, 0)
    recall = {
        f"top{k}": sum(1 for d in diagnosticos if d.melhor_rank is not None and d.melhor_rank <= k)
        / len(diagnosticos)
        for k in range(1, profundidade + 1)
    }
    com_conjunto = [
        (d.clip_id, tuple((pools or {}).get(d.clip_id) or ()))
        for d in diagnosticos
        if (pools or {}).get(d.clip_id) and d.clip_id in (gabarito or {})
    ]
    acaso = (
        {
            f"top{k}": sum(
                _acaso_do_clipe(quadros, (gabarito or {})[clip], k, tolerancia)
                for clip, quadros in com_conjunto
            )
            / len(com_conjunto)
            for k in range(1, profundidade + 1)
        }
        if com_conjunto
        else {}
    )
    return TaxonomyReport(
        n=len(diagnosticos),
        erros=erros,
        tipos=dict(sorted(contagem.items())),
        cobertura=1.0 if erros == 0 else nomeados / erros,
        recall=recall if diagnosticos else {},
        acaso=acaso,
        n_com_conjunto=len(com_conjunto),
    )


SPOTS_FILE = "spots.jsonl"
"""Instante por clipe produzido por um detector de fora deste repositorio.

Existe para medir alternativa externa sem trazer o codigo dela para dentro. O T-DEED, que e a
alternativa em avaliacao, e GPL-3.0 e o `src/` deste projeto e CC-BY-4.0: importa-lo obrigaria o
produto inteiro a mudar de licenca para responder a uma pergunta de avaliacao. O detector roda
fora, escreve `clip_id` e `frame_index`, e a medida acontece aqui, com os mesmos nulos.
"""


def read_spots(caminho: Path) -> dict[str, int]:
    """Um instante por clipe, recusando duplicata em vez de deixar a ultima linha vencer.

    Duplicata silenciosa e como um detector com duas saidas para o mesmo clipe passa por um
    detector com uma, e a medida sai sobre um conjunto que ninguem escolheu.
    """
    if not caminho.is_file():
        raise InstantError(f"{caminho} nao existe")
    spots: dict[str, int] = {}
    for numero, linha in enumerate(caminho.read_text(encoding="utf-8").splitlines(), start=1):
        if not linha.strip():
            continue
        try:
            documento = json.loads(linha)
            clip = str(documento["clip_id"])
            quadro = documento["frame_index"]
        except (json.JSONDecodeError, KeyError, TypeError) as erro:
            raise InstantError(f"{caminho}:{numero} nao e um spot valido: {erro}") from erro
        if isinstance(quadro, bool) or not isinstance(quadro, int | float):
            raise InstantError(f"{caminho}:{numero}: frame_index precisa ser numero")
        if clip in spots:
            raise InstantError(f"{caminho}:{numero}: {clip} aparece duas vezes")
        spots[clip] = int(quadro)
    if not spots:
        raise InstantError(f"{caminho} nao tem nenhum spot")
    return spots


@dataclass(frozen=True)
class BaselineReport:
    n: int
    acaso: float
    ponto_medio: float

    def to_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "quadro_ao_acaso": self.acaso,
            "ponto_medio_fixo": self.ponto_medio,
        }


def baselines(
    gabarito: Mapping[str, int],
    comprimentos: Mapping[str, int],
    *,
    tolerancia: int = TOLERANCIA_ALVO,
) -> BaselineReport:
    """As duas linhas de base que toda taxa de instante precisa ter ao lado.

    `quadro_ao_acaso` e exato e nao simulado: e a fracao dos quadros do clipe que caem dentro da
    tolerancia do gabarito, que e a chance de um quadro sorteado acertar. Com tolerancia de 30
    quadros num clipe de 420 ele ja da 0,145, e uma taxa publicada sem esse numero ao lado parece
    melhor do que e.

    `ponto_medio_fixo` e o preditor constante, que e a linha de base de maioria deste problema: os
    cortes do parceiro contem a finalizacao, entao chutar o meio do clipe nao e absurdo e precisa
    ser batido explicitamente.
    """
    comuns = sorted(set(gabarito) & set(comprimentos))
    if not comuns:
        return BaselineReport(0, 0.0, 0.0)
    acaso = 0.0
    meio = 0
    for clip in comuns:
        total = comprimentos[clip]
        verdade = gabarito[clip]
        dentro = min(total - 1, verdade + tolerancia) - max(0, verdade - tolerancia) + 1
        acaso += max(0, dentro) / total if total else 0.0
        meio += 1 if abs(total // 2 - verdade) <= tolerancia else 0
    return BaselineReport(len(comuns), acaso / len(comuns), meio / len(comuns))


def wilson(acertos: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """O intervalo da taxa observada, porque uma fracao sobre 40 clipes nao e um numero exato."""
    if n <= 0:
        return (0.0, 1.0)
    p = acertos / n
    denominador = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / denominador
    meia = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denominador
    return (max(0.0, centro - meia), min(1.0, centro + meia))


@dataclass(frozen=True)
class Resolucao:
    """O menor efeito que esta amostra consegue distinguir de ruido."""

    n: int
    clipes: float
    fracao: float
    discordantes_para_sinal: int

    def to_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "menor_efeito_visivel_em_clipes": round(self.clipes, 1),
            "menor_efeito_visivel_em_fracao": round(self.fracao, 3),
            "pares_discordantes_para_p_005": self.discordantes_para_sinal,
        }


DISCORDANTES_PARA_SINAL = 5
"""Pares discordantes, todos do mesmo lado, para o teste de sinal pareado dar p < 0,05.

Sao `0,5 ** k`, entao quatro dao 0,0625 e cinco dao 0,0312. Abaixo disso a comparacao pareada nao
sustenta conclusao, por mais que a taxa tenha subido.
"""


def resolution(acertos: int, n: int) -> Resolucao:
    """Quanta melhoria este conjunto consegue enxergar, ao lado da taxa que ele mede.

    Existe pelo mesmo motivo dos nulos, e por uma experiencia desta sprint: varias mudancas foram
    medidas, cada uma moveu um ou dois clipes, e cada uma pareceu ganho. Com 24 clipes a meia
    largura do intervalo de Wilson e de quatro clipes, entao mudanca de um ou dois esta dentro do
    ruido por construcao e nenhuma quantidade de cuidado no experimento resolve isso.

    Publicar a resolucao junto da taxa transforma "melhorou de 0,292 para 0,333" em "moveu um clipe
    num conjunto que so enxerga quatro", que e a mesma frase dita de forma honesta.
    """
    if n <= 0:
        return Resolucao(0, 0.0, 0.0, DISCORDANTES_PARA_SINAL)
    baixo, alto = wilson(acertos, n)
    meia = (alto - baixo) / 2
    return Resolucao(n, meia * n, meia, DISCORDANTES_PARA_SINAL)


def sample(clipes: Sequence[tuple[str, str]], quantos: int, seed: int) -> tuple[str, ...]:
    """Amostra estratificada por partida, com semente fixa, para o gabarito nao virar de uma so.

    Sem estratificar, o sorteio concentra em partidas com mais cortes: uma delas tem 38 clipes e
    outra tem 12, e gramado, camera e iluminacao mudam entre elas.
    """
    import random

    por_partida: dict[str, list[str]] = {}
    for clip_id, match_id in sorted(clipes):
        por_partida.setdefault(match_id, []).append(clip_id)
    sorteador = random.Random(seed)
    for lista in por_partida.values():
        sorteador.shuffle(lista)

    escolhidos: list[str] = []
    rodada = 0
    while len(escolhidos) < quantos:
        disponivel = [lista[rodada] for lista in por_partida.values() if len(lista) > rodada]
        if not disponivel:
            break
        escolhidos.extend(disponivel[: quantos - len(escolhidos)])
        rodada += 1
    return tuple(escolhidos)
