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
from pathlib import Path

from edge.paths import var_dir

ARQUIVO = "instants.jsonl"
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


def wilson(acertos: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """O intervalo da taxa observada, porque uma fracao sobre 40 clipes nao e um numero exato."""
    if n <= 0:
        return (0.0, 1.0)
    p = acertos / n
    denominador = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / denominador
    meia = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denominador
    return (max(0.0, centro - meia), min(1.0, centro + meia))


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
