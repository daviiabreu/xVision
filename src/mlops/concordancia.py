"""Concordância entre anotadores do instante do contato: a barra de erro do próprio gabarito.

Todo erro do modelo medido contra o gabarito soma dois erros que a acurácia sozinha não separa: o
do detector e o de quem anotou. A frente de validação independente existe para isolar o segundo, e
ela precisa saber quanto duas leituras honestas do mesmo clipe discordam entre si.

Este módulo não anota, não abre vídeo e não decide o instante. Ele recebe as `Anotacao` que
`mlops.instant` já grava e mede a distância entre anotadores. Reusa `latest`, então quem volta a um
clipe e corrige a própria marca não vira segunda opinião: correção não é discordância.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations

from mlops.instant import TOLERANCIA_ALVO, TOLERANCIA_LIMITE, Anotacao, latest


class ConcordanciaError(ValueError): ...


@dataclass(frozen=True)
class Par:
    """Um clipe lido por duas pessoas, o par que a concordância mede.

    `a` e `b` saem ordenados pelo nome do anotador, então o mesmo par não aparece nas duas ordens e
    dois relatórios do mesmo log ficam iguais quadro a quadro.
    """

    clip_id: str
    a: str
    b: str
    frame_a: int
    frame_b: int

    @property
    def delta(self) -> int:
        """Distância em quadros entre as duas leituras, sempre não negativa."""
        return abs(self.frame_a - self.frame_b)


def pares(anotacoes: Iterable[Anotacao]) -> tuple[Par, ...]:
    """Todos os pares de anotadores por clipe, a última leitura de cada pessoa.

    Clipe com uma anotação só não gera par. Clipe com três anotadores gera os três pares, porque a
    concordância é entre pessoas e não uma média que esconde quem destoou.
    """
    por_clipe: dict[str, list[Anotacao]] = {}
    for anotacao in latest(anotacoes):
        por_clipe.setdefault(anotacao.clip_id, []).append(anotacao)
    resultado: list[Par] = []
    for clip_id, leituras in por_clipe.items():
        ordenadas = sorted(leituras, key=lambda anotacao: anotacao.annotator)
        for primeira, segunda in combinations(ordenadas, 2):
            resultado.append(
                Par(
                    clip_id=clip_id,
                    a=primeira.annotator,
                    b=segunda.annotator,
                    frame_a=primeira.frame_index,
                    frame_b=segunda.frame_index,
                )
            )
    return tuple(resultado)


def _percentil(ordenado: Sequence[int], fracao: float) -> float:
    """O elemento na posição `fracao` de uma sequência já ordenada, sem interpolar.

    Índice discreto de propósito: com poucos pares não há resolução para interpolar, e o número
    medido é mais honesto que uma casa decimal inventada entre dois valores.
    """
    if not ordenado:
        return 0.0
    posicao = min(len(ordenado) - 1, int(fracao * (len(ordenado) - 1)))
    return float(ordenado[posicao])


@dataclass(frozen=True)
class ConcordanciaReport:
    n_clipes: int
    n_pares: int
    mediana_delta: float
    media_delta: float
    p90_delta: float
    max_delta: float
    dentro_da_tolerancia: float
    dentro_do_limite: float
    mediana_delta_s: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "n_clipes": self.n_clipes,
            "n_pares": self.n_pares,
            "tolerancia_quadros": TOLERANCIA_ALVO,
            "limite_quadros": TOLERANCIA_LIMITE,
            "mediana_delta_quadros": self.mediana_delta,
            "media_delta_quadros": self.media_delta,
            "p90_delta_quadros": self.p90_delta,
            "max_delta_quadros": self.max_delta,
            "dentro_da_tolerancia": self.dentro_da_tolerancia,
            "dentro_do_limite": self.dentro_do_limite,
            "mediana_delta_segundos": self.mediana_delta_s,
        }


def medir(anotacoes: Iterable[Anotacao], *, fps: float | None = None) -> ConcordanciaReport | None:
    """A concordância agregada entre anotadores, ou `None` quando nenhum clipe foi lido duas vezes.

    `None` e não um relatório de zeros: sem clipe repetido não há concordância para medir, e um zero
    aqui se confundiria com concordância perfeita, que é o oposto do que o dado diz.

    Com `fps`, a mediana também sai em segundos, que é a unidade em que a tolerância da janela foi
    escrita. Sem `fps`, o campo em segundos fica ausente em vez de assumir uma taxa que ninguém deu.
    """
    if fps is not None and fps <= 0.0:
        raise ConcordanciaError(f"fps precisa ser positivo, e veio {fps!r}")
    todos = pares(anotacoes)
    if not todos:
        return None
    deltas = sorted(par.delta for par in todos)
    n = len(deltas)
    clipes = {par.clip_id for par in todos}
    mediana = float(deltas[n // 2])
    return ConcordanciaReport(
        n_clipes=len(clipes),
        n_pares=n,
        mediana_delta=mediana,
        media_delta=sum(deltas) / n,
        p90_delta=_percentil(deltas, 0.90),
        max_delta=float(deltas[-1]),
        dentro_da_tolerancia=sum(1 for d in deltas if d <= TOLERANCIA_ALVO) / n,
        dentro_do_limite=sum(1 for d in deltas if d <= TOLERANCIA_LIMITE) / n,
        mediana_delta_s=None if fps is None else mediana / fps,
    )


def discordancias(
    anotacoes: Iterable[Anotacao], *, tolerancia: int = TOLERANCIA_ALVO
) -> tuple[Par, ...]:
    """Os pares acima da tolerância, do maior desacordo para o menor, para a revisão de piloto.

    É a lista de trabalho da validação independente: o clipe em que duas leituras honestas ficam a
    mais de `tolerancia` quadros é onde uma terceira decide, e ordenar pelo desacordo põe o pior no
    topo. A agregação de `medir` diz se o gabarito tem barra de erro; esta lista diz onde mexer.
    """
    if tolerancia < 0:
        raise ConcordanciaError(f"tolerancia é distância em quadros, e veio {tolerancia!r}")
    passaram = [par for par in pares(anotacoes) if par.delta > tolerancia]
    return tuple(sorted(passaram, key=lambda par: par.delta, reverse=True))
