"""Uma inferencia por quadro, mesmo quando duas especies usam o mesmo peso.

O caso que criou este arquivo: um peso de futebol que detecta bola e gente na mesma passada, e que o
projeto registra em duas especies, `ball` e `players`, porque o contrato tem uma especie por
responsabilidade. Com as duas amostrando o mesmo conjunto de quadros, o mesmo modelo roda duas vezes
sobre exatamente os mesmos pixels. Medido no clipe mediano: 105 quadros viram 210 passadas, e a GPU
fica em 95% fazendo metade do trabalho duas vezes.

A configuracao de producao deixou de estar nesse caso e o arquivo continua valendo. Desde que `ball`
ganhou peso proprio, dedicado a bola, os dois pesos sao diferentes e a chave nunca colide, entao o
cache nao economiza nada e tambem nao atrapalha. Ele volta a economizar assim que duas especies
voltarem a compartilhar peso, o que e uma linha de configuracao.

A separacao das especies nao e desperdicio, e nao e ela que muda: quem consome `ball` nao quer
saber de jogador, e trocar o peso da bola sem tocar no de jogador continua sendo uma linha de
configuracao. O que muda e que a segunda chamada com a mesma entrada nao paga de novo.

A chave e o que identifica a computacao, e nao o nome de quem pediu: o caminho do peso, a impressao
digital do pre-processamento e o indice do quadro. Peso diferente, resolucao diferente ou quadro
diferente sao computacoes diferentes e nao se encontram no cache.

Nao e o `edge.cache.perception`, que guarda saida de modulo em disco entre execucoes. Este vive na
memoria e o `run_pipeline` o descarta no comeco de cada clipe, porque o que ele evita e repeticao
dentro de um clipe.

O descarte por clipe nao e higiene, e correcao. A chave nao tem identidade de clipe, e sem o
descarte o quadro 92 de um clipe serviria o quadro 92 do proximo: medido, seis clipes de duracoes
diferentes devolveram o mesmo instante 92, porque todos consumiram a deteccao do primeiro. Poe o
`clip_id` na chave e o descarte vira desnecessario, mas ai o `FrameBatch` precisa carregar o
identificador ate os modulos, e nenhum deles tem outro uso para ele.

Nao decide o que a deteccao significa. Ele devolve caixa, confianca e identificador de classe crus,
e quem interpreta e cada especie: para `ball` a classe da bola, para `players` as classes de gente.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from contracts import Box
from edge.modules.base import PreprocessSpec

Deteccao = tuple[Box, float, int]
"""Caixa em pixel do tensor, confianca, e o identificador de classe que o peso declarou."""


class SharedInference:
    """Memoriza a saida crua do detector por peso, pre-processamento e quadro.

    O tamanho nao tem teto porque o ciclo de vida e um clipe: sao poucas centenas de quadros, e
    quem cria a instancia a descarta junto com a execucao. Teto com despejo introduziria decisao de
    politica para um problema que nao existe nessa escala.
    """

    def __init__(self) -> None:
        self._memoria: dict[tuple[str, str, int], tuple[Deteccao, ...]] = {}
        self.hits = 0
        self.misses = 0

    def get(
        self,
        *,
        weights: str,
        preprocess: PreprocessSpec,
        frame_index: int,
        compute: Callable[[], Sequence[Deteccao]],
    ) -> tuple[Deteccao, ...]:
        chave = (weights, preprocess.fingerprint, frame_index)
        guardado = self._memoria.get(chave)
        if guardado is not None:
            self.hits += 1
            return guardado
        self.misses += 1
        calculado = tuple(compute())
        self._memoria[chave] = calculado
        return calculado

    def clear(self) -> None:
        self._memoria.clear()

    @property
    def size(self) -> int:
        return len(self._memoria)


_atual: SharedInference | None = None


def shared() -> SharedInference:
    """A instancia do processo, criada na primeira chamada.

    Global e nao parametro porque o caminho ate os dois modulos passa por `run_pipeline`,
    `_run_stage` e `_infer`, e nenhum deles tem motivo para conhecer um cache de inferencia. Passar
    por todos eles alargaria tres assinaturas para servir dois modulos.
    """
    global _atual
    if _atual is None:
        _atual = SharedInference()
    return _atual


def reset() -> None:
    """Descarta a memoria entre clipes, e e o teste que mais precisa disto.

    Sem o descarte, um teste que meca `hits` depende do que os testes anteriores rodaram, e a suite
    passa a depender de ordem.
    """
    global _atual
    _atual = None


def as_deteccoes(xyxy: Any, conf: Any, cls: Any) -> tuple[Deteccao, ...]:
    """Converte as tres listas paralelas que o Ultralytics devolve numa tupla de deteccoes."""
    return tuple(
        (Box(*(float(v) for v in caixa)), float(c), int(k))
        for caixa, c, k in zip(xyxy, conf, cls, strict=False)
    )
