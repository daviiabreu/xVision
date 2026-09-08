"""Quanto a bola detectada no instante do contato e a bola do lance.

Existe porque `xvision eval` mede a feature no fim da cadeia e nao diz onde ela quebrou. Medido nos
38 clipes anotados: `body_part` fica em 0.323 tanto com o instante do detector quanto com o
instante anotado a mao, entao o instante nao e a causa. A causa esta antes, e este modulo a mede
direto.

O criterio nao e gosto. Numa finalizacao a bola esta em contato com o finalizador por definicao,
entao no quadro do contato a deteccao tem que cair sobre ou colada a uma caixa de jogador. Se cai a
120 px de todo mundo, ou acima da linha do campo onde so ha placa e arquibancada, ela nao e a bola
do lance, e o finalizador escolhido por proximidade dela e outra pessoa.

O piso e do gramado e nao do agrado: `y_maximo` sai do proprio quadro. Enquadramento de futebol na
transmissao poe a linha de fundo mais alta em cerca de 0,45 da altura, entao deteccao acima disso
esta no publico. E limiar grosseiro de proposito: refina-lo pediria homografia de campo, que o
projeto ainda nao tem, e o numero que interessa aqui nao muda por causa de cinco por cento.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from contracts import Box, FramePerception

ENCOSTADA_PX = 20.0
"""Folga em pixel que ainda conta como contato: raio da bola mais erro de caixa do detector."""

PERDIDA_PX = 50.0
"""Acima disto a deteccao nao descreve o lance, mesmo que haja bola em algum lugar do quadro."""

FORA_DO_GRAMADO_Y = 0.28
"""Abaixo desta fracao da altura a deteccao esta acima da linha do gramado, ou seja na arquibancada.

O valor anterior era 0,45 e vinha de estimativa a olho, com o sinal invertido na cabeca de quem o
escolheu: `y` menor e mais alto na imagem, entao arquibancada e `y` PEQUENO. Com 0,45 o campo
sinalizava quase toda deteccao, e oito dos nove clipes que ele apontava estavam sobre grama. Foi
essa contagem errada que fez tres frentes de diagnostico proporem um filtro de altura que, medido,
vale no maximo um clipe liquido.

O valor novo e a mediana do horizonte real, medida em 337 quadros do conjunto: mediana 0,285,
maximo 0,411. Ele separa os dois casos conhecidos, deixando passar 4997505 a y = 0,366, que e
acerto de 0 px sobre grama, e sinalizando 4996643 a y = 0,172, que esta na arquibancada.

Continua sendo aproximacao, e a docstring diz de que tipo: o horizonte e por clipe e depende do
enquadramento, entao uma deteccao a 0,25 esta sobre grama num clipe cujo horizonte e 0,21 e na
arquibancada num cujo horizonte e 0,35. Como o campo e diagnostico e nao filtro, errar nos dois
sentidos em torno da mediana e aceitavel; usa-lo para descartar candidato nao seria. Medir o
horizonte por clipe tornaria o campo exato e depende de segmentacao de campo, que o projeto nao tem.
"""

DETECTADO = "detected"
"""A origem que conta como percepcao. `interpolated` e reta tracada por cima da virada da bola."""

ACASO = 0.1307
"""A taxa que um detector que sorteasse um ponto do gramado alcancaria, medida e nao suposta.

Sem este numero a taxa observada nao tem escala. O quadro do contato tem 24 jogadores na mediana e
as caixas deles cobrem 5,4% da imagem, entao cair a 20 px de alguma nao e evento raro: 2000 pontos
uniformes na faixa de gramado de cada um dos 38 quadros anotados acertam 13,07% das vezes.

A linha de base de 0,2895 e portanto 2,2 vezes o acaso, num teto de 7,6. Ler 0,29 como "quase um
terco certo" seria erro de escala: o detector esta bem mais perto do sorteio do que do acerto.

Medido com src/edge/modules/ball em 26/08/2026, semente 20260825. Refazer se o detector de jogador
mudar, porque o acaso sobe com o numero e o tamanho das caixas.
"""


@dataclass(frozen=True)
class Contato:
    clip_id: str
    tem_bola: bool
    distancia_px: float | None
    y_relativo: float | None
    detectada: bool = True
    """Falso quando a posicao veio da interpolacao do rastreador em vez de um detector."""

    @property
    def encostada(self) -> bool:
        """Exige deteccao, e nao so posicao no lugar certo.

        Interpolacao acerta por acidente e nao por percepcao: ela e linear e o instante do chute e
        onde a trajetoria vira, entao a reta passa por cima da virada. Contar o acerto dela premia o
        rastreador por inventar e esconde o quanto o detector realmente ve. Medido: na linha de base
        de 38 clipes, cerca de metade dos 28 com bola era posicao inventada.
        """
        return (
            self.detectada and self.distancia_px is not None and self.distancia_px <= ENCOSTADA_PX
        )

    @property
    def encostada_incluindo_inventada(self) -> bool:
        """A taxa frouxa, publicada ao lado da estrita para a queda ficar visivel e explicada."""
        return self.distancia_px is not None and self.distancia_px <= ENCOSTADA_PX

    @property
    def perdida(self) -> bool:
        return self.distancia_px is None or self.distancia_px > PERDIDA_PX

    @property
    def fora_do_gramado(self) -> bool:
        return self.y_relativo is not None and self.y_relativo < FORA_DO_GRAMADO_Y


@dataclass(frozen=True)
class BallReport:
    n: int
    com_bola: int
    com_bola_detectada: int
    encostada: int
    encostada_incluindo_inventada: int
    perdida: int
    fora_do_gramado: int

    @property
    def taxa(self) -> float:
        """A fracao dos clipes anotados em que a bola do contato e utilizavel.

        Sobre `n` e nao sobre `com_bola`: clipe sem bola nenhuma no contato falha do mesmo jeito, e
        dividir pelos que tem bola premiaria um detector que aprendesse a nao detectar.
        """
        return self.encostada / self.n if self.n else 0.0

    @property
    def taxa_wilson_95(self) -> tuple[float, float]:
        return _wilson(self.encostada, self.n)

    def to_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "com_bola": self.com_bola,
            "com_bola_detectada": self.com_bola_detectada,
            "encostada": self.encostada,
            "encostada_incluindo_inventada": self.encostada_incluindo_inventada,
            "perdida": self.perdida,
            "fora_do_gramado": self.fora_do_gramado,
            "taxa_encostada": round(self.taxa, 4),
            "taxa_encostada_wilson_95": [round(value, 4) for value in self.taxa_wilson_95],
            "acaso": ACASO,
            "vezes_o_acaso": round(self.taxa / ACASO, 2),
            "limiar_encostada_px": ENCOSTADA_PX,
            "limiar_perdida_px": PERDIDA_PX,
        }


def distancia_ate_jogador(centro: tuple[float, float], caixas: Iterable[Box]) -> float | None:
    """Zero quando o centro cai dentro de alguma caixa, senao a distancia ate a borda."""
    distancias = [_ate_a_caixa(centro, caixa) for caixa in caixas]
    return min(distancias) if distancias else None


def _ate_a_caixa(centro: tuple[float, float], caixa: Box) -> float:
    dx = max(caixa.x1 - centro[0], 0.0, centro[0] - caixa.x2)
    dy = max(caixa.y1 - centro[1], 0.0, centro[1] - caixa.y2)
    return (dx * dx + dy * dy) ** 0.5


def contato_de(clip_id: str, quadro: FramePerception | None, altura: int) -> Contato:
    if quadro is None or quadro.ball is None:
        return Contato(clip_id, tem_bola=False, distancia_px=None, y_relativo=None)
    caixa = quadro.ball.box
    centro = ((caixa.x1 + caixa.x2) / 2.0, (caixa.y1 + caixa.y2) / 2.0)
    return Contato(
        clip_id,
        tem_bola=True,
        distancia_px=distancia_ate_jogador(centro, [j.box for j in quadro.players]),
        y_relativo=centro[1] / altura if altura else None,
        detectada=quadro.ball.source != "interpolated",
    )


def quadro_do_contato(quadros: Sequence[FramePerception], alvo: int) -> FramePerception | None:
    """O quadro amostrado mais proximo do contato anotado, e nada se a amostragem nao o alcanca.

    Devolver o mais proximo sem limite faria um clipe cuja amostra para 200 quadros antes contar
    como se tivesse bola no contato.
    """
    if not quadros:
        return None
    perto = min(quadros, key=lambda f: abs(f.frame_index - alvo))
    return perto if abs(perto.frame_index - alvo) <= 12 else None


def measure(contatos: Iterable[Contato]) -> BallReport:
    lista = list(contatos)
    return BallReport(
        n=len(lista),
        com_bola=sum(1 for c in lista if c.tem_bola),
        com_bola_detectada=sum(1 for c in lista if c.tem_bola and c.detectada),
        encostada=sum(1 for c in lista if c.encostada),
        encostada_incluindo_inventada=sum(1 for c in lista if c.encostada_incluindo_inventada),
        perdida=sum(1 for c in lista if c.perdida),
        fora_do_gramado=sum(1 for c in lista if c.fora_do_gramado),
    )


def compara(antes: Mapping[str, bool], depois: Mapping[str, bool]) -> dict[str, int]:
    """Quantos clipes cada lado ganhou, para uma mudanca nao esconder regressao atras da media."""
    comuns = set(antes) & set(depois)
    return {
        "ganhou": sum(1 for c in comuns if depois[c] and not antes[c]),
        "perdeu": sum(1 for c in comuns if antes[c] and not depois[c]),
        "n": len(comuns),
    }


def _wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))
