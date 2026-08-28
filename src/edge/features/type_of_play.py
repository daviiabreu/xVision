"""Como a jogada chegou ao chute: bola parada, jogada aberta, ou lance individual.

A barra desta metrica nao e a maioria de 0,657. A coluna `origin_of_move` do parceiro, com o
melhor mapa possivel, ja acerta 485 de 609 sem tocar em video, e `config/labels.toml` declara
esse 0,796 como `baseline_metadata`. Comparar so contra a maioria publicaria como ganho o que
ninguem comprou.

O enum do parceiro junta dois eixos que a Opta mantem ortogonais: padrao de jogo, com os
qualificadores 22 a 26 para bola parada, e assistido contra individual, com os qualificadores 29 e
215. Medido na tabela cheia, `individual` implica `assist=false` em 717 de 722 casos preenchidos,
o que confirma o colapso. Isso limita o teto: nenhuma regra sobre a bola separa "individual" de
"jogada aberta sem assistencia", porque a diferenca esta na intencao do anotador e nao no video.
A seccao de riscos da spec registra isso, e a feature sai declaradamente parcial nesse ramo.

A arvore tem tres ramos e olha so a trajetoria da bola antes do contato:

- bola quase imovel por um trecho longo antes do contato, e e bola parada
- bola perto de imovel logo antes, sem ter vindo de longe, e o finalizador conduzia: individual
- bola chegando com velocidade de outro lugar, e alguem tocou para ele: jogada aberta

`requires` fica so em `ball` de proposito. Amarrar em `players` e `tracking` dobraria a superficie
de falha para refinar o ramo que a propria origem do rotulo ja nao deixa separar.

Nao decide quem tocou antes, nao mede assistencia e nao distingue falta de escanteio.
"""

from __future__ import annotations

from collections.abc import Sequence

from contracts import FeatureValue, FramePerception, Perception
from edge.features.base import FeatureSpec, ShotOutput, unavailable

ABERTA = "open play"
PARADA = "set piece"
INDIVIDUAL = "individual"

MIN_OBSERVACOES = 3
JANELA_QUADROS = 90
"""Quantos quadros antes do contato entram na leitura, que sao os 1,5 s da janela declarada."""

PARADA_PX_POR_QUADRO = 1.5
"""Acima disso a bola esta em movimento. Bola parada de verdade fica em zero mais ruido de caixa."""

PARADA_MIN_QUADROS = 24
"""Quanto tempo imovel caracteriza bola parada: 0,4 s a 59,94 fps, mais que um quique."""

CONDUCAO_PX_POR_QUADRO = 6.0
"""Piso de velocidade de um passe. Abaixo disso a bola vinha junto com quem chutou."""


class TypeOfPlay:
    spec = FeatureSpec(
        name="type_of_play",
        version="0.1.0",
        requires=("ball",),
        unit="classe",
    )

    def extract(self, perception: Perception, shot: ShotOutput) -> FeatureValue:
        antes = _antes_do_contato(perception, shot.frame_index)
        if len(antes) < MIN_OBSERVACOES:
            return unavailable(
                self.spec,
                f"a especie ball tem menos de {MIN_OBSERVACOES} amostras nos "
                f"{JANELA_QUADROS} quadros anteriores ao contato",
            )

        velocidades = _velocidades(antes)
        recentes = velocidades[-2:]
        if not recentes:
            return unavailable(self.spec, "a trajetoria da bola nao tem dois pontos para comparar")
        imoveis = _imoveis_no_fim(antes, velocidades)
        chegada = max(recentes)

        if imoveis >= PARADA_MIN_QUADROS:
            valor, motivo = PARADA, f"bola imovel por {imoveis} quadros antes do contato"
        elif chegada < CONDUCAO_PX_POR_QUADRO:
            valor, motivo = INDIVIDUAL, f"bola chegou a {chegada:.1f} px por quadro"
        else:
            valor, motivo = ABERTA, f"bola chegou a {chegada:.1f} px por quadro"

        return FeatureValue(
            name=self.spec.name,
            version=self.spec.version,
            status="ok",
            value=valor,
            confidence=None,
            reason=None,
            evidence={
                "quadros_lidos": len(antes),
                "quadros_imoveis": imoveis,
                "velocidade_de_chegada_px_por_quadro": round(chegada, 2),
                "leitura": motivo,
            },
        )


def _antes_do_contato(perception: Perception, contato: int) -> tuple[tuple[int, float, float], ...]:
    """Os centros de bola dos quadros da janela anterior, em ordem de quadro."""
    return tuple(
        (quadro.frame_index, *_centro(quadro))
        for quadro in perception.frames
        if quadro.ball is not None and contato - JANELA_QUADROS <= quadro.frame_index <= contato
    )


def _centro(quadro: FramePerception) -> tuple[float, float]:
    caixa = quadro.ball.box if quadro.ball is not None else None
    if caixa is None:  # pragma: no cover - o chamador ja garantiu que ha bola
        raise ValueError("quadro sem bola")
    return ((caixa.x1 + caixa.x2) / 2.0, (caixa.y1 + caixa.y2) / 2.0)


def _velocidades(pontos: Sequence[tuple[int, float, float]]) -> tuple[float, ...]:
    """Pixel por quadro, dividido pelo indice e nao pela posicao na lista.

    A especie `ball` roda a 5 fps e interpola lacunas, entao duas amostras vizinhas na lista podem
    estar a doze quadros de distancia: dividir pela posicao daria velocidade doze vezes maior.
    """
    return tuple(
        ((b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2) ** 0.5 / max(1, b[0] - a[0])
        for a, b in zip(pontos, pontos[1:], strict=False)
    )


def _imoveis_no_fim(
    pontos: Sequence[tuple[int, float, float]], velocidades: Sequence[float]
) -> int:
    """Quantos quadros a bola passou parada imediatamente antes do contato.

    Conta do fim para o comeco e para no primeiro trecho em movimento: uma bola que ficou parada,
    andou, e parou de novo nao e bola parada, e somar os dois trechos diria que sim.
    """
    total = 0
    for posicao in range(len(velocidades) - 1, -1, -1):
        if velocidades[posicao] > PARADA_PX_POR_QUADRO:
            break
        total += pontos[posicao + 1][0] - pontos[posicao][0]
    return total
