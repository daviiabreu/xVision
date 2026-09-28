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
from dataclasses import replace
from pathlib import Path

from contracts import FeatureValue, FramePerception, Perception
from edge.features.base import FeatureSpec, ShotOutput, shot_provenance, unavailable
from edge.features.evidence_model import EvidenceModel
from edge.features.play_actions import ActionEvidence, measure_actions
from edge.features.play_evidence import PlayEvidence, PlayTelemetry, measure
from edge.features.play_restart import RestartEvidence, measure_restart

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

ASSOCIADO = "associated"
INDISPONIVEL = "unavailable"
"""Como a evidencia declara toque anterior presente e ausente, sem expor `track_id`.

O identificador do jogador nao entra na evidencia publicada: ela sai no `ShotEvent` e conteudo
restrito do parceiro nao atravessa essa fronteira. O que o consumidor precisa e se houve toque e se
ele foi do finalizador, e `toque_anterior_do_finalizador` responde a segunda parte.
"""

PONTO_MEDIO = "midpoint"
"""Procedencia de instante que esta feature recusa, porque ela le o antes e ali nao ha antes.

O caminho `midpoint` do detector de instante nao mede contato: ele devolve o quadro do meio do
clipe quando a trajetoria nao sustenta quebra nenhuma. A janela de 90 quadros antes desse quadro e
um trecho arbitrario do clipe, e a regra que le velocidade e imobilidade ali produz classe com a
mesma cara de uma classe medida. Recusar declara a ausencia; classificar a esconde.

Instante que nao declara procedencia nao entra aqui. Ler ausencia como `midpoint` recusaria clipe
por falta de declaracao, e todos os dubles de teste do projeto nao declaram.
"""


class TypeOfPlay:
    spec = FeatureSpec(
        name="type_of_play",
        version="0.3.0",
        requires=("ball",),
        unit="classe",
    )

    def __init__(self, model: str | None = None) -> None:
        """Com `model`, o artefato do `xvision treino` decide a classe; sem ele, decide a regra."""
        self._modelo = None if model is None else EvidenceModel.load(Path(model))
        if self._modelo is not None:
            if self._modelo.feature != self.spec.name:
                raise ValueError(f"{model} foi treinado para {self._modelo.feature!r}")
            self.spec = replace(TypeOfPlay.spec, version=f"{TypeOfPlay.spec.version}+modelo")

    def extract(self, perception: Perception, shot: ShotOutput) -> FeatureValue:
        procedencia = shot_provenance(shot)
        if procedencia.source == PONTO_MEDIO:
            return unavailable(
                self.spec,
                f"o instante veio de {PONTO_MEDIO!r}, entao a janela esta ancorada em estimativa "
                "e nao existe um antes do contato para ler",
            )

        antes = _antes_do_contato(perception, shot.frame_index)
        if len(antes) < MIN_OBSERVACOES:
            return unavailable(
                self.spec,
                f"a especie ball tem menos de {MIN_OBSERVACOES} amostras nos "
                f"{JANELA_QUADROS} quadros anteriores ao contato",
            )

        velocidades = _velocidades(antes)
        recentes = velocidades[-2:]
        if not recentes:  # pragma: no cover - MIN_OBSERVACOES ja garante duas velocidades
            # Inalcancavel com MIN_OBSERVACOES em 3, e mantido de proposito: baixar essa constante
            # para 1 tornaria o caminho vivo, e sem esta guarda ele estouraria em max(()).
            return unavailable(self.spec, "a trajetoria da bola nao tem dois pontos para comparar")
        imoveis = _imoveis_no_fim(antes, velocidades)
        chegada = max(recentes)

        if imoveis >= PARADA_MIN_QUADROS:
            valor, motivo = PARADA, f"bola imovel por {imoveis} quadros antes do contato"
        elif chegada < CONDUCAO_PX_POR_QUADRO:
            valor, motivo = INDIVIDUAL, f"bola chegou a {chegada:.1f} px por quadro"
        else:
            valor, motivo = ABERTA, f"bola chegou a {chegada:.1f} px por quadro"

        evidencia, contagem = measure(perception, shot, procedencia)
        publicada: dict[str, object] = {
            "quadros_lidos": len(antes),
            "quadros_imoveis": imoveis,
            "velocidade_de_chegada_px_por_quadro": round(chegada, 2),
            "leitura": motivo,
            **_em_unidades(evidencia, contagem),
            **_acoes(measure_actions(perception, shot)),
            **_reinicio(measure_restart(perception, shot)),
        }
        confianca = None
        if self._modelo is not None:
            probabilidades = self._modelo.probabilities([(valor, publicada)])[0]
            publicada["classe_da_regra"] = valor
            publicada["probabilidades"] = {
                c: round(float(p), 4)
                for c, p in zip(self._modelo.classes, probabilidades, strict=True)
            }
            valor = self._modelo.classes[int(probabilidades.argmax())]
            confianca = round(float(probabilidades.max()), 4)
        return FeatureValue(
            name=self.spec.name,
            version=self.spec.version,
            status="ok",
            value=valor,
            confidence=confianca,
            reason=None,
            evidence=publicada,
        )


def _acoes(acoes: ActionEvidence | None) -> dict[str, object]:
    """As acoes do T-DEED antes do chute; tudo `None` quando o instante nao veio dele."""
    if acoes is None:
        return {
            "acoes_antes_do_chute": None,
            "ultima_acao": None,
            "ultima_acao_antecedencia_s": None,
        }
    return {
        "acoes_antes_do_chute": {
            classe: {"probabilidade": pico.probability, "antecedencia_s": pico.lead_s}
            for classe, pico in acoes.peaks.items()
        },
        "ultima_acao": acoes.last_action,
        "ultima_acao_antecedencia_s": acoes.last_action_lead_s,
    }


def _reinicio(reinicio: RestartEvidence) -> dict[str, object]:
    return {
        "reinicio_s": reinicio.restart_s,
        "reinicio_antecedencia_s": reinicio.restart_lead_s,
        "plano_fechado_antecedencia_s": reinicio.close_up_lead_s,
        "ultimo_corte_antecedencia_s": reinicio.cut_lead_s,
    }


def _em_unidades(evidencia: PlayEvidence, contagem: PlayTelemetry) -> dict[str, object]:
    """A mesma jogada em segundos e alturas de jogador, ao lado das chaves em pixel por quadro.

    Ao lado, e nao no lugar. Trocar unidade e trocar limiar na mesma rodada tornaria impossivel
    separar o efeito de um do efeito do outro, e a spec 04 pede a reproducao antes da mudanca. As
    quatro chaves antigas continuam identicas e um golden de sete percepcoes prende isso.

    `None` atravessa como `None`: `_freeze` do contrato aceita nulo, e virar zero aqui faria recusa
    de medida chegar ao consumidor com a mesma cara de medida que deu zero.

    Nada de `track_id`, coordenada ou identificador entra. A evidencia sai no evento publicado, e
    conteudo restrito do parceiro nao atravessa essa fronteira.
    """
    return {
        "imobilidade_s": _arredonda(evidencia.stationary_s),
        "caminho_alturas": _arredonda(evidencia.path_player_heights),
        "velocidade_de_chegada_alturas_por_s": _arredonda(evidencia.arrival_speed_player_heights_s),
        "fracao_observada": round(evidencia.observed_fraction, 4),
        "toque_anterior": ASSOCIADO if evidencia.previous_touch_source else INDISPONIVEL,
        "toque_anterior_do_finalizador": evidencia.previous_touch_is_shooter,
        "toque_anterior_relacao": evidencia.previous_touch_relation,
        "idade_do_toque_anterior_s": _arredonda(evidencia.previous_touch_age_s),
        "corte_de_camera": evidencia.camera_cut,
        "procedencia_do_instante": evidencia.shot_source,
        "apoio_da_bola": evidencia.shot_ball_support,
        "escala_alturas_px": _arredonda(contagem.median_player_height_px),
        "quadros_na_janela": contagem.frames_in_window,
        "bola_detectada": contagem.ball_detected,
        "episodios_de_posse": contagem.touch.episodios,
    }


def _arredonda(valor: float | None) -> float | None:
    """Quatro casas nos campos com unidade, preservando ausencia como ausencia.

    A conta em ponto flutuante deixa lixo visivel no artefato publicado, como 2,997000000000001 para
    uma velocidade de 2,997, e ele varia entre execucoes sem que nada tenha mudado, o que suja
    qualquer diff de evento. Quatro casas sao mais resolucao do que qualquer limiar desta feature
    usa. `None` atravessa: `round(None)` estouraria e zero seria medida onde nao houve.
    """
    return None if valor is None else round(valor, 4)


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
