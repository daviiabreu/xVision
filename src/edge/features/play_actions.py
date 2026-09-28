"""As acoes com bola que o T-DEED pontuou antes do chute, por classe, sem decidir classe nenhuma."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from contracts import Perception
from edge.features.base import ShotOutput, shot_actions
from edge.features.play_evidence import contact_time_s

JANELA_ACOES_S = 6.0
"""Quanto antes do contato a leitura alcanca, em segundos. Preliminar."""

MARGEM_DO_CHUTE_S = 0.2
"""O trecho colado no contato fica de fora, porque ali o pico e do proprio chute. Preliminar."""

LIMIAR_DE_ACAO = 0.5
"""Probabilidade a partir da qual um passo conta como acao vista. Preliminar."""

FORA_DA_LEITURA = frozenset({"BACKGROUND", "SHOT", "GOAL"})


@dataclass(frozen=True)
class ActionPeak:
    probability: float
    lead_s: float
    """Quanto antes do contato o pico aconteceu."""


@dataclass(frozen=True)
class ActionEvidence:
    peaks: Mapping[str, ActionPeak]
    last_action: str | None
    last_action_lead_s: float | None


def measure_actions(perception: Perception, shot: ShotOutput) -> ActionEvidence | None:
    """O pico de cada classe na janela e a ultima acao acima do limiar, ou `None` sem T-DEED."""
    acoes = shot_actions(shot)
    t_contato = contact_time_s(perception, shot.frame_index)
    if acoes is None or t_contato is None:
        return None
    passos: list[tuple[float, tuple[float, ...]]] = []
    for quadro, linha in zip(acoes.frame_indices, acoes.probabilities, strict=True):
        instante = contact_time_s(perception, quadro)
        if instante is not None and MARGEM_DO_CHUTE_S <= t_contato - instante <= JANELA_ACOES_S:
            passos.append((t_contato - instante, linha))
    lidas = [(coluna, c) for coluna, c in enumerate(acoes.classes) if c not in FORA_DA_LEITURA]
    picos: dict[str, ActionPeak] = {}
    if passos:
        for coluna, classe in lidas:
            antecedencia, linha = max(passos, key=lambda passo: (passo[1][coluna], -passo[0]))
            picos[classe] = ActionPeak(round(linha[coluna], 4), round(antecedencia, 4))
    for antecedencia, linha in sorted(passos, key=lambda passo: passo[0]):
        vistas = [
            (linha[coluna], classe) for coluna, classe in lidas if linha[coluna] >= LIMIAR_DE_ACAO
        ]
        if vistas:
            return ActionEvidence(picos, max(vistas)[1], round(antecedencia, 4))
    return ActionEvidence(picos, None, None)
