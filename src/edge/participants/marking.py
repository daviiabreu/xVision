"""Quem marca quem num quadro, a partir de posicao e equipe."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import dist
from typing import Any, cast

from edge.participants.identity import Team
from edge.participants.types import FramePerception, as_role, ground_point

Projection = Callable[[tuple[float, float]], tuple[float, float]]

DEFAULT_RADIUS_M = 5.0
"""Raio maximo de marcacao, em metros de campo. Escolha declarada, nao calibrada."""


class MarkingError(ValueError):
    """Entrada que nao descreve um quadro possivel."""


@dataclass(frozen=True)
class MarkingCandidate:
    """Um jogador do quadro. `key` e o `track_id`, unico no quadro."""

    key: int
    team: Team
    position: tuple[float, float]


@dataclass(frozen=True)
class MarkingPair:
    """Dois adversarios que se marcam, sem direcao. `distance` na unidade das posicoes."""

    team_a_key: int
    team_b_key: int
    distance: float


def pair_markings(
    candidates: Sequence[MarkingCandidate],
    *,
    radius: float = DEFAULT_RADIUS_M,
) -> tuple[MarkingPair, ...]:
    """Empareja adversarios que sao o mais proximo um do outro dentro do raio.

    A reciprocidade garante no maximo um par por jogador. Sai ordenado por `team_a_key`.
    """
    if radius <= 0:
        raise MarkingError(f"radius {radius!r} precisa ser positivo")
    _refuse_duplicate_keys(candidates)

    team_a = _ordered(candidates, "team_a")
    team_b = _ordered(candidates, "team_b")
    if not team_a or not team_b:
        return ()

    nearest_in_b = {candidate.key: _nearest(candidate, team_b) for candidate in team_a}
    nearest_in_a = {candidate.key: _nearest(candidate, team_a) for candidate in team_b}

    pairs: list[MarkingPair] = []
    for candidate in team_a:
        opponent, distance = nearest_in_b[candidate.key]
        if distance > radius:
            continue
        back, _ = nearest_in_a[opponent.key]
        if back.key != candidate.key:
            continue
        pairs.append(
            MarkingPair(
                team_a_key=candidate.key,
                team_b_key=opponent.key,
                distance=distance,
            )
        )
    return tuple(pairs)


def marked_keys(pairs: Iterable[MarkingPair]) -> frozenset[int]:
    """Todos os jogadores que aparecem em algum par."""
    return frozenset(key for pair in pairs for key in (pair.team_a_key, pair.team_b_key))


def teams_from_identities(document: Mapping[str, Any]) -> dict[int, Team]:
    """Le o `identities.json`: track_id -> equipe, so o que saiu com status `available`."""
    equipes: dict[int, Team] = {}
    for bruto in cast("list[Mapping[str, Any]]", document.get("tracks", ())):
        if bruto.get("status") != "available":
            continue
        team = bruto.get("team")
        if team not in ("team_a", "team_b"):
            continue
        equipes[int(bruto["track_id"])] = team
    return equipes


def candidates_from_frame(
    frame: FramePerception,
    teams: Mapping[int, Team],
    *,
    project: Projection | None = None,
) -> tuple[MarkingCandidate, ...]:
    """Monta os candidatos do quadro: so `player`, com `track_id` e com equipe resolvida.

    A posicao e o `ground_point`, e `project` a converte para metros quando ha homografia.
    """
    candidatos: list[MarkingCandidate] = []
    for detection in frame.players:
        if as_role(detection.role) != "player":
            continue
        track_id = detection.track_id
        if track_id is None:
            continue
        team = teams.get(track_id)
        if team is None:
            continue
        ponto = ground_point(detection.box)
        candidatos.append(
            MarkingCandidate(
                key=track_id,
                team=team,
                position=project(ponto) if project else ponto,
            )
        )
    return tuple(candidatos)


def _refuse_duplicate_keys(candidates: Sequence[MarkingCandidate]) -> None:
    vistas: set[int] = set()
    for candidate in candidates:
        if candidate.key in vistas:
            raise MarkingError(f"a chave {candidate.key} aparece duas vezes no mesmo quadro")
        vistas.add(candidate.key)


def _ordered(
    candidates: Sequence[MarkingCandidate],
    team: Team,
) -> tuple[MarkingCandidate, ...]:
    return tuple(sorted((c for c in candidates if c.team == team), key=lambda c: c.key))


def _nearest(
    origin: MarkingCandidate,
    targets: tuple[MarkingCandidate, ...],
) -> tuple[MarkingCandidate, float]:
    """O alvo mais proximo, desempatando pela menor chave para nao depender da ordem das caixas."""
    best = targets[0]
    best_distance = dist(origin.position, best.position)
    for target in targets[1:]:
        distance = dist(origin.position, target.position)
        if distance < best_distance or (distance == best_distance and target.key < best.key):
            best, best_distance = target, distance
    return best, best_distance


__all__ = [
    "DEFAULT_RADIUS_M",
    "MarkingCandidate",
    "MarkingError",
    "MarkingPair",
    "Projection",
    "candidates_from_frame",
    "marked_keys",
    "pair_markings",
    "teams_from_identities",
]
