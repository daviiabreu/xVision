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


class MarkingError(ValueError):
    """Entrada que nao descreve um quadro possivel."""


@dataclass(frozen=True)
class MarkingCandidate:
    key: int
    team: Team
    position: tuple[float, float]


@dataclass(frozen=True)
class MarkingPair:
    team_a_key: int
    team_b_key: int
    distance: float


def pair_markings(
    candidates: Sequence[MarkingCandidate], *, radius: float = DEFAULT_RADIUS_M
) -> tuple[MarkingPair, ...]:
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
        if back.key == candidate.key:
            pairs.append(MarkingPair(candidate.key, opponent.key, distance))
    return tuple(pairs)


def marked_keys(pairs: Iterable[MarkingPair]) -> frozenset[int]:
    return frozenset(key for pair in pairs for key in (pair.team_a_key, pair.team_b_key))


def teams_from_identities(document: Mapping[str, Any]) -> dict[int, Team]:
    teams: dict[int, Team] = {}
    for raw in cast("list[Mapping[str, Any]]", document.get("tracks", ())):
        if raw.get("status") != "available":
            continue
        if raw.get("team") in ("team_a", "team_b"):
            teams[int(raw["track_id"])] = cast(Team, raw["team"])
    return teams


def candidates_from_frame(
    frame: FramePerception,
    teams: Mapping[int, Team],
    *,
    project: Projection | None = None,
) -> tuple[MarkingCandidate, ...]:
    candidates: list[MarkingCandidate] = []
    for detection in frame.players:
        if as_role(detection.role) != "player" or detection.track_id is None:
            continue
        team = teams.get(detection.track_id)
        if team is None:
            continue
        position = detection.pitch_xy
        if position is None:
            position = ground_point(detection.box)
        candidates.append(
            MarkingCandidate(detection.track_id, team, project(position) if project else position)
        )
    return tuple(candidates)


def _refuse_duplicate_keys(candidates: Sequence[MarkingCandidate]) -> None:
    keys: set[int] = set()
    for candidate in candidates:
        if candidate.key in keys:
            raise MarkingError(f"a chave {candidate.key} aparece duas vezes no mesmo quadro")
        keys.add(candidate.key)


def _ordered(candidates: Sequence[MarkingCandidate], team: Team) -> tuple[MarkingCandidate, ...]:
    return tuple(sorted((c for c in candidates if c.team == team), key=lambda c: c.key))


def _nearest(
    origin: MarkingCandidate, targets: tuple[MarkingCandidate, ...]
) -> tuple[MarkingCandidate, float]:
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
