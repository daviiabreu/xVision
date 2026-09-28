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


@dataclass(frozen=True)
class PressureRoster:
    """Quem pode entrar numa conta de pressao neste quadro, separado por ter posicao ou nao.

    A separacao existe porque as duas metades respondem perguntas diferentes: os `posicionados`
    decidem se ha marcador, e os `sem_posicao` decidem se um "nao achei" e uma negativa honesta ou
    apenas ignorancia.
    """

    posicionados: tuple[MarkingCandidate, ...]
    sem_posicao: tuple[tuple[int, Team], ...]


@dataclass(frozen=True)
class MarkingPressure:
    """Havia adversario no raio do jogador observado.

    `marcado` e `None` quando o quadro nao permite afirmar nem negar; `motivo` diz qual dos dois
    buracos causou isso.
    """

    marcado: bool | None
    marcadores: tuple[tuple[int, float], ...]
    adversarios_sem_posicao: int
    motivo: str | None = None


def pressure_roster(frame: FramePerception, teams: Mapping[int, Team]) -> PressureRoster:
    """Elenco de um quadro para a conta de pressao, so com quem tem equipe conhecida.

    Inclui goleiro, ao contrario de `candidates_from_frame`. Para um chute, o goleiro que sai do
    gol e um adversario em cima do finalizador como qualquer outro, e exclui-lo publicava `False`
    justamente no lance em que a pressao e maior. Arbitro e papel desconhecido continuam fora: o
    primeiro nao marca ninguem, e o segundo nao se sabe o que e.

    Posicao vem so de `pitch_xy`, sem o fallback `ground_point` de `candidates_from_frame`. O
    fallback e pixel enquanto `DEFAULT_RADIUS_M` e metro, e como ele e por deteccao um unico quadro
    misturaria as duas unidades dentro do mesmo `dist()`, em silencio. Quem nao tem `pitch_xy` sai
    do calculo e vai para `sem_posicao`, onde vira indeterminacao em vez de distancia inventada.
    """
    posicionados: list[MarkingCandidate] = []
    sem_posicao: list[tuple[int, Team]] = []
    for detection in frame.players:
        if as_role(detection.role) not in ("player", "goalkeeper"):
            continue
        if detection.track_id is None:
            continue
        team = teams.get(detection.track_id)
        if team is None:
            continue
        if detection.pitch_xy is None:
            sem_posicao.append((detection.track_id, team))
            continue
        posicionados.append(MarkingCandidate(detection.track_id, team, detection.pitch_xy))
    return PressureRoster(tuple(posicionados), tuple(sem_posicao))


def opponent_pressure(
    roster: PressureRoster, observed_key: int, *, radius: float = DEFAULT_RADIUS_M
) -> MarkingPressure:
    """Existe adversario do jogador `observed_key` dentro de `radius`.

    Afirmacao existencial, e nao pareamento reciproco como `pair_markings`. Um zagueiro que cobre
    dois atacantes marca os dois, mas o pareamento mutuo so devolvia um par e publicava `False`
    para o outro; para a pergunta "havia pressao no chute" isso e falso negativo. Aqui basta um
    adversario no raio, e `marcadores` lista todos os que estao la, do mais perto ao mais longe.

    Dado faltante so pode produzir falso negativo, nunca falso positivo: achou marcador, a resposta
    e `True` mesmo que outros adversarios estejam sem posicao, porque mais dado nao desfaz um
    marcador ja encontrado. Nao achou e havia adversario sem posicao, a resposta e indeterminada,
    porque o marcador podia ser exatamente ele.
    """
    if radius <= 0:
        raise MarkingError(f"radius {radius!r} precisa ser positivo")
    _refuse_duplicate_keys(roster.posicionados)
    observado = next((c for c in roster.posicionados if c.key == observed_key), None)
    if observado is None:
        ausente = any(key == observed_key for key, _ in roster.sem_posicao)
        return MarkingPressure(
            None, (), 0, "observado-sem-pitch-xy" if ausente else "observado-fora-do-quadro"
        )
    adversarios_sem_posicao = sum(1 for _, team in roster.sem_posicao if team != observado.team)
    marcadores = tuple(
        sorted(
            (
                (c.key, dist(observado.position, c.position))
                for c in roster.posicionados
                if c.team != observado.team and dist(observado.position, c.position) <= radius
            ),
            key=lambda item: (item[1], item[0]),
        )
    )
    if marcadores:
        return MarkingPressure(True, marcadores, adversarios_sem_posicao)
    if adversarios_sem_posicao:
        return MarkingPressure(None, (), adversarios_sem_posicao, "adversario-sem-pitch-xy")
    return MarkingPressure(False, (), 0)


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
    "MarkingPressure",
    "PressureRoster",
    "Projection",
    "candidates_from_frame",
    "marked_keys",
    "opponent_pressure",
    "pair_markings",
    "pressure_roster",
    "teams_from_identities",
]
