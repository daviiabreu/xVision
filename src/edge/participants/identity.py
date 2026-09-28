"""Identificação temporal de participantes a partir do tracking existente."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import dist
from statistics import median
from typing import Literal, TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

from edge.participants.detector import Frame
from edge.participants.types import Box, PlayerDetection, Role, as_role

Team: TypeAlias = Literal["team_a", "team_b"]
LabColor: TypeAlias = tuple[float, float, float]
RoleSource: TypeAlias = Literal["temporal_vote", "color_assisted"]
ROLE_ORDER: tuple[Role, ...] = ("player", "goalkeeper", "referee")


@dataclass(frozen=True)
class TrackObservation:
    frame_index: int
    detection: PlayerDetection

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("frame_index não pode ser negativo")
        if self.detection.track_id is None:
            raise ValueError("a observação precisa de track_id")


def stabilize_role(observations: Iterable[TrackObservation]) -> tuple[Role, float]:
    """Escolhe a função da trajetória pelo total de confiança de cada classe.

    Observação cujo `role` não é papel deste detector não entra na soma: o campo é `str | None` no
    contrato, e uma trajetória inteira sem papel reconhecido cai no erro de "sem observações" em
    vez de eleger `player` por ser o primeiro da ordem.
    """
    scores: dict[Role, float] = {role: 0.0 for role in ROLE_ORDER}
    for observation in observations:
        detection = observation.detection
        role = as_role(detection.role)
        if role is None:
            continue
        scores[role] += detection.confidence

    total = sum(scores.values())
    if total == 0:
        raise ValueError("não há observações para estabilizar")

    stable_role = max(ROLE_ORDER, key=lambda role: scores[role])
    return stable_role, scores[stable_role] / total


def select_sample_observations(
    observations: Iterable[TrackObservation],
    *,
    limit: int = 5,
    role: Role | None = "player",
) -> tuple[TrackObservation, ...]:
    """Seleciona os melhores recortes, opcionalmente filtrados por função."""
    if limit <= 0:
        raise ValueError("limit deve ser positivo")

    candidates = [
        observation
        for observation in observations
        if role is None or as_role(observation.detection.role) == role
    ]

    def quality(observation: TrackObservation) -> tuple[float, float, int]:
        box = observation.detection.box
        area = max(box.x2 - box.x1, 0.0) * max(box.y2 - box.y1, 0.0)
        return observation.detection.confidence, area, -observation.frame_index

    selected = sorted(candidates, key=quality, reverse=True)[:limit]
    return tuple(sorted(selected, key=lambda observation: observation.frame_index))


def kmeans_two(
    points: NDArray[np.float32],
    *,
    max_iterations: int = 20,
) -> tuple[NDArray[np.int64], NDArray[np.float32]] | None:
    """Agrupa vetores em dois centroides sem inicialização aleatória."""
    if points.ndim != 2 or len(points) < 2:
        return None

    mean = points.mean(axis=0)
    first_index = int(np.argmax(np.sum((points - mean) ** 2, axis=1)))
    second_index = int(np.argmax(np.sum((points - points[first_index]) ** 2, axis=1)))
    centroids = np.stack((points[first_index], points[second_index])).astype(np.float32)
    if np.allclose(centroids[0], centroids[1]):
        return None

    labels = np.zeros(len(points), dtype=np.int64)
    for _ in range(max_iterations):
        distances = np.sum((points[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        next_labels = np.argmin(distances, axis=1).astype(np.int64)
        if np.array_equal(labels, next_labels):
            break
        labels = next_labels
        if any(not np.any(labels == cluster) for cluster in (0, 1)):
            return None
        next_centroids = np.stack(
            tuple(points[labels == cluster].mean(axis=0) for cluster in (0, 1))
        ).astype(np.float32)
        if np.allclose(centroids, next_centroids):
            centroids = next_centroids
            break
        centroids = next_centroids

    return labels, centroids


def extract_shirt_color(frame: Frame, box: Box) -> LabColor | None:
    """Extrai a cor da camisa na região do torso e usa os cantos como fundo."""
    frame_height, frame_width = frame.shape[:2]
    box_width = box.x2 - box.x1
    box_height = box.y2 - box.y1
    x1 = min(max(round(box.x1 + 0.2 * box_width), 0), frame_width)
    x2 = min(max(round(box.x2 - 0.2 * box_width), 0), frame_width)
    y1 = min(max(round(box.y1 + 0.2 * box_height), 0), frame_height)
    y2 = min(max(round(box.y1 + 0.55 * box_height), 0), frame_height)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None

    crop = frame[y1:y2, x1:x2]
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    points = lab.reshape(-1, 3).astype(np.float32)
    clustered = kmeans_two(points)
    if clustered is None:
        return None

    labels, _ = clustered
    label_image = labels.reshape(lab.shape[:2])
    corner_size = max(1, min(lab.shape[:2]) // 5)
    corner_labels = np.concatenate(
        (
            label_image[:corner_size, :corner_size].reshape(-1),
            label_image[:corner_size, -corner_size:].reshape(-1),
            label_image[-corner_size:, :corner_size].reshape(-1),
            label_image[-corner_size:, -corner_size:].reshape(-1),
        )
    )
    background_cluster = int(np.argmax(np.bincount(corner_labels, minlength=2)))
    shirt_points = points[labels != background_cluster]
    if len(shirt_points) < 4:
        return None

    color = np.median(shirt_points, axis=0)
    return float(color[0]), float(color[1]), float(color[2])


def representative_shirt_color(samples: Sequence[LabColor]) -> LabColor | None:
    """Resume as amostras da trajetória com uma mediana robusta."""
    if not samples:
        return None
    color = np.median(np.asarray(samples, dtype=np.float32), axis=0)
    return float(color[0]), float(color[1]), float(color[2])


DEFAULT_SAMPLE_LIMIT = 20
"""Amostras por trajetoria quando as cores ja estao em memoria e nenhuma custa um seek."""


def selected_colors(
    samples: Mapping[int, Sequence[tuple[TrackObservation, LabColor]]],
    *,
    limit: int = DEFAULT_SAMPLE_LIMIT,
) -> dict[int, list[LabColor]]:
    """Cores das melhores amostras de cada trajetoria, as mesmas que a cor representativa usa."""
    selecionadas: dict[int, list[LabColor]] = {}
    for track_id, itens in samples.items():
        cor_por_quadro = {observacao.frame_index: cor for observacao, cor in itens}
        melhores = select_sample_observations(
            [observacao for observacao, _ in itens], limit=limit, role=None
        )
        selecionadas[track_id] = [
            cor_por_quadro[observacao.frame_index]
            for observacao in melhores
            if observacao.frame_index in cor_por_quadro
        ]
    return selecionadas


def representative_colors(
    colors: Mapping[int, Sequence[LabColor]], *, minimum: int = 2
) -> dict[int, LabColor]:
    """Cor representativa de cada trajetoria que tem pelo menos `minimum` amostras."""
    return {
        track_id: cor
        for track_id, cores in colors.items()
        if len(cores) >= minimum and (cor := representative_shirt_color(cores)) is not None
    }


def teams_from_color_samples(
    samples: Mapping[int, Sequence[tuple[TrackObservation, LabColor]]],
    *,
    limit: int = DEFAULT_SAMPLE_LIMIT,
    minimum: int = 2,
) -> tuple[dict[int, Team], dict[Team, LabColor]]:
    """Equipe por trajetoria a partir de amostras ja extraidas. `({}, {})` quando nao separa."""
    if limit <= 0 or minimum <= 0 or minimum > limit:
        raise ValueError("limites de amostragem invalidos")

    representativas = representative_colors(selected_colors(samples, limit=limit), minimum=minimum)
    agrupado = cluster_teams(representativas)
    return agrupado if agrupado is not None else ({}, {})


def cluster_teams(
    colors_by_track: Mapping[int, LabColor],
) -> tuple[dict[int, Team], dict[Team, LabColor]] | None:
    """Separa as cores representativas em duas equipes com nomes estáveis."""
    if len(colors_by_track) < 2:
        return None

    track_ids = sorted(colors_by_track)
    points = np.asarray([colors_by_track[track_id] for track_id in track_ids], dtype=np.float32)
    clustered = kmeans_two(points[:, 1:])
    if clustered is None:
        return None

    labels, chroma_centroids = clustered
    ordered_clusters = sorted(
        (0, 1),
        key=lambda cluster: tuple(float(value) for value in chroma_centroids[cluster]),
    )
    team_by_cluster: dict[int, Team] = {
        ordered_clusters[0]: "team_a",
        ordered_clusters[1]: "team_b",
    }
    assignments: dict[int, Team] = {
        track_id: team_by_cluster[int(label)]
        for track_id, label in zip(track_ids, labels, strict=True)
    }
    team_centroids: dict[Team, LabColor] = {}
    for cluster in (0, 1):
        centroid = np.median(points[labels == cluster], axis=0)
        team_centroids[team_by_cluster[cluster]] = (
            float(centroid[0]),
            float(centroid[1]),
            float(centroid[2]),
        )
    return assignments, team_centroids


TEAM_CHROMA_LIMIT_FLOOR = 30.0
"""O limite minimo de distancia de croma ate o centroide para entrar num time."""
MIN_TEAM_CONTRAST = 40.0
"""Separacao minima, em Lab completo, entre os centroides dos dois times.

Nos 30 clipes de 17 partidas, as divisoes erradas, um time so partido em dois ou laranja
misturado nos dois grupos, ficam a 27, e as certas a partir de 58, incluindo branco contra
azul-marinho, que difere quase so no brilho e por isso nao se mede so pela croma."""
REFEREE_ANCHOR_RADIUS = 18.0
"""Raio em volta da trajetoria mais votada como arbitro em que as outras semeiam o centroide."""
AMBIGUOUS_TEAM_RATIO = 0.7
"""Razao entre as distancias ao time mais proximo e ao outro acima da qual a cor nao decide."""


def enough_team_contrast(centroids: Mapping[Team, LabColor]) -> bool:
    """Os dois times so se separam se os centroides ficam a mais de `MIN_TEAM_CONTRAST` em Lab.

    Abaixo disso o agrupamento dividiu o mesmo uniforme em dois, o que acontece no clipe em que
    so um time aparece, ou os uniformes sao parecidos demais para a cor decidir, e em vez de um
    time errado a trajetoria fica sem time, o `insufficient_team_contrast` do offline.
    """
    if len(centroids) != 2:
        return False
    first, second = centroids.values()
    return dist(first, second) > MIN_TEAM_CONTRAST


def ambiguous_team(color: LabColor, centroids: Mapping[Team, LabColor]) -> bool:
    """A cor fica perto demais do meio caminho entre os dois times para escolher um."""
    near, far = sorted(dist(color[1:], centroid[1:]) for centroid in centroids.values())
    return far > 0 and near / far > AMBIGUOUS_TEAM_RATIO


def _nearest_team(color: LabColor, centroids: Mapping[Team, LabColor]) -> tuple[Team, float]:
    distances: dict[Team, float] = {
        team: dist(color[1:], centroid[1:]) for team, centroid in centroids.items()
    }
    return min(distances.items(), key=lambda item: (item[1], item[0]))


def _robust_distance_limit(values: list[float]) -> float:
    if not values:
        return 30.0
    center = median(values)
    deviation = median(abs(value - center) for value in values)
    return min(max(center + 3 * deviation + 5, 18.0), 60.0)


def _full_team_limits(
    colors: Mapping[int, LabColor],
    seed_assignments: Mapping[int, Team],
    centroids: Mapping[Team, LabColor],
) -> dict[Team, float]:
    full_distances_by_team: dict[Team, list[float]] = defaultdict(list)
    for track_id, team in seed_assignments.items():
        color = colors.get(track_id)
        if color is not None:
            full_distances_by_team[team].append(dist(color, centroids[team]))
    return {
        team: _robust_distance_limit(full_distances_by_team.get(team, [])) for team in centroids
    }


def _outside_both_teams(
    color: LabColor, centroids: Mapping[Team, LabColor], full_team_limits: Mapping[Team, float]
) -> bool:
    distances: list[tuple[Team, float]] = [
        (team, dist(color, centroid)) for team, centroid in centroids.items()
    ]
    nearest_full_team, full_team_distance = min(distances, key=lambda item: (item[1], item[0]))
    return full_team_distance > full_team_limits[nearest_full_team]


def resolve_roles_and_teams(
    stable_roles: dict[int, Role],
    colors: dict[int, LabColor],
    seed_assignments: dict[int, Team],
    centroids: dict[Team, LabColor],
    *,
    role_evidence: dict[int, set[Role]] | None = None,
    color_samples: dict[int, list[LabColor]] | None = None,
    referee_votes: Mapping[int, float] | None = None,
) -> tuple[dict[int, Role], dict[int, Team], dict[int, RoleSource]]:
    """Papel e equipe por trajetoria, com a arbitragem como terceira cor.

    O detector erra o arbitro de forma consistente, e o voto por confianca da trajetoria perde:
    no piloto de 8 clipes, 20 das 31 trajetorias de arbitro tem algum quadro marcado como
    `referee` e mesmo assim ficam estaveis como jogador, e 11 nao tem quadro nenhum. Por isso o
    centroide da arbitragem nasce de toda trajetoria com evidencia de arbitro em algum quadro, e
    nao so das que ficaram estaveis como arbitro, o que deixava clipes inteiros sem centroide; e a
    cor que cai perto desse centroide e fora do limite dos dois times, em Lab completo, vira
    arbitro mesmo sem evidencia.

    Com `referee_votes`, a confianca somada dos quadros de `referee` de cada trajetoria, so semeia
    o centroide quem fica a ate `REFEREE_ANCHOR_RADIUS` da mais votada. Sem isso a semente mistura
    o arbitro com jogadores na sombra que tiveram um quadro de `referee`, a mediana da mistura e
    uma cor de ninguem e o limite abre ate 60: numa partida de branco contra azul, jogadores dos
    dois times viravam arbitro, e nos 30 clipes a arbitragem passava de 6% para 16% das deteccoes.
    Sem o argumento, o comportamento e o do offline.
    """
    resolved_roles = dict(stable_roles)
    resolved_teams: dict[int, Team] = {}
    sources: dict[int, RoleSource] = {track_id: "temporal_vote" for track_id in stable_roles}
    if len(centroids) != 2:
        return resolved_roles, resolved_teams, sources
    role_evidence = role_evidence or {}
    color_samples = color_samples or {}

    def has_referee_evidence(track_id: int) -> bool:
        return stable_roles[track_id] == "referee" or "referee" in role_evidence.get(
            track_id, set()
        )

    distances_by_team: dict[Team, list[float]] = defaultdict(list)
    for track_id, team in seed_assignments.items():
        color = colors.get(track_id)
        if color is not None:
            distances_by_team[team].append(dist(color[1:], centroids[team][1:]))
    team_limits = {
        team: max(_robust_distance_limit(distances_by_team.get(team, [])), TEAM_CHROMA_LIMIT_FLOOR)
        for team in centroids
    }
    full_team_limits = _full_team_limits(colors, seed_assignments, centroids)

    def outside_both_teams(color: LabColor) -> bool:
        return _outside_both_teams(color, centroids, full_team_limits)

    candidate_ids = [
        track_id
        for track_id in stable_roles
        if has_referee_evidence(track_id)
        and track_id in colors
        and outside_both_teams(colors[track_id])
    ]
    if referee_votes is not None and candidate_ids:
        anchor = max(
            candidate_ids, key=lambda track_id: (referee_votes.get(track_id, 0.0), -track_id)
        )
        candidate_ids = [
            track_id
            for track_id in candidate_ids
            if dist(colors[track_id], colors[anchor]) <= REFEREE_ANCHOR_RADIUS
        ]
    referee_candidates = [colors[track_id] for track_id in candidate_ids]
    referee_centroid = representative_shirt_color(referee_candidates)
    referee_limit = _robust_distance_limit(
        [dist(color, referee_centroid) for color in referee_candidates]
        if referee_centroid is not None
        else []
    )

    for track_id, stable_role in stable_roles.items():
        if stable_role == "goalkeeper":
            continue
        color = colors.get(track_id)
        if color is None:
            continue

        nearest_team, team_distance = _nearest_team(color, centroids)
        referee_distance = (
            dist(color, referee_centroid) if referee_centroid is not None else float("inf")
        )
        sample_referee_distance = (
            min(
                (dist(sample, referee_centroid) for sample in color_samples.get(track_id, ())),
                default=float("inf"),
            )
            if referee_centroid is not None
            else float("inf")
        )
        if has_referee_evidence(track_id) and (
            sample_referee_distance <= referee_limit or referee_distance <= referee_limit
        ):
            resolved_roles[track_id] = "referee"
            if stable_role != "referee":
                sources[track_id] = "color_assisted"
        elif referee_distance <= referee_limit and outside_both_teams(color):
            # Antes do teste de equipe, que olha so a crominancia: a arbitragem de preto tem o tom
            # neutro de um uniforme claro e so difere no brilho, e cairia no time dele.
            resolved_roles[track_id] = "referee"
            if stable_role != "referee":
                sources[track_id] = "color_assisted"
        elif team_distance <= team_limits[nearest_team]:
            resolved_roles[track_id] = "player"
            resolved_teams[track_id] = nearest_team
            if stable_role != "player":
                sources[track_id] = "color_assisted"

    return resolved_roles, resolved_teams, sources


GOAL_X_M = 52.5
PENALTY_AREA_DEPTH_M = 16.5
PENALTY_AREA_HALF_WIDTH_M = 20.16
GOAL_AREA_DEPTH_M = 5.5
GOAL_AREA_HALF_WIDTH_M = 9.16
PITCH_HALF_WIDTH_M = 34.0
OFF_PITCH_MARGIN_M = 3.0
"""No piloto, o staff atrás do gol fica a mais de 7 m da linha de fundo, e o jogador mais longe
das linhas, perto da bandeirinha de escanteio, fica a 1 m da lateral."""
OFF_PITCH_ROLE = "off_pitch"
"""Papel de quem está fora do jogo, staff, técnico e fotógrafo, que não é participante."""
GOALKEEPER_COLOR_LIMIT = 18.0
"""O piso do limite robusto: o outro fragmento do mesmo goleiro cai bem dentro dele."""


def off_pitch(point: tuple[float, float]) -> bool:
    """Posição além das linhas do campo, com a margem do erro de projeção do pé."""
    return (
        abs(point[0]) > GOAL_X_M + OFF_PITCH_MARGIN_M
        or abs(point[1]) > PITCH_HALF_WIDTH_M + OFF_PITCH_MARGIN_M
    )


def in_penalty_area(point: tuple[float, float]) -> bool:
    return _in_area(point, PENALTY_AREA_DEPTH_M, PENALTY_AREA_HALF_WIDTH_M)


def in_goal_area(point: tuple[float, float]) -> bool:
    return _in_area(point, GOAL_AREA_DEPTH_M, GOAL_AREA_HALF_WIDTH_M)


def _in_area(point: tuple[float, float], depth: float, half_width: float) -> bool:
    # A área termina na linha de fundo: o fotógrafo atrás do gol projeta em |x| acima de 52,5.
    return GOAL_X_M - depth <= abs(point[0]) <= GOAL_X_M and abs(point[1]) <= half_width


def resolve_goalkeepers(
    roles: Mapping[int, Role],
    teams: Mapping[int, Team],
    colors: Mapping[int, LabColor],
    centroids: Mapping[Team, LabColor],
    *,
    goalkeeper_votes: Mapping[int, float],
    positions: Mapping[int, tuple[float, float]],
    frames: Mapping[int, set[int]],
) -> tuple[dict[int, Role], dict[int, Team]]:
    """Goleiro pela cor fora dos dois times, no máximo um por lado, com o time do gol que defende.

    Roda depois de `resolve_roles_and_teams`, que deixa o goleiro estável de fora. O goleiro
    estável fica, porque o uniforme dele pode cair dentro do limite de um time: no piloto, o
    verde-água do goleiro fica a menos de 40 do uniforme claro do México em Lab, e rebaixar por
    isso tirava 5 goleiros de 18. Com `pitch_xy`, a trajetória com algum quadro de `goalkeeper`
    cuja posição mediana cai na pequena área também é goleiro. A partir desses, a cor a até
    `GOALKEEPER_COLOR_LIMIT` de um goleiro e mais perto dele que dos dois times vira goleiro, que é
    o fragmento que o voto deixou como jogador, desde que esteja na grande área ou, sem posição,
    tenha algum quadro de `goalkeeper`. Estar fora dos times na grande área não basta, porque o
    jogador na sombra também está.

    `positions` é a posição mediana no campo e `frames` são os quadros em que a trajetória
    aparece. Duas trajetórias de goleiro do mesmo lado ao mesmo tempo não cabem: fica a de cor
    mais longe dos dois times, e a outra volta a jogador com o time da cor. O time do goleiro é o
    que tem os jogadores, na mediana, mais perto daquele gol, porque é quem defende, e o elenco da
    `marked` conta o goleiro com time.
    """
    resolved_roles = dict(roles)
    resolved_teams = dict(teams)

    def team_distance(track_id: int) -> float:
        if track_id not in colors:
            return 0.0
        return min(
            (dist(colors[track_id], centroid) for centroid in centroids.values()),
            default=float("inf"),
        )

    def has_votes(track_id: int) -> bool:
        return goalkeeper_votes.get(track_id, 0.0) > 0.0

    # No conflito fica a cor mais longe dos dois times: o jogador na grande área que o detector
    # chama de goleiro costuma ter mais votos que o goleiro de verdade, e a cor dele é a do time.
    def one_per_side(candidates: set[int]) -> list[int]:
        ranked = sorted(
            candidates,
            key=lambda track_id: (
                -team_distance(track_id),
                -goalkeeper_votes.get(track_id, 0.0),
                track_id,
            ),
        )
        kept: list[int] = []
        for track_id in ranked:
            side = _side(positions.get(track_id))
            clash = side is not None and any(
                _side(positions.get(other)) == side
                and frames.get(track_id, set()) & frames.get(other, set())
                for other in kept
            )
            if not clash:
                kept.append(track_id)
        return kept

    initial = {
        track_id
        for track_id, role in resolved_roles.items()
        if role == "goalkeeper"
        or (
            role == "player"
            and has_votes(track_id)
            and track_id in positions
            and in_goal_area(positions[track_id])
        )
    }
    # O conflito sai antes da semente de cor, para o jogador que o detector chama de goleiro não
    # promover pela cor dele um colega de time na sombra.
    seeds = [track_id for track_id in one_per_side(initial) if track_id in colors]
    promoted: set[int] = set()
    for track_id, color in colors.items():
        if track_id in initial or resolved_roles.get(track_id) != "player":
            continue
        if track_id in positions:
            if not in_penalty_area(positions[track_id]):
                continue
        elif not has_votes(track_id):
            continue
        seed_distance = min((dist(color, colors[seed]) for seed in seeds), default=float("inf"))
        if seed_distance <= GOALKEEPER_COLOR_LIMIT and seed_distance < team_distance(track_id):
            promoted.add(track_id)

    kept = one_per_side(set(seeds) | promoted | (initial - set(colors)))
    for track_id in (initial | promoted) - set(kept):
        resolved_roles[track_id] = "player"
        if track_id not in resolved_teams and track_id in colors and len(centroids) == 2:
            resolved_teams[track_id] = _nearest_team(colors[track_id], centroids)[0]
    for track_id in kept:
        resolved_roles[track_id] = "goalkeeper"
        resolved_teams.pop(track_id, None)
    for track_id in kept:
        team = _defending_team(
            _side(positions.get(track_id)), resolved_roles, resolved_teams, positions
        )
        if team is not None:
            resolved_teams[track_id] = team
    return resolved_roles, resolved_teams


def _side(position: tuple[float, float] | None) -> int | None:
    if position is None or position[0] == 0:
        return None
    return 1 if position[0] > 0 else -1


def _defending_team(
    side: int | None,
    roles: Mapping[int, Role],
    teams: Mapping[int, Team],
    positions: Mapping[int, tuple[float, float]],
) -> Team | None:
    if side is None:
        return None
    goal = (side * GOAL_X_M, 0.0)
    distances: dict[Team, list[float]] = defaultdict(list)
    for track_id, team in teams.items():
        if roles.get(track_id) == "player" and track_id in positions:
            distances[team].append(dist(positions[track_id], goal))
    if len(distances) != 2:
        return None
    return min(distances, key=lambda team: (median(distances[team]), team))
