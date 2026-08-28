"""Identificação temporal de participantes a partir do tracking existente."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

from edge.participants.detector import Frame
from edge.participants.types import Box, PlayerDetection, Role, as_role

Team: TypeAlias = Literal["team_a", "team_b"]
LabColor: TypeAlias = tuple[float, float, float]
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


def teams_from_color_samples(
    samples: Mapping[int, Sequence[tuple[TrackObservation, LabColor]]],
    *,
    limit: int = DEFAULT_SAMPLE_LIMIT,
    minimum: int = 2,
) -> tuple[dict[int, Team], dict[Team, LabColor]]:
    """Equipe por trajetoria a partir de amostras ja extraidas. `({}, {})` quando nao separa."""
    if limit <= 0 or minimum <= 0 or minimum > limit:
        raise ValueError("limites de amostragem invalidos")

    representativas: dict[int, LabColor] = {}
    for track_id, itens in samples.items():
        cor_por_quadro = {observacao.frame_index: cor for observacao, cor in itens}
        melhores = select_sample_observations(
            [observacao for observacao, _ in itens], limit=limit, role=None
        )
        cores = [
            cor_por_quadro[observacao.frame_index]
            for observacao in melhores
            if observacao.frame_index in cor_por_quadro
        ]
        if len(cores) < minimum:
            continue
        cor = representative_shirt_color(cores)
        if cor is not None:
            representativas[track_id] = cor

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
