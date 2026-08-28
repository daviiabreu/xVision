import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from xvision_players_yolo import Box
from xvision_players_yolo.identity import (
    LabColor,
    TrackObservation,
    cluster_teams,
    extract_shirt_color,
    representative_shirt_color,
    select_sample_observations,
    stabilize_role,
    teams_from_color_samples,
)
from xvision_players_yolo.types import player


def observation(
    frame_index: int,
    role: str,
    confidence: float,
    *,
    track_id: int = 4,
    width: int = 20,
) -> TrackObservation:
    return TrackObservation(
        frame_index,
        player(Box(0, 0, width, 40), confidence, role, track_id),  # type: ignore[arg-type]
    )


def test_role_is_stabilized_by_accumulated_confidence() -> None:
    stable_role, confidence = stabilize_role(
        (
            observation(0, "goalkeeper", 0.9),
            observation(1, "player", 0.6),
            observation(2, "goalkeeper", 0.8),
        )
    )

    assert stable_role == "goalkeeper"
    assert confidence == pytest.approx(1.7 / 2.3)


def test_sample_selection_prefers_confidence_then_box_area() -> None:
    selected = select_sample_observations(
        (
            observation(0, "player", 0.8, width=30),
            observation(1, "player", 0.9, width=10),
            observation(2, "referee", 0.99),
            observation(3, "player", 0.8, width=20),
        ),
        limit=2,
    )

    assert [item.frame_index for item in selected] == [0, 1]


def test_sample_selection_can_include_every_role() -> None:
    selected = select_sample_observations(
        (
            observation(0, "player", 0.8),
            observation(1, "referee", 0.9),
        ),
        role=None,
    )

    assert [item.frame_index for item in selected] == [0, 1]


def test_observation_requires_track_id() -> None:
    with pytest.raises(ValueError, match="track_id"):
        TrackObservation(0, player(Box(0, 0, 10, 20), 0.8, "player"))


def test_shirt_color_discards_background_from_crop_corners() -> None:
    frame = np.full((40, 40, 3), (0, 160, 0), dtype=np.uint8)
    frame[2:18, 10:30] = (0, 0, 220)

    color = extract_shirt_color(frame, Box(0, 0, 40, 40))
    red_pixel: NDArray[np.uint8] = np.array([[[0, 0, 220]]], dtype=np.uint8)
    expected = cv2.cvtColor(red_pixel, cv2.COLOR_BGR2LAB)[0, 0]

    assert color == pytest.approx(tuple(float(value) for value in expected), abs=1)


def test_shirt_color_is_unavailable_without_color_contrast() -> None:
    frame = np.full((20, 20, 3), (0, 160, 0), dtype=np.uint8)

    assert extract_shirt_color(frame, Box(0, 0, 20, 20)) is None


def test_representative_color_uses_median_of_samples() -> None:
    assert representative_shirt_color(((10, 20, 30), (12, 22, 32), (200, 200, 200))) == (
        12,
        22,
        32,
    )


def test_team_clustering_is_deterministic_for_mapping_order() -> None:
    colors = {
        4: (40, 80, 70),
        1: (42, 82, 72),
        9: (180, 140, 190),
        2: (178, 138, 188),
    }

    first = cluster_teams(colors)
    second = cluster_teams(dict(reversed(tuple(colors.items()))))

    assert first is not None
    assert second is not None
    assert first == second
    assignments, _ = first
    assert assignments[1] == assignments[4]
    assert assignments[2] == assignments[9]
    assert assignments[1] != assignments[2]


def test_team_clustering_requires_two_distinct_colors() -> None:
    assert cluster_teams({1: (10, 20, 30)}) is None
    assert cluster_teams({1: (10, 20, 30), 2: (10, 20, 30)}) is None


# --- equipe a partir de amostras ja em memoria ---------------------------------------------


def amostras(
    track_id: int,
    cores: list[LabColor],
    *,
    confianca: float = 0.9,
) -> list[tuple[TrackObservation, LabColor]]:
    return [
        (observation(indice, "player", confianca, track_id=track_id), cor)
        for indice, cor in enumerate(cores)
    ]


def test_equipe_sai_das_amostras_em_memoria() -> None:
    resultado, centroides = teams_from_color_samples(
        {
            1: amostras(1, [(150.0, 120.0, 200.0)] * 3),
            2: amostras(2, [(150.0, 120.0, 130.0)] * 3),
        }
    )

    assert resultado[1] != resultado[2]
    assert set(centroides) == {"team_a", "team_b"}


def test_track_com_menos_amostras_que_o_minimo_fica_de_fora() -> None:
    resultado, _ = teams_from_color_samples(
        {
            1: amostras(1, [(150.0, 120.0, 200.0)] * 3),
            2: amostras(2, [(150.0, 120.0, 130.0)] * 3),
            3: amostras(3, [(150.0, 120.0, 165.0)]),
        },
        minimum=2,
    )

    assert 3 not in resultado


def test_o_limite_de_amostras_muda_a_equipe_de_quem_esta_na_fronteira() -> None:
    """Com poucas amostras, um recorte ruim de confianca alta leva o jogador para o outro time."""
    verdes = {tid: amostras(tid, [(150.0, 120.0, 130.0)] * 12) for tid in (1, 2, 3)}
    amarelos = {tid: amostras(tid, [(150.0, 120.0, 210.0)] * 12) for tid in (4, 5, 6)}
    # amarelo de verdade, mas os quatro recortes de maior confianca sairam lavados
    fronteira = [
        (observation(indice, "player", 0.99, track_id=7), (150.0, 120.0, 140.0))
        for indice in range(4)
    ] + [
        (observation(4 + indice, "player", 0.10, track_id=7), (150.0, 120.0, 210.0))
        for indice in range(8)
    ]
    quadro = {**verdes, **amarelos, 7: fronteira}

    curto, _ = teams_from_color_samples(quadro, limit=4)
    longo, _ = teams_from_color_samples(quadro, limit=12)

    assert curto[7] == curto[1], "com quatro amostras o atacante vai para o time dos verdes"
    assert longo[7] == longo[4], "com doze ele volta para o time dos amarelos"


def test_sem_contraste_devolve_vazio_em_vez_de_estourar() -> None:
    resultado, centroides = teams_from_color_samples({1: amostras(1, [(150.0, 120.0, 200.0)] * 3)})

    assert resultado == {}
    assert centroides == {}


def test_limites_de_amostragem_invalidos_sao_erro() -> None:
    with pytest.raises(ValueError, match="limites de amostragem"):
        teams_from_color_samples({}, limit=0)
    with pytest.raises(ValueError, match="limites de amostragem"):
        teams_from_color_samples({}, limit=2, minimum=3)
