import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from edge.participants import Box
from edge.participants.identity import (
    LabColor,
    Team,
    TrackObservation,
    ambiguous_team,
    cluster_teams,
    enough_team_contrast,
    extract_shirt_color,
    off_pitch,
    representative_shirt_color,
    resolve_goalkeepers,
    resolve_roles_and_teams,
    select_sample_observations,
    stabilize_role,
    teams_from_color_samples,
)
from edge.participants.types import Role, player


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


UNIFORME_CLARO: LabColor = (200, 118, 138)
AMARELO: LabColor = (211, 120, 204)
PRETO: LabColor = (70, 126, 132)
"""O preto da arbitragem tem quase a crominancia do uniforme claro e so difere no brilho."""


def test_arbitro_sem_papel_estavel_de_arbitro_ainda_semeia_o_centroide() -> None:
    """Clipe em que o voto deixa todos como jogador: um quadro de `referee` basta de semente."""
    roles, teams, sources = resolve_roles_and_teams(
        {1: "player", 2: "player", 3: "player"},
        {1: UNIFORME_CLARO, 2: AMARELO, 3: PRETO},
        {1: "team_a", 2: "team_b"},
        {"team_a": UNIFORME_CLARO, "team_b": AMARELO},
        role_evidence={3: {"player", "referee"}},
    )

    assert roles[3] == "referee"
    assert 3 not in teams
    assert sources[3] == "color_assisted"


def test_arbitro_de_preto_sem_evidencia_vira_arbitro_pela_cor_e_nao_jogador_do_time_claro() -> None:
    """Sem quadro de `referee`, a cor perto da arbitragem e fora dos dois times decide."""
    estaveis: dict[int, Role] = {1: "player", 2: "player", 3: "referee", 4: "player", 5: "player"}
    roles, teams, _ = resolve_roles_and_teams(
        estaveis,
        {1: UNIFORME_CLARO, 2: AMARELO, 3: PRETO, 4: (60, 124, 134), 5: (212, 121, 203)},
        {1: "team_a", 2: "team_b", 5: "team_b"},
        {"team_a": UNIFORME_CLARO, "team_b": AMARELO},
        role_evidence={5: {"player", "referee"}},
    )

    assert roles[4] == "referee"
    assert 4 not in teams
    # O jogador de amarelo que o detector marcou como arbitro em algum quadro continua jogador.
    assert roles[5] == "player"
    assert teams[5] == "team_b"


VERDE_AGUA: LabColor = (180, 112, 110)
"""O uniforme do goleiro do piloto, a menos de 40 do uniforme claro em Lab completo."""
CENTROIDES: dict[Team, LabColor] = {"team_a": UNIFORME_CLARO, "team_b": AMARELO}


def _goleiros(
    roles: dict[int, Role],
    colors: dict[int, LabColor],
    *,
    teams: dict[int, Team] | None = None,
    votos: dict[int, float] | None = None,
    posicoes: dict[int, tuple[float, float]] | None = None,
    quadros: dict[int, set[int]] | None = None,
) -> tuple[dict[int, Role], dict[int, Team]]:
    return resolve_goalkeepers(
        roles,
        teams or {},
        colors,
        CENTROIDES,
        goalkeeper_votes=votos or {},
        positions=posicoes or {},
        frames=quadros or {},
    )


def test_goleiro_estavel_com_cor_perto_de_um_time_continua_goleiro() -> None:
    roles, teams = _goleiros({1: "goalkeeper"}, {1: VERDE_AGUA})

    assert roles[1] == "goalkeeper"
    assert 1 not in teams


def test_fragmento_com_quadro_de_goleiro_e_a_cor_do_goleiro_vira_goleiro() -> None:
    """Sem posição, a cor do goleiro estável e algum quadro de `goalkeeper` bastam."""
    roles, teams = _goleiros(
        {1: "goalkeeper", 2: "player", 3: "player"},
        {1: VERDE_AGUA, 2: (178, 113, 112), 3: (179, 112, 111)},
        teams={2: "team_a", 3: "team_a"},
        votos={1: 5.0, 2: 0.4},
    )

    assert roles[2] == "goalkeeper"
    assert 2 not in teams
    # Mesma cor, mas nenhum quadro de goleiro e nenhuma posição: continua jogador.
    assert roles[3] == "player"
    assert teams[3] == "team_a"


def test_na_pequena_area_com_quadro_de_goleiro_vira_goleiro_e_o_amarelo_na_grande_area_nao() -> (
    None
):
    roles, _ = _goleiros(
        {1: "player", 2: "player"},
        {1: VERDE_AGUA, 2: (160, 124, 190)},
        teams={1: "team_a", 2: "team_b"},
        votos={1: 1.5},
        posicoes={1: (48.2, -1.8), 2: (41.5, -7.0)},
    )

    assert roles[1] == "goalkeeper"
    assert roles[2] == "player"


def test_a_area_termina_na_linha_de_fundo() -> None:
    """O fotógrafo atrás do gol projeta em x acima de 52,5 e não é goleiro pela área."""
    roles, _ = _goleiros({1: "player"}, {1: VERDE_AGUA}, votos={1: 2.0}, posicoes={1: (60.9, 0.0)})

    assert roles[1] == "player"


def test_dois_goleiros_do_mesmo_lado_ao_mesmo_tempo_fica_a_cor_mais_longe_dos_times() -> None:
    roles, teams = _goleiros(
        {1: "goalkeeper", 2: "goalkeeper"},
        # A cor do goleiro do 4995957 e o amarelo na sombra que o detector chamou de goleiro.
        {1: (150, 114, 112), 2: (166, 125, 194)},
        votos={1: 10.0, 2: 30.0},
        posicoes={1: (49.9, 0.3), 2: (51.2, -7.5)},
        quadros={1: {0, 4, 8}, 2: {4, 8, 12}},
    )

    assert roles[1] == "goalkeeper"
    assert roles[2] == "player"
    assert teams[2] == "team_b"


def test_goleiros_de_lados_opostos_convivem() -> None:
    roles, _ = _goleiros(
        {1: "goalkeeper", 2: "goalkeeper"},
        {1: VERDE_AGUA, 2: (150, 180, 120)},
        posicoes={1: (50.0, 0.0), 2: (-50.0, 0.0)},
        quadros={1: {0, 4}, 2: {0, 4}},
    )

    assert roles == {1: "goalkeeper", 2: "goalkeeper"}


def test_o_goleiro_recebe_o_time_que_defende_o_gol_dele() -> None:
    """Quem defende tem os jogadores, na mediana, mais perto do próprio gol."""
    roles, teams = _goleiros(
        {1: "goalkeeper", 2: "player", 3: "player", 4: "player", 5: "player"},
        {1: VERDE_AGUA, 2: UNIFORME_CLARO, 3: UNIFORME_CLARO, 4: AMARELO, 5: AMARELO},
        teams={2: "team_a", 3: "team_a", 4: "team_b", 5: "team_b"},
        posicoes={1: (50.0, 0.0), 2: (30.0, 5.0), 3: (20.0, -5.0), 4: (45.0, 3.0), 5: (40.0, -8.0)},
    )

    assert roles[1] == "goalkeeper"
    assert teams[1] == "team_b"


def test_fora_do_campo_e_alem_das_linhas_com_a_margem_da_projecao() -> None:
    assert off_pitch((60.9, -0.5))  # fotógrafo atrás do gol
    assert off_pitch((20.0, -38.0))  # banco, além da lateral
    assert not off_pitch((51.8, -35.0))  # jogador perto da bandeirinha de escanteio
    assert not off_pitch((0.0, 0.0))


def test_times_com_croma_perto_demais_nao_tem_contraste() -> None:
    """Um uniforme só, dividido em dois pelo agrupamento, fica a menos do limite de croma."""
    assert not enough_team_contrast({"team_a": (200, 118, 138), "team_b": (170, 122, 150)})
    assert enough_team_contrast({"team_a": UNIFORME_CLARO, "team_b": AMARELO})
    assert not enough_team_contrast({})


def test_branco_contra_azul_marinho_tem_contraste_mesmo_com_croma_parecida() -> None:
    """A Tunísia contra o Japão difere quase só no brilho, e o agrupamento acerta a divisão."""
    assert enough_team_contrast({"team_a": (172, 130, 131), "team_b": (62, 136, 110)})


def test_cor_a_meio_caminho_entre_os_times_e_ambigua() -> None:
    assert ambiguous_team((205, 119, 171), CENTROIDES)
    assert not ambiguous_team((205, 119, 140), CENTROIDES)


def test_com_votos_so_o_arbitro_mais_votado_e_quem_fica_perto_dele_semeiam_o_centroide() -> None:
    """Branco contra azul, árbitro de verde-limão e um jogador de branco na sombra com um quadro
    de `referee`: sem os votos a semente mistura os dois e o cinza da mistura leva o jogador."""
    entradas = dict(
        stable_roles={1: "player", 2: "player", 3: "referee", 4: "player"},
        colors={1: (216, 126, 134), 2: (88, 144, 84), 3: (204, 72, 160), 4: (150, 128, 140)},
        seed_assignments={1: "team_a", 2: "team_b"},
        centroids={"team_a": (216, 126, 134), "team_b": (88, 144, 84)},
        role_evidence={4: {"player", "referee"}},
    )

    sem_votos, _, _ = resolve_roles_and_teams(**entradas)  # type: ignore[arg-type]
    com_votos, teams, _ = resolve_roles_and_teams(
        **entradas,  # type: ignore[arg-type]
        referee_votes={3: 20.0, 4: 0.6},
    )

    assert sem_votos[4] == "referee"
    assert com_votos[3] == "referee"
    assert com_votos[4] == "player"
    assert teams[4] == "team_a"
