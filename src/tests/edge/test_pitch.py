"""O gabarito da fixture: posicao em metros conhecida por construcao, nao medida depois."""

from __future__ import annotations

import numpy as np
import pytest

from edge.testing.pitch import (
    GOAL_HALF_WIDTH_M,
    GOAL_LINE_X_M,
    SHOOTER_TRACK_ID,
    SHOT_FRAME_INDEX,
    clip_truth,
    homography_matrix,
    player_box,
    project,
)


def test_the_truth_covers_every_frame_of_the_clip() -> None:
    gabarito = clip_truth()

    assert len(gabarito.ball_pitch_xy) == gabarito.frame_count
    for jogador in gabarito.players:
        assert len(jogador.pitch_xy) == gabarito.frame_count


def test_the_shot_frame_is_where_the_ball_leaves_the_shooter() -> None:
    gabarito = clip_truth()
    finalizador = next(j for j in gabarito.players if j.track_id == SHOOTER_TRACK_ID)

    contato = gabarito.ball_pitch_xy[SHOT_FRAME_INDEX]

    assert contato == finalizador.pitch_xy[SHOT_FRAME_INDEX]
    assert gabarito.ball_pitch_xy[-1] == (GOAL_LINE_X_M, 0.0)


def test_the_ball_only_reaches_the_goal_after_the_contact() -> None:
    gabarito = clip_truth()
    antes = gabarito.ball_pitch_xy[SHOT_FRAME_INDEX - 1]
    depois = gabarito.ball_pitch_xy[SHOT_FRAME_INDEX + 1]

    assert antes[0] < gabarito.ball_pitch_xy[SHOT_FRAME_INDEX][0] < depois[0]


def test_exactly_one_goalkeeper_stands_between_the_posts() -> None:
    gabarito = clip_truth()
    goleiros = [j for j in gabarito.players if j.role == "goalkeeper"]

    assert len(goleiros) == 1
    x, y = goleiros[0].pitch_xy[SHOT_FRAME_INDEX]
    assert abs(y) < GOAL_HALF_WIDTH_M
    assert x < GOAL_LINE_X_M


def test_the_homography_projects_the_pitch_inside_the_frame() -> None:
    gabarito = clip_truth()

    for x, y in ((36.0, -20.0), (36.0, 20.0), (GOAL_LINE_X_M, 0.0)):
        u, v = project(gabarito.homography, x, y)
        assert 0 <= u <= gabarito.width
        assert 0 <= v <= gabarito.height


def test_the_homography_is_invertible_and_round_trips_in_metres() -> None:
    gabarito = clip_truth()
    inversa = np.linalg.inv(homography_matrix(gabarito.homography))

    for x, y in ((40.0, -8.0), (48.0, 3.0)):
        u, v = project(gabarito.homography, x, y)
        volta = inversa @ np.array([u, v, 1.0])
        assert abs(float(volta[0] / volta[2]) - x) < 1e-6
        assert abs(float(volta[1] / volta[2]) - y) < 1e-6


def test_the_homography_goes_from_metres_to_pixels_and_not_the_other_way() -> None:
    gabarito = clip_truth()

    no_gol = project(gabarito.homography, GOAL_LINE_X_M, 0.0)
    na_entrada_da_area = project(gabarito.homography, 36.0, 0.0)

    assert no_gol == pytest.approx((160.0, 188.47), abs=0.01)
    assert na_entrada_da_area == pytest.approx((160.0, 63.09), abs=0.01)
    assert no_gol[1] > na_entrada_da_area[1]


def test_the_player_box_stands_on_its_feet_and_has_human_proportions() -> None:
    gabarito = clip_truth()

    for jogador in gabarito.players:
        posicao = jogador.pitch_xy[SHOT_FRAME_INDEX]
        x1, y1, x2, y2 = player_box(gabarito, posicao)
        largura, altura = x2 - x1, y2 - y1
        _, v = project(gabarito.homography, posicao[0], posicao[1])

        assert y2 == pytest.approx(v), f"a caixa de {jogador.track_id} nao esta nos pes"
        assert largura < altura
        assert 0.25 <= largura / altura <= 0.6
