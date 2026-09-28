"""`occupied_area` sobre a cena do chute: medidas em metro e decisao preliminar."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from contracts import (
    BallDetection,
    Box,
    FieldGeometry,
    FramePerception,
    Perception,
    PlayerDetection,
)
from edge.features.occupied_area import OccupiedArea, measure_occupation

FPS = 59.94
CAMPO = FieldGeometry(
    homography=(10.0, 0.0, 960.0, 0.0, -10.0, 540.0, 0.0, 0.0, 1.0),
    reprojection_error_px=0.2,
    reprojection_error_p95_px=0.4,
    keypoints=(),
    observed_keypoint_count=8,
    inlier_count=8,
    source="detected",
    age_frames=0,
    template_id="pitch-105x68-center-v1",
    reason=None,
)
FINALIZADOR = PlayerDetection(
    Box(480, 500, 520, 600), 0.8, track_id=7, team="team_a", role="player", pitch_xy=(40.0, 0.0)
)


@dataclass(frozen=True)
class _Chute:
    frame_index: int
    shooter_track_id: int | None = None


def _outro(
    track: int, metro: tuple[float, float], team: str | None = "team_b", role: str = "player"
) -> PlayerDetection:
    """Longe do finalizador em pixel, para o toque nao disputar a bola."""
    x = 1100.0 + 60.0 * track
    return PlayerDetection(
        Box(x - 20, 500, x + 20, 600), 0.8, track_id=track, team=team, role=role, pitch_xy=metro
    )


def _cena(*outros: PlayerDetection, finalizador: PlayerDetection = FINALIZADOR) -> Perception:
    bola = BallDetection(Box(495, 585, 505, 595), 0.9, source="detected")
    return Perception(
        clip_id="c",
        frames=tuple(
            FramePerception(
                frame_index=i,
                timestamp_s=i / FPS,
                players=(finalizador, *outros),
                ball=bola,
                field=CAMPO,
            )
            for i in range(0, 44, 4)
        ),
    )


def test_adversario_no_cone_ocupa_a_area() -> None:
    valor = OccupiedArea().extract(_cena(_outro(9, (46.0, 0.0))), _Chute(28))

    assert (valor.status, valor.value) == ("ok", True)
    assert valor.evidence["adversarios_no_cone"] == 1
    assert valor.evidence["origem_do_finalizador"] == "touch"


def test_adversario_fora_do_cone_nao_ocupa_mas_conta_na_area() -> None:
    valor = OccupiedArea().extract(_cena(_outro(9, (46.0, 10.0))), _Chute(28))

    assert valor.value is False
    assert valor.evidence["jogadores_no_cone"] == 0
    assert valor.evidence["jogadores_na_area"] == 1


def test_goleiro_e_arbitro_nao_contam_como_jogador_no_cone() -> None:
    medidas = measure_occupation(
        (
            FINALIZADOR,
            _outro(1, (52.0, 0.0), team=None, role="goalkeeper"),
            _outro(2, (45.0, 0.0), team=None, role="referee"),
        ),
        FINALIZADOR,
        (40.0, 0.0),
    )

    assert medidas["jogadores_no_cone"] == 0
    assert medidas["goleiro_no_cone"] is True


def test_companheiro_no_cone_ocupa_mas_nao_e_adversario() -> None:
    medidas = measure_occupation(
        (FINALIZADOR, _outro(3, (46.0, 0.5), team="team_a")), FINALIZADOR, (40.0, 0.0)
    )

    assert (medidas["jogadores_no_cone"], medidas["adversarios_no_cone"]) == (1, 0)
    assert medidas["adversario_mais_proximo_m"] is None


def test_finalizador_sem_time_deixa_adversario_como_none() -> None:
    sem_time = PlayerDetection(FINALIZADOR.box, 0.8, track_id=7, role="player", pitch_xy=(40.0, 0))

    medidas = measure_occupation((sem_time, _outro(9, (46.0, 0.0))), sem_time, (40.0, 0.0))

    assert medidas["jogadores_no_cone"] == 1
    assert medidas["adversarios_no_cone"] is None
    assert medidas["adversarios_na_area"] is None


def test_raios_distancia_e_gol_atacado_do_outro_lado() -> None:
    medidas = measure_occupation(
        (FINALIZADOR, _outro(9, (-41.5, 0.0)), _outro(10, (-44.0, 3.0))), FINALIZADOR, (-40.0, 0.0)
    )

    assert (medidas["jogadores_a_2_m"], medidas["jogadores_a_5_m"]) == (1, 2)
    assert medidas["adversario_mais_proximo_m"] == pytest.approx(1.5)
    assert medidas["distancia_ao_gol_m"] == pytest.approx(12.5)
    assert medidas["jogadores_no_cone"] == 1


def test_sem_homografia_perto_do_contato_e_indisponivel() -> None:
    cena = _cena(_outro(9, (46.0, 0.0)))
    sem_campo = Perception(
        clip_id="c",
        frames=tuple(
            FramePerception(
                frame_index=q.frame_index,
                timestamp_s=q.timestamp_s,
                players=tuple(
                    PlayerDetection(j.box, j.confidence, track_id=j.track_id, team=j.team)
                    for j in q.players
                ),
                ball=q.ball,
            )
            for q in cena.frames
        ),
    )

    valor = OccupiedArea().extract(sem_campo, _Chute(28))

    assert valor.status == "unavailable"
    assert "homografia" in (valor.reason or "")
