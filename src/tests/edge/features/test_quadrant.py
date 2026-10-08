"""`quadrant`: posicao do finalizador em metro levada a grade da Footstats."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from contracts import (
    BallDetection,
    Box,
    FeatureValue,
    FieldGeometry,
    FramePerception,
    Perception,
    PlayerDetection,
)
from edge.features.quadrant import Quadrant, celula_px

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


@dataclass(frozen=True)
class _Chute:
    frame_index: int
    shooter_track_id: int | None = None


def _cena(metro: tuple[float, float] | None, campo: FieldGeometry | None = CAMPO) -> Perception:
    finalizador = PlayerDetection(
        Box(480, 500, 520, 600), 0.8, track_id=7, team="team_a", role="player", pitch_xy=metro
    )
    bola = BallDetection(Box(495, 585, 505, 595), 0.9, source="detected")
    return Perception(
        clip_id="c",
        frames=tuple(
            FramePerception(
                frame_index=i,
                timestamp_s=i / FPS,
                players=(finalizador,),
                ball=bola,
                field=campo,
            )
            for i in range(0, 44, 4)
        ),
    )


def _extrair(metro: tuple[float, float]) -> FeatureValue:
    return Quadrant().extract(_cena(metro), _Chute(28))


def test_chute_na_boca_do_gol_vale_o_maximo() -> None:
    valor = _extrair((49.5, 0.5))

    assert (valor.status, valor.value) == ("ok", 0.49)
    assert (valor.evidence["linha"], valor.evidence["coluna"]) == (1, 9)
    assert (valor.evidence["lateral_m"], valor.evidence["profundidade_m"]) == (0.5, 3.0)


def test_os_dois_gols_caem_na_mesma_celula() -> None:
    direito, esquerdo = _extrair((49.5, 0.5)), _extrair((-49.5, -0.5))

    assert direito.evidence == esquerdo.evidence


@pytest.mark.parametrize(("y", "coluna", "esperado"), [(10.0, 11, 0.08), (-10.0, 6, 0.10)])
def test_lado_de_quem_ataca(y: float, coluna: int, esperado: float) -> None:
    valor = _extrair((49.5, y))

    assert (valor.evidence["coluna"], valor.value) == (coluna, esperado)


def test_celula_sem_rotulo_sai_com_baixa_confianca() -> None:
    valor = _extrair((7.5, -32.0))

    assert (valor.evidence["linha"], valor.evidence["coluna"]) == (9, 1)
    assert (valor.status, valor.value) == ("low_confidence", 0.01)
    assert valor.reason


def test_sem_homografia_nao_ha_quadrante() -> None:
    valor = Quadrant().extract(_cena(None, campo=None), _Chute(28))

    assert (valor.status, valor.value) == ("unavailable", None)


@pytest.mark.parametrize(
    ("x", "y", "celula"),
    [
        (-3.8, -11.5, (1, 1)),
        (50.9, 45.9, (1, 1)),
        (51.0, 46.0, (2, 2)),
        (780.0, 402.0, (9, 16)),
        (900.0, 500.0, (9, 16)),
    ],
)
def test_celula_px_segue_as_faixas_da_footstats(
    x: float, y: float, celula: tuple[int, int]
) -> None:
    assert celula_px(x, y) == celula
