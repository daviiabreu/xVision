"""A cena do chute: quadro com homografia perto do contato, finalizador nele e o pe como ancora."""

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
from edge.features.scene import TOLERANCIA_S, SceneRefusal, ShotScene, shot_scene

FPS = 59.94
H = (10.0, 0.0, 960.0, 0.0, -10.0, 540.0, 0.0, 0.0, 1.0)
CAMPO = FieldGeometry(
    homography=H,
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
FINALIZADOR = 7
OUTRO = 9


@dataclass(frozen=True)
class _Chute:
    frame_index: int
    shooter_track_id: int | None = None


def _jogador(x: float, track: int | None, *, metro: tuple[float, float] | None) -> PlayerDetection:
    """Caixa de 40 por 100 com base em y = 600, e o ponto em metro que o runner teria projetado."""
    return PlayerDetection(Box(x - 20, 500, x + 20, 600), 0.8, track_id=track, pitch_xy=metro)


def _quadro(
    indice: int,
    *,
    campo: bool,
    bola_em: float | None = 500.0,
    finalizador: int | None = FINALIZADOR,
) -> FramePerception:
    """Finalizador em x = 500 com a bola no pe, e outro jogador longe, em x = 1100."""
    jogadores = (
        _jogador(500, finalizador, metro=(-46.0, -6.0) if campo else None),
        _jogador(1100, OUTRO, metro=(14.0, -6.0) if campo else None),
    )
    bola = (
        None
        if bola_em is None
        else BallDetection(Box(bola_em - 5, 585, bola_em + 5, 595), 0.9, source="detected")
    )
    return FramePerception(
        frame_index=indice,
        timestamp_s=indice / FPS,
        players=jogadores,
        ball=bola,
        field=CAMPO if campo else None,
    )


def _percepcao(com_campo: set[int], *, bola_em: float | None = 500.0) -> Perception:
    """Onze quadros no passo 4 do `dense`, de 0 a 40, com homografia so nos indices pedidos."""
    return Perception(
        clip_id="c",
        frames=tuple(_quadro(i, campo=i in com_campo, bola_em=bola_em) for i in range(0, 44, 4)),
    )


def test_a_cena_usa_o_quadro_com_campo_mais_perto_e_no_empate_o_anterior() -> None:
    """O contato em 28 fica a quatro quadros de 24 e de 32. O anterior ainda tem a bola no pe."""
    cena = shot_scene(_percepcao({24, 32}), _Chute(28))

    assert isinstance(cena, ShotScene)
    assert cena.frame.frame_index == 24
    assert cena.shooter.track_id == FINALIZADOR
    assert cena.shooter_source == "touch"
    assert cena.anchor_xy == (-46.0, -6.0)
    assert cena.offset_s == pytest.approx(-4 / FPS)


def test_homografia_fora_da_tolerancia_e_recusa_e_nao_quadro_de_outra_jogada() -> None:
    """O quadro 0 esta a 0,67 s do contato em 40, mais que os 0,2 s da tolerancia."""
    assert 40 / FPS > TOLERANCIA_S

    cena = shot_scene(_percepcao({0}), _Chute(40))

    assert isinstance(cena, SceneRefusal)
    assert cena.reason == "sem-quadro-com-campo"


def test_a_tolerancia_e_ajustavel_para_quem_mede_cobertura_contra_acerto() -> None:
    cena = shot_scene(_percepcao({0}), _Chute(40), tolerance_s=1.0)

    assert isinstance(cena, ShotScene)
    assert cena.frame.frame_index == 0


def test_quadro_com_homografia_mas_sem_jogador_em_metro_nao_serve() -> None:
    quadros = list(_percepcao(set()).frames)
    quadros[7] = FramePerception(
        frame_index=28, timestamp_s=28 / FPS, players=(), ball=None, field=CAMPO
    )

    cena = shot_scene(Perception(clip_id="c", frames=tuple(quadros)), _Chute(28))

    assert isinstance(cena, SceneRefusal)
    assert cena.reason == "sem-quadro-com-campo"


def test_o_finalizador_declarado_vence_o_toque() -> None:
    cena = shot_scene(_percepcao({28}), _Chute(28, shooter_track_id=OUTRO))

    assert isinstance(cena, ShotScene)
    assert cena.shooter.track_id == OUTRO
    assert cena.shooter_source == "declared"
    assert cena.anchor_xy == (14.0, -6.0)


def test_sem_bola_perto_de_ninguem_nao_ha_finalizador() -> None:
    cena = shot_scene(_percepcao({28}, bola_em=800.0), _Chute(28))

    assert isinstance(cena, SceneRefusal)
    assert cena.reason == "sem-finalizador"


def test_finalizador_que_some_do_quadro_de_referencia_e_recusa_e_nao_troca_de_pessoa() -> None:
    """No contato a bola esta com o track 7; no unico quadro com campo ele virou o track 70."""
    quadros = list(_percepcao(set()).frames)
    quadros[6] = _quadro(24, campo=True, finalizador=70)

    cena = shot_scene(Perception(clip_id="c", frames=tuple(quadros)), _Chute(28))

    assert isinstance(cena, SceneRefusal)
    assert cena.reason == "finalizador-fora"


def test_percepcao_sem_relogio_e_recusa() -> None:
    cena = shot_scene(Perception(clip_id="c", frames=(_quadro(0, campo=True),)), _Chute(28))

    assert isinstance(cena, SceneRefusal)
    assert cena.reason == "sem-relogio"
