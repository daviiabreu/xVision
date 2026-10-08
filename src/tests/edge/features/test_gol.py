from __future__ import annotations

from contracts import BallDetection, Box, FieldGeometry, FramePerception, Perception
from edge.features.gol import (
    GOL_MEIA_LARGURA_M,
    gol_mais_proximo,
    gol_pelo_rumo_da_bola,
    gols,
    nome_do_gol,
)

FPS = 30.0
CONTATO = 30
HOMOGRAPHY = (10.0, 0.0, 600.0, 0.0, -10.0, 400.0, 0.0, 0.0, 1.0)


def _field() -> FieldGeometry:
    return FieldGeometry(
        homography=HOMOGRAPHY,
        reprojection_error_px=2.0,
        reprojection_error_p95_px=3.0,
        keypoints=(),
        observed_keypoint_count=6,
        inlier_count=6,
        source="detected",
        age_frames=0,
        template_id="pitch-105x68-center-v1",
        reason=None,
    )


def _bola(pitch_xy: tuple[float, float] | None, *, detectada: bool = True) -> BallDetection:
    return BallDetection(
        box=Box(100.0, 100.0, 106.0, 106.0),
        confidence=0.8 if detectada else 0.0,
        pitch_xy=pitch_xy,
        source="detected" if detectada else "interpolated",
        gap_frames=0 if detectada else 2,
    )


def _perception(
    trajetoria: tuple[tuple[float, float], ...],
    *,
    detectada: bool = True,
    desde: int = CONTATO,
) -> Perception:
    frames = tuple(
        FramePerception(
            frame_index=desde + passo,
            timestamp_s=(desde + passo) / FPS,
            players=(),
            ball=_bola(pitch_xy, detectada=detectada),
            field=_field(),
        )
        for passo, pitch_xy in enumerate(trajetoria)
    )
    return Perception(clip_id="clipe-gol", frames=frames)


def test_os_dois_gols_saem_do_template() -> None:
    assert gols() == ((-52.5, 0.0), (52.5, 0.0))
    assert GOL_MEIA_LARGURA_M == 3.66


def test_o_gol_mais_proximo_e_o_mais_proximo() -> None:
    assert gol_mais_proximo((40.0, 0.0)) == (52.5, 0.0)
    assert gol_mais_proximo((-40.0, 10.0)) == (-52.5, 0.0)
    assert nome_do_gol((52.5, 0.0)) == "direito"
    assert nome_do_gol((-52.5, 0.0)) == "esquerdo"


def test_a_bola_indo_para_a_direita_escolhe_o_gol_direito() -> None:
    perception = _perception(((30.0, 0.0), (36.0, 0.0), (42.0, 0.0)))
    assert gol_pelo_rumo_da_bola(perception, CONTATO, CONTATO / FPS) == (52.5, 0.0)


def test_a_bola_indo_para_a_esquerda_escolhe_o_gol_esquerdo() -> None:
    """O criterio nao e a distancia: a bola sai de um ponto mais proximo do gol direito."""
    perception = _perception(((30.0, 0.0), (24.0, 0.0), (18.0, 0.0)))
    assert gol_pelo_rumo_da_bola(perception, CONTATO, CONTATO / FPS) == (-52.5, 0.0)


def test_bola_interpolada_nao_escolhe() -> None:
    perception = _perception(((30.0, 0.0), (36.0, 0.0), (42.0, 0.0)), detectada=False)
    assert gol_pelo_rumo_da_bola(perception, CONTATO, CONTATO / FPS) is None


def test_um_quadro_so_nao_da_rumo() -> None:
    perception = _perception(((30.0, 0.0),))
    assert gol_pelo_rumo_da_bola(perception, CONTATO, CONTATO / FPS) is None


def test_deslocamento_abaixo_do_piso_nao_da_rumo() -> None:
    """Meio metro entre o primeiro e o ultimo quadro e ruido de projecao, nao rumo de chute."""
    perception = _perception(((30.0, 0.0), (30.2, 0.0), (30.4, 0.0)))
    assert gol_pelo_rumo_da_bola(perception, CONTATO, CONTATO / FPS) is None


def test_bola_perpendicular_aos_dois_gols_nao_escolhe() -> None:
    """Bola pela lateral nao alinha com baliza nenhuma, e o criterio se declara ausente."""
    perception = _perception(((0.0, 0.0), (0.0, 10.0), (0.0, 20.0)))
    assert gol_pelo_rumo_da_bola(perception, CONTATO, CONTATO / FPS) is None


def test_quadro_antes_do_contato_nao_entra() -> None:
    """A bola antes do pe anda na direcao do drible, e nao na do chute."""
    perception = _perception(((30.0, 0.0), (36.0, 0.0), (42.0, 0.0)), desde=CONTATO - 10)
    assert gol_pelo_rumo_da_bola(perception, CONTATO, CONTATO / FPS) is None


def test_quadro_tarde_demais_nao_entra() -> None:
    """Passado um segundo a bola pode ter batido em alguem, e o rumo deixa de ser o do chute."""
    longe = tuple(
        FramePerception(
            frame_index=CONTATO + passo,
            timestamp_s=(CONTATO + passo) / FPS + 2.0,
            players=(),
            ball=_bola(pitch_xy),
            field=_field(),
        )
        for passo, pitch_xy in enumerate(((30.0, 0.0), (36.0, 0.0), (42.0, 0.0)))
    )
    perception = Perception(clip_id="clipe-gol", frames=longe)
    assert gol_pelo_rumo_da_bola(perception, CONTATO, CONTATO / FPS) is None


def test_sem_pitch_xy_nao_escolhe() -> None:
    frames = (
        FramePerception(
            frame_index=CONTATO,
            timestamp_s=CONTATO / FPS,
            players=(),
            ball=_bola(None),
            field=_field(),
        ),
    )
    assert gol_pelo_rumo_da_bola(Perception("clipe-gol", frames), CONTATO, CONTATO / FPS) is None
