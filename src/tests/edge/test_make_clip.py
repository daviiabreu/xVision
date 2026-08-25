"""A fixture canonica e o gabarito de cinco cards, entao ela precisa ser deterministica.

Nenhum clipe da Footstats entra no repositorio. O video e escrito em codigo na hora do teste,
e a posicao de cada elemento vem do gabarito em metros, o que permite afirmar distancia e
angulo esperados mais tarde em vez de conferir que "rodou".
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from edge.testing.make_clip import BALL_COLOR, ball_pixel, make_synthetic_clip, render_frame
from edge.testing.pitch import BALL_RADIUS_PX, clip_truth


def _propriedades(caminho: Path) -> tuple[int, int, int, int]:
    captura = cv2.VideoCapture(str(caminho))
    try:
        assert captura.isOpened(), f"nao abriu {caminho}"
        return (
            int(captura.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(captura.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            round(captura.get(cv2.CAP_PROP_FPS)),
        )
    finally:
        captura.release()


def test_synthetic_clip_has_expected_fps_duration_and_size(tmp_path: Path) -> None:
    gabarito = clip_truth()

    destino = make_synthetic_clip(tmp_path / "clipe.mp4")

    assert destino.is_file()
    assert _propriedades(destino) == (
        gabarito.frame_count,
        gabarito.width,
        gabarito.height,
        gabarito.fps,
    )


def test_synthetic_clip_ball_position_is_known_for_every_frame() -> None:
    gabarito = clip_truth()
    esperado = np.array(BALL_COLOR, dtype=np.uint8)

    for indice in range(gabarito.frame_count):
        quadro = render_frame(indice)
        u, v = ball_pixel(indice)
        assert bool((quadro[v, u] == esperado).all()), f"quadro {indice} sem bola em ({u}, {v})"


def test_no_other_element_uses_the_ball_color() -> None:
    quadro = render_frame(0)
    u, v = ball_pixel(0)
    pintados = int(((quadro == np.array(BALL_COLOR, dtype=np.uint8)).all(axis=2)).sum())
    margem = round(BALL_RADIUS_PX) + 1
    quadro[v - margem : v + margem + 1, u - margem : u + margem + 1] = 0

    restantes = int(((quadro == np.array(BALL_COLOR, dtype=np.uint8)).all(axis=2)).sum())
    assert pintados > 0
    assert restantes == 0


def test_the_ball_moves_between_the_first_and_the_last_frame() -> None:
    gabarito = clip_truth()

    assert ball_pixel(0) != ball_pixel(gabarito.frame_count - 1)


def test_synthetic_clip_is_byte_identical_across_runs(tmp_path: Path) -> None:
    primeiro = make_synthetic_clip(tmp_path / "a.mp4", seed=7)
    segundo = make_synthetic_clip(tmp_path / "b.mp4", seed=7)

    assert hashlib.sha256(primeiro.read_bytes()).hexdigest() == (
        hashlib.sha256(segundo.read_bytes()).hexdigest()
    )


def test_different_seeds_produce_different_clips(tmp_path: Path) -> None:
    um = make_synthetic_clip(tmp_path / "um.mp4", seed=0)
    outro = make_synthetic_clip(tmp_path / "outro.mp4", seed=3)

    assert um.read_bytes() != outro.read_bytes()


def test_the_destination_directory_is_created(tmp_path: Path) -> None:
    destino = make_synthetic_clip(tmp_path / "fundo" / "do" / "poco" / "clipe.mp4")

    assert destino.is_file()
