"""A folha de contato: legivel, numerada, e escrita sem depender de janela.

A ferramenta existe porque `cv2.imshow` nao roda no wheel de Linux que o lock trava, entao um
teste que dependesse de GUI teria o mesmo defeito que a ferramenta evita.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from contracts import Clip
from edge.frames.provider import probe
from edge.testing.make_clip import make_synthetic_clip
from mlops.sheets import CELULA, COLUNAS, coarse_indices, contact_sheet, fine_indices


@pytest.fixture(scope="module")
def clipe(tmp_path_factory: pytest.TempPathFactory) -> Clip:
    caminho = make_synthetic_clip(tmp_path_factory.mktemp("folha") / "synthetic.mp4")
    return probe(caminho, clip_id="folha")


def test_a_folha_sai_em_png_com_uma_celula_por_quadro(clipe: Clip, tmp_path: Path) -> None:
    indices = (0, 2, 4, 6, 8, 10, 12)

    caminho = contact_sheet(clipe, indices, tmp_path / "folha.png")

    imagem = cv2.imread(str(caminho))
    linhas = (len(indices) + COLUNAS - 1) // COLUNAS
    assert imagem is not None
    assert imagem.shape[1] == COLUNAS * CELULA.width
    assert imagem.shape[0] == linhas * (CELULA.height + 22)


def test_uma_folha_sem_quadro_nenhum_e_recusada(clipe: Clip, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        contact_sheet(clipe, (), tmp_path / "vazia.png")


def test_a_folha_grossa_cobre_o_clipe_inteiro(clipe: Clip) -> None:
    indices = coarse_indices(clipe, fps=5.0)

    assert indices[0] == 0
    assert indices[-1] < clipe.frame_count
    assert len(indices) < clipe.frame_count


def test_a_folha_fina_e_contigua_em_torno_da_ancora(clipe: Clip) -> None:
    """Achar o quadro exato pede taxa nativa, e uma folha unica nativa teria 400 celulas."""
    indices = fine_indices(clipe, ancora=15, raio=5)

    assert indices == tuple(range(10, 21))


def test_a_folha_fina_nao_sai_do_clipe(clipe: Clip) -> None:
    inicio = fine_indices(clipe, ancora=0, raio=10)
    fim = fine_indices(clipe, ancora=clipe.frame_count - 1, raio=10)

    assert inicio[0] == 0
    assert fim[-1] == clipe.frame_count - 1
