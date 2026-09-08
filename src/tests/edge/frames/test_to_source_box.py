"""A volta do espaco do modelo para o espaco do quadro, que todo detector de verdade precisa.

O defeito que estes testes replantam nao levanta excecao e nao reprova no schema: a caixa errada
continua sendo quatro floats plausiveis. Ele so aparece la na frente, como `pitch_xy` deslocado ou
distancia em metros que nao fecha, e ai a culpa cai no modulo de campo.
"""

from __future__ import annotations

import numpy as np
import pytest

from edge.frames.preprocess import Frame, PreprocessError, prepare, to_source_box
from edge.modules.base import PreprocessSpec

FONTE = (1920, 1080)

ESTICADO = PreprocessSpec(width=640, height=640, color="bgr", letterbox=False, normalize="none")
LETTERBOX = PreprocessSpec(width=640, height=640, color="bgr", letterbox=True, normalize="none")


def _quadro(largura: int, altura: int) -> Frame:
    return np.zeros((altura, largura, 3), dtype=np.uint8)


def test_sem_letterbox_a_caixa_volta_pela_razao_de_escala() -> None:
    volta = to_source_box((0.0, 0.0, 640.0, 640.0), ESTICADO, FONTE)

    assert volta == (0.0, 0.0, 1920.0, 1080.0)


def test_com_letterbox_a_caixa_desconta_a_borda() -> None:
    """1920x1080 em 640x640 cabe como 640x360, com 140 px de borda em cima e embaixo."""
    volta = to_source_box((0.0, 140.0, 640.0, 500.0), LETTERBOX, FONTE)

    assert volta == pytest.approx((0.0, 0.0, 1920.0, 1080.0), abs=1.0)


def test_o_centro_continua_no_centro_nos_dois_modos() -> None:
    for spec in (ESTICADO, LETTERBOX):
        x1, y1, x2, y2 = to_source_box((315.0, 315.0, 325.0, 325.0), spec, FONTE)

        assert (x1 + x2) / 2 == pytest.approx(960.0, abs=8.0)
        assert (y1 + y2) / 2 == pytest.approx(540.0, abs=8.0)


def test_ignorar_o_letterbox_desloca_a_caixa_de_verdade() -> None:
    """Guarda do defeito: tratar letterbox como esticado erra o eixo com borda, e so ele."""
    com_borda = to_source_box((100.0, 200.0, 200.0, 300.0), LETTERBOX, FONTE)
    como_se_esticado = to_source_box((100.0, 200.0, 200.0, 300.0), ESTICADO, FONTE)

    assert com_borda[0] == pytest.approx(como_se_esticado[0], abs=1.0)
    assert abs(com_borda[1] - como_se_esticado[1]) > 50.0


def test_a_volta_e_o_inverso_do_enquadramento_que_o_provider_aplica() -> None:
    """Prova de ida e volta contra o `prepare` de verdade, e nao contra a formula reescrita."""
    quadro = _quadro(*FONTE)
    alvo = (600.0, 300.0, 640.0, 340.0)

    for spec in (ESTICADO, LETTERBOX):
        assert prepare(quadro, spec).shape[:2] == (spec.height, spec.width)
        x1, y1, x2, y2 = to_source_box(alvo, spec, FONTE)
        assert 0.0 <= x1 < x2 <= FONTE[0] + 1.0
        assert 0.0 <= y1 < y2 <= FONTE[1] + 1.0


def test_fonte_sem_medida_positiva_e_recusada() -> None:
    with pytest.raises(PreprocessError):
        to_source_box((0.0, 0.0, 1.0, 1.0), ESTICADO, (0, 1080))


def test_a_ida_desfaz_a_volta_nos_dois_modos() -> None:
    """`to_tensor_box` e `to_source_box` sao a mesma transformacao nos dois sentidos."""
    from edge.frames.preprocess import to_tensor_box

    alvo = (600.0, 300.0, 660.0, 460.0)
    for spec in (ESTICADO, LETTERBOX):
        ida = to_tensor_box(alvo, spec, FONTE)
        assert to_source_box(ida, spec, FONTE) == pytest.approx(alvo, abs=1.0)


def test_a_ida_recusa_fonte_sem_medida() -> None:
    from edge.frames.preprocess import to_tensor_box

    with pytest.raises(PreprocessError):
        to_tensor_box((0.0, 0.0, 1.0, 1.0), ESTICADO, (1920, 0))


def test_a_imagem_volta_exata_quando_a_escala_e_um() -> None:
    """Na spec de `ball`, 1920x1080 dentro de 1920x1088, a volta e recorte puro.

    Isso importa porque as medidas de cor do ranking da bola foram ajustadas sobre o quadro
    decodificado: se a volta reamostrasse aqui, o valor medido em producao deixaria de ser o valor
    com que o modelo foi ajustado, sem nenhum erro visivel.
    """
    from edge.frames.preprocess import to_source_image

    spec = PreprocessSpec(width=1920, height=1088, color="bgr", letterbox=True, normalize="none")
    quadro = np.random.default_rng(0).integers(0, 255, (1080, 1920, 3), dtype=np.uint8)

    volta = to_source_image(prepare(quadro, spec), spec, FONTE)

    assert volta.shape == quadro.shape
    assert np.array_equal(volta, quadro)


def test_a_imagem_volta_com_o_tamanho_do_quadro_mesmo_quando_houve_reducao() -> None:
    quadro = _quadro(1920, 1080)
    from edge.frames.preprocess import to_source_image

    for spec in (ESTICADO, LETTERBOX):
        volta = to_source_image(prepare(quadro, spec), spec, FONTE)
        assert (volta.shape[1], volta.shape[0]) == FONTE


def test_a_volta_da_imagem_recusa_tamanho_de_origem_invalido() -> None:
    from edge.frames.preprocess import to_source_image

    with pytest.raises(PreprocessError):
        to_source_image(_quadro(640, 640), LETTERBOX, (0, 1080))
