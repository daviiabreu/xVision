"""A casca `PerceptionModule` em volta do detector e do rastreador do Gabriel.

O nucleo dele ja tem teste proprio, portado junto em `test_ball_tracker.py` e
`test_ball_shape_filter.py`. Estes aqui cobram so o que a passagem para a fundacao acrescentou: o
espaco da caixa, o ponto interpolado que derrubaria o `assemble`, e a ausencia declarada.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from contracts import BallDetection, Box
from edge.modules.ball import DetectorConfig
from edge.modules.ball.detector import AjusteDesconhecido
from edge.modules.ball.module import PREPROCESS, YoloBall, chave_do_cache
from edge.modules.base import FrameBatch, ModuleNotLoaded, PerceptionModule
from edge.modules.conformance import ModuleConformance
from edge.modules.records import check_records

FONTE = (1920, 1080)


class _DetectorFalso:
    """Devolve sempre a mesma caixa, no espaco do tensor pre-processado."""

    def __init__(self, caixa: Box | None, confianca: float = 0.8) -> None:
        self._caixa = caixa
        self._confianca = confianca

    def detect_raw(self, frame: Any) -> tuple[tuple[Box, float, int], ...]:
        if self._caixa is None:
            return ()
        return ((self._caixa, self._confianca, 32),)

    def candidates_from(
        self, cruas: Sequence[tuple[Box, float, int]], frame_height: float
    ) -> tuple[BallDetection, ...]:
        return tuple(BallDetection(box=c, confidence=k) for c, k, _ in cruas)

    def detect_candidates(self, frame: Any) -> tuple[BallDetection, ...]:
        return self.candidates_from(self.detect_raw(frame), 0.0)


def _lote(indices: tuple[int, ...], fonte: tuple[int, int] | None = FONTE) -> FrameBatch:
    tensor = np.zeros((len(indices), PREPROCESS.height, PREPROCESS.width, 3), dtype=np.uint8)
    return FrameBatch(indices=indices, tensor=tensor, preprocess=PREPROCESS, source_size=fonte)


def _centro(caixa: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = caixa
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class TestYoloBall(ModuleConformance):
    @pytest.fixture
    def module(self) -> PerceptionModule:
        return YoloBall()


def test_sem_detector_a_especie_sai_sem_posicao() -> None:
    """Peso ausente nao pode virar bola no canto do campo: sai sem registro, e o cliente ve."""
    modulo = YoloBall()
    modulo.load()

    saida = modulo.infer(_lote((0, 12, 24)), {})

    assert saida.frames == (0, 12, 24)
    assert saida.records == ((), (), ())


def test_a_caixa_volta_para_o_espaco_do_quadro_original() -> None:
    """O detector ve 1280x736 e o `records.py` cobra pixel do quadro original.

    Sem a volta, a bola no centro do tensor viraria bola em (640, 368) para a feature, que num
    quadro de 1920x1080 e o quadrante superior esquerdo.
    """
    meio = PREPROCESS.width / 2.0
    altura_conteudo = PREPROCESS.width * FONTE[1] / FONTE[0]
    meio_y = PREPROCESS.height / 2.0
    modulo = YoloBall(_DetectorFalso(Box(meio - 6, meio_y - 6, meio + 6, meio_y + 6)))
    modulo.load()

    saida = modulo.infer(_lote((0,)), {})

    (registro,) = saida.records[0]
    x, y = _centro(registro["box"])  # type: ignore[arg-type]
    assert altura_conteudo < PREPROCESS.height
    assert x == pytest.approx(FONTE[0] / 2.0, abs=4.0)
    assert y == pytest.approx(FONTE[1] / 2.0, abs=4.0)


def test_caixa_grande_demais_e_reprovada_pelo_filtro_de_forma() -> None:
    """O filtro do Gabriel roda depois da volta, senao a fracao de altura mede outra coisa."""
    modulo = YoloBall(_DetectorFalso(Box(100.0, 100.0, 700.0, 700.0)))
    modulo.load()

    saida = modulo.infer(_lote((0,)), {})

    assert saida.records == ((),)


def test_oregistro_de_cumpre_o_que_a_especie_promete() -> None:
    meio = PREPROCESS.width / 2.0
    modulo = YoloBall(_DetectorFalso(Box(meio - 6, meio - 6, meio + 6, meio + 6)))
    modulo.load()

    check_records(modulo.infer(_lote((0, 12)), {}))


def test_ponto_interpolado_sai_com_caixa_e_nao_com_none() -> None:
    """`assemble` faz `Box(*registro["box"])` fora do try da etapa: um None derruba o clipe.

    O `TrackPoint` interpolado da branch de origem sai com `box=None`, entao a casca precisa
    sintetizar a caixa em volta do centro e declarar que ela e interpolada.
    """
    from edge.modules.ball.module import registro_de
    from edge.modules.ball.tracking import TrackPoint

    ponto = TrackPoint(
        frame_index=4,
        timestamp_s=4.0,
        source="interpolated",
        center=(100.0, 200.0),
        box=None,
        confidence=None,
    )

    (registro,) = registro_de(ponto)

    assert registro["source"] == "interpolated"
    assert _centro(registro["box"]) == (100.0, 200.0)  # type: ignore[arg-type]
    assert isinstance(registro["confidence"], float)


def test_lacuna_longa_nao_vira_caixa_de_zeros() -> None:
    from edge.modules.ball.module import registro_de
    from edge.modules.ball.tracking import TrackPoint

    perdido = TrackPoint(
        frame_index=4, timestamp_s=4.0, source="missing", center=None, box=None, confidence=None
    )

    assert registro_de(perdido) == ()
    assert registro_de(None) == ()


def test_infer_sem_load_recusa() -> None:
    with pytest.raises(ModuleNotLoaded):
        YoloBall().infer(_lote((0,)), {})


def test_lote_sem_tamanho_de_origem_nao_estoura() -> None:
    """Lote montado a mao nao carrega `source_size`, e o modulo nao pode morrer por isso."""
    meio = PREPROCESS.width / 2.0
    modulo = YoloBall(_DetectorFalso(Box(meio - 6, meio - 6, meio + 6, meio + 6)))
    modulo.load()

    saida = modulo.infer(_lote((0,), fonte=None), {})

    assert len(saida.records[0]) == 1


def test_close_solta_o_detector() -> None:
    modulo = YoloBall(_DetectorFalso(Box(0.0, 0.0, 10.0, 10.0)))
    modulo.load()
    modulo.close()

    with pytest.raises(ModuleNotLoaded):
        modulo.infer(_lote((0,)), {})


def test_o_detector_yolo_so_e_construido_quando_ha_configuracao() -> None:
    """Sem `DetectorConfig` o `load` nao importa ultralytics, e a conformidade roda sem o extra."""
    modulo = YoloBall()
    modulo.load()

    assert modulo.infer(_lote((0,)), {}).records == ((),)
    assert DetectorConfig(weights="x.pt").ball_class_id == 32


ASPECTO_BORRADO = 2.05
"""Aspecto de bola borrada pelo chute, dentro da faixa medida no clipe 4996030 (1,75 a 2,09)."""


def _caixa_com_aspecto(aspecto: float, lado_maior: float) -> Box:
    """Caixa no espaco do tensor.

    O letterbox escala igual nos dois eixos, entao o aspecto sobrevive a volta para o quadro
    original e o teste pode escolher exatamente o numero que quer medir.
    """
    meio_x, meio_y = PREPROCESS.width / 2.0, PREPROCESS.height / 2.0
    altura = lado_maior / aspecto
    return Box(
        meio_x - lado_maior / 2, meio_y - altura / 2, meio_x + lado_maior / 2, meio_y + altura / 2
    )


def test_o_limite_de_aspecto_do_toml_alcanca_o_filtro_no_espaco_do_quadro() -> None:
    """O filtro de forma roda duas vezes, e o ajuste precisa valer nas duas.

    A segunda passada, ja em pixel do quadro original, montava um `DetectorConfig` do zero quando o
    modulo nao tinha recebido configuracao por parametro. Com isso um `max_aspect` declarado no
    TOML valia so na primeira passada, a caixa borrada era descartada de novo pelo padrao 2,0, e
    nada no arquivo de configuracao indicava o motivo.
    """
    modulo = YoloBall(_DetectorFalso(_caixa_com_aspecto(ASPECTO_BORRADO, 27.0)))
    modulo.configure({"max_aspect": 2.2})
    modulo.load()

    assert len(modulo.infer(_lote((0,)), {}).records[0]) == 1


def test_sem_ajuste_o_padrao_de_hoje_continua_valendo() -> None:
    """O padrao 2,0 recusa a mesma caixa: e o que prova que o teste acima mede o ajuste."""
    modulo = YoloBall(_DetectorFalso(_caixa_com_aspecto(ASPECTO_BORRADO, 27.0)))
    modulo.load()

    assert modulo.infer(_lote((0,)), {}).records[0] == ()


def test_ajuste_desconhecido_no_toml_levanta_em_vez_de_ficar_calado() -> None:
    """`confidence` no lugar de `conf` deixaria o detector no padrao e a medicao mentiria."""
    modulo = YoloBall()
    modulo.configure({"confidence": 0.10})

    with pytest.raises(AjusteDesconhecido, match="confidence"):
        modulo.load()


def test_a_chave_do_cache_compartilhado_separa_conf_e_imgsz() -> None:
    """O `PreprocessSpec` nao carrega `conf`, e `ball` e `players` usam o mesmo peso.

    Com a chave sendo so o caminho do peso, a passada de `players` a 0,25 servia `ball` a 0,10 e o
    limiar novo nao mudava um candidato sequer.
    """
    peso = "var/models/abc/yolov8n.pt"
    baixa = chave_do_cache(DetectorConfig(weights=peso, conf=0.10))
    alta = chave_do_cache(DetectorConfig(weights=peso, conf=0.25))
    outro_tamanho = chave_do_cache(DetectorConfig(weights=peso, conf=0.10, imgsz=1920))

    assert baixa != alta
    assert baixa != outro_tamanho
    assert peso not in (baixa, alta, outro_tamanho)


def test_com_ajustes_recusa_valor_que_nao_e_numero() -> None:
    with pytest.raises(AjusteDesconhecido, match="conf"):
        DetectorConfig(weights="x.pt").com_ajustes({"conf": "0.10"})


def test_com_ajustes_aplica_conf_e_preserva_o_resto() -> None:
    ajustada = DetectorConfig(weights="x.pt", device="mps").com_ajustes({"conf": 0.10})

    assert ajustada.conf == pytest.approx(0.10)
    assert ajustada.weights == "x.pt"
    assert ajustada.device == "mps"
