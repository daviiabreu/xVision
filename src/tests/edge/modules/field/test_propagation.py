"""Propagacao da homografia pelo movimento de camera, desligada por padrao."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import cv2
import numpy as np
import pytest

from contracts import Clip, ContractError, FieldGeometry
from edge.graph.runner import assemble
from edge.modules.base import FrameBatch, ModuleOutput
from edge.modules.field.geometry import PITCH_105X68, ImageKeypoint, pitch_to_image
from edge.modules.field.module import PREPROCESS, FieldConfigError, YoloField
from edge.modules.field.motion import camera_motion
from edge.modules.records import check_records

ESCOLHIDOS = (0, 2, 9, 11, 12, 16, 18, 25)
CLIPE = Clip(clip_id="c", path="c.mp4", fps=59.94, frame_count=100, width=1920, height=1080)


def _textura(semente: int = 0) -> np.ndarray[Any, Any]:
    ruido = np.random.default_rng(semente).integers(
        0, 255, (PREPROCESS.height, PREPROCESS.width), dtype=np.uint8
    )
    return cv2.cvtColor(cv2.GaussianBlur(ruido, (0, 0), 2.0), cv2.COLOR_GRAY2BGR)


def _desloca(quadro: np.ndarray[Any, Any], dx: float, dy: float) -> np.ndarray[Any, Any]:
    matriz = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]])
    return cv2.warpAffine(
        quadro, matriz, (quadro.shape[1], quadro.shape[0]), borderMode=cv2.BORDER_REFLECT
    )


def _pontos(
    dx: float = 0.0, dy: float = 0.0, quais: tuple[int, ...] = ESCOLHIDOS
) -> tuple[ImageKeypoint, ...]:
    """Os keypoints no tensor para a homografia fonte (10x + 960, -10y + 540), deslocados."""
    return tuple(
        ImageKeypoint(
            p.name,
            (
                (p.pitch_xy[0] * 10.0 + 960.0) * 0.5 + dx,
                (p.pitch_xy[1] * -10.0 + 540.0) * 0.5 + 2 + dy,
            ),
            0.9,
        )
        for p in (PITCH_105X68.keypoints[i] for i in quais)
    )


class _Respostas:
    def __init__(self, *respostas: tuple[ImageKeypoint, ...] | None) -> None:
        self._respostas = list(respostas)

    def estimate(self, frame: object) -> tuple[ImageKeypoint, ...] | None:
        return self._respostas.pop(0)


def _infer(passos: int, quadros: list[np.ndarray[Any, Any]], *respostas: Any) -> ModuleOutput:
    modulo = YoloField(_Respostas(*respostas))
    if passos:
        modulo.configure({"propagation_steps": passos})
    modulo.load()
    lote = FrameBatch(
        indices=tuple(4 * i for i in range(len(quadros))),
        tensor=np.stack(quadros),
        preprocess=PREPROCESS,
        source_size=(1920, 1080),
    )
    return modulo.infer(lote, {})


def _registro(saida: ModuleOutput, posicao: int) -> dict[str, object]:
    return dict(saida.records[posicao][0])


def test_desligada_por_padrao() -> None:
    base = _textura()

    saida = _infer(0, [base, _desloca(base, 7, 3)], _pontos(), None)

    assert _registro(saida, 1)["source"] == "unavailable"


def test_leva_a_homografia_pelo_movimento_de_camera() -> None:
    """7 e 3 pixels no tensor viram 14 e 6 na fonte, que tem o dobro da largura."""
    base = _textura()

    saida = _infer(2, [base, _desloca(base, 7, 3)], _pontos(), None)

    check_records(saida)
    registro = _registro(saida, 1)
    assert registro["source"] == "propagated"
    assert registro["age_frames"] == 4
    assert registro["reprojection_error_px"] is None
    homografia = cast("tuple[float, ...]", registro["homography"])
    assert pitch_to_image(homografia, (12.0, -8.0)) == pytest.approx((1094.0, 626.0), abs=1.0)


def test_propaga_para_tras_tambem() -> None:
    base = _textura()

    saida = _infer(2, [_desloca(base, -7, -3), base], None, _pontos())

    homografia = cast("tuple[float, ...]", _registro(saida, 0)["homography"])
    assert pitch_to_image(homografia, (12.0, -8.0)) == pytest.approx((1066.0, 614.0), abs=1.0)


def test_corte_de_camera_nao_propaga() -> None:
    saida = _infer(2, [_textura(0), _textura(1)], _pontos(), None)

    assert _registro(saida, 1)["source"] == "unavailable"


def test_fora_do_alcance_nao_propaga_nem_encadeia_propagada() -> None:
    base = _textura()
    quadros = [base, _desloca(base, 3, 1), _desloca(base, 6, 2)]

    saida = _infer(1, quadros, _pontos(), None, None)

    assert _registro(saida, 1)["source"] == "propagated"
    assert _registro(saida, 2)["source"] == "unavailable"


def test_keypoints_do_quadro_que_discordam_recusam_a_propagacao() -> None:
    base = _textura()
    poucos = (0, 2, 9)

    concorda = _infer(2, [base, _desloca(base, 7, 3)], _pontos(), _pontos(7, 3, poucos))
    discorda = _infer(2, [base, _desloca(base, 7, 3)], _pontos(), _pontos(47, 3, poucos))

    assert _registro(concorda, 1)["source"] == "propagated"
    assert cast("float", _registro(concorda, 1)["reprojection_error_p95_px"]) < 3.0
    assert _registro(discorda, 1)["source"] == "unavailable"


def test_a_maioria_dos_keypoints_concordando_basta_mesmo_com_os_errados() -> None:
    """Keypoints de quadro que nao calibrou erram por definicao; a mediana ignora os errados."""
    base = _textura()
    certos = _pontos(7, 3, (0, 2, 9, 11, 12))
    errados = _pontos(67, 3, (16, 18, 25))

    saida = _infer(2, [base, _desloca(base, 7, 3)], _pontos(), certos + errados)

    assert _registro(saida, 1)["source"] == "propagated"


def test_o_quadro_propagado_projeta_jogador_no_assemble() -> None:
    base = _textura()
    campo = _infer(2, [base, _desloca(base, 7, 3)], _pontos(), None)
    jogador = {"box": (1060.0, 500.0, 1100.0, 620.0), "confidence": 0.9}
    jogadores = ModuleOutput("players", campo.frames, (((jogador),),) * len(campo.frames))

    percepcao = assemble(CLIPE, {"field": campo, "players": jogadores})

    quadro = percepcao.frames[1]
    assert quadro.field is not None and quadro.field.source == "propagated"
    assert quadro.players[0].pitch_xy is not None


@pytest.mark.parametrize("valor", [-1, True, 1.5, "2"])
def test_configuracao_invalida_e_recusada(valor: object) -> None:
    with pytest.raises(FieldConfigError, match="propagation_steps"):
        YoloField().configure({"propagation_steps": valor})


def test_a_propagacao_muda_a_impressao_do_cache() -> None:
    ligada = YoloField()
    ligada.configure({"propagation_steps": 8})

    assert ligada.fingerprint() != YoloField().fingerprint()


def _geometria(source: str, age_frames: int) -> FieldGeometry:
    return FieldGeometry(
        homography=(10.0, 0.0, 960.0, 0.0, -10.0, 540.0, 0.0, 0.0, 1.0),
        reprojection_error_px=None,
        reprojection_error_p95_px=None,
        keypoints=(),
        observed_keypoint_count=0,
        inlier_count=0,
        source=source,  # type: ignore[arg-type]
        age_frames=age_frames,
        template_id="pitch-105x68-center-v1",
        reason=None,
    )


def test_o_contrato_aceita_propagada_sem_erro_de_reprojecao() -> None:
    assert _geometria("propagated", 4).age_frames == 4


@pytest.mark.parametrize(("source", "age_frames"), [("propagated", 0), ("detected", 4)])
def test_o_contrato_amarra_idade_a_origem(source: str, age_frames: int) -> None:
    with pytest.raises(ContractError, match="age_frames"):
        _geometria(source, age_frames)


def test_sem_quadro_calibrado_nada_e_propagado() -> None:
    base = _textura()

    saida = _infer(2, [base, _desloca(base, 7, 3)], None, None)

    assert [_registro(saida, k)["source"] for k in (0, 1)] == ["unavailable", "unavailable"]


def test_movimento_sem_textura_ou_com_zoom_forte_e_recusado() -> None:
    liso = np.full((PREPROCESS.height, PREPROCESS.width, 3), 90, dtype=np.uint8)
    base = _textura()
    zoom = cv2.resize(base, None, fx=2.0, fy=2.0)[: PREPROCESS.height, : PREPROCESS.width]

    assert camera_motion(liso, liso) is None
    assert camera_motion(base, zoom) is None
    assert camera_motion(base, _desloca(base, 7, 3)) is not None


@pytest.mark.parametrize(
    ("mediana", "p95", "reason"),
    [(1.0, None, None), (2.0, 1.0, None), (None, None, "insufficient-keypoints")],
)
def test_o_contrato_recusa_propagada_com_erro_ou_motivo_invalido(
    mediana: float | None, p95: float | None, reason: str | None
) -> None:
    with pytest.raises(ContractError, match="propagated"):
        replace(
            _geometria("propagated", 4),
            reprojection_error_px=mediana,
            reprojection_error_p95_px=p95,
            reason=reason,  # type: ignore[arg-type]
        )
