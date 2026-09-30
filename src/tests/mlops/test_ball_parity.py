"""Paridade da bola entre runtimes no contato, com detector falso e sem GPU.

Os testes fecham as formas de a paridade mentir a favor: dois lados vazios contando como iguais, a
segunda configuracao lendo do cache as deteccoes da primeira, e a troca de detectado por interpolado
na mesma posicao passando como o mesmo ponto.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from contracts import BallDetection, Box, Clip
from edge.modules.ball.module import PREPROCESS, YoloBall
from edge.modules.base import FrameBatch, ModuleInputs, ModuleOutput, ModuleSpec, PreprocessSpec
from edge.modules.inference import shared
from mlops.ball_parity import (
    RAIO_PAR_PX,
    LadoDoContato,
    ParidadeError,
    compara,
    escolha_mudou,
    lado_do_contato,
    parear,
)

PEQUENO = PreprocessSpec(64, 64, "bgr", True, "none")


def _bola(x: float, y: float, confianca: float = 0.5, origem: str = "detected") -> BallDetection:
    caixa = Box(x - 7, y - 7, x + 7, y + 7)
    if origem == "interpolated":
        return BallDetection(caixa, 0.0, source=origem, gap_frames=1)
    return BallDetection(caixa, confianca, source=origem)


class _Detector:
    """Uma caixa fixa no espaco do tensor, a mesma em todo quadro."""

    def __init__(self, caixa: Box) -> None:
        self.caixa = caixa

    def detect_raw(self, frame: Any) -> tuple[tuple[Box, float, int], ...]:
        return ((self.caixa, 0.8, 0),)

    def candidates_from(
        self, cruas: Sequence[tuple[Box, float, int]], frame_height: float
    ) -> tuple[BallDetection, ...]:
        return tuple(BallDetection(box=c, confidence=k) for c, k, _ in cruas)

    def detect_candidates(self, frame: Any) -> tuple[BallDetection, ...]:
        return self.candidates_from(self.detect_raw(frame), 0.0)


class _BolaDoMesmoPeso:
    """Duas instancias com a mesma chave de cache e caixas diferentes, como `.pt` e plano."""

    spec = ModuleSpec(
        kind="ball", name="mesmo-peso", version="1", frames="dense", preprocess=PEQUENO
    )

    def __init__(self, x: float) -> None:
        self.x = x

    def _cruas(self, indice: int) -> tuple[tuple[Box, float, int], ...]:
        return shared().get(
            weights="mesmo-peso",
            preprocess=PEQUENO,
            frame_index=indice,
            compute=lambda: ((Box(self.x - 2, 28.0, self.x + 2, 32.0), 0.9, 0),),
        )

    def infer(self, batch: FrameBatch, ctx: ModuleInputs) -> ModuleOutput:
        registros = tuple(
            tuple(
                {"box": (b.x1, b.y1, b.x2, b.y2), "confidence": k, "source": "detected"}
                for b, k, _ in self._cruas(indice)
            )
            for indice in batch.indices
        )
        return ModuleOutput(kind="ball", frames=batch.indices, records=registros)

    def candidatos(self, batch: FrameBatch) -> tuple[tuple[BallDetection, ...], ...]:
        return tuple(
            tuple(BallDetection(b, k) for b, k, _ in self._cruas(i)) for i in batch.indices
        )


def _lote(indices: tuple[int, ...], spec: PreprocessSpec = PREPROCESS) -> FrameBatch:
    tensor = np.zeros((len(indices), spec.height, spec.width, 3), dtype=np.uint8)
    return FrameBatch(
        indices=indices, tensor=tensor, preprocess=spec, source_size=(1920, 1080), source_fps=30.0
    )


def test_pares_pelo_centro_mais_proximo_ate_o_raio() -> None:
    referencia = [_bola(100, 100, 0.9), _bola(103, 100, 0.1), _bola(600, 600, 0.4)]
    comparada = [_bola(101, 100), _bola(604, 600), _bola(900, 900)]

    pares, so_ref, so_cmp = parear(referencia, comparada, RAIO_PAR_PX)

    assert [(a.box.x1 + 7, b.box.x1 + 7) for a, b in pares] == [(100, 101), (600, 604)]
    assert (so_ref, so_cmp) == (1, 1)
    assert parear([_bola(0, 0)], [_bola(4.01, 0)], RAIO_PAR_PX)[0] == ()


def test_relatorio_traz_maximo_e_p95_da_confianca_e_do_centro_e_os_sem_par() -> None:
    referencia = [
        LadoDoContato(
            "a", 8, (_bola(100, 100, 0.50), _bola(300, 300, 0.30), _bola(500, 500)), _bola(100, 100)
        ),
        LadoDoContato("b", 4, (), None),
    ]
    comparada = [
        LadoDoContato(
            "a",
            8,
            (_bola(100.6, 100.8, 0.51), _bola(300, 302, 0.28), _bola(800, 800, 0.9)),
            _bola(100.6, 100.8),
        ),
        LadoDoContato("b", 4, (), None),
    ]

    resumo = compara(referencia, comparada).to_dict()

    assert (resumo["pareados"], resumo["so_na_referencia"], resumo["so_na_comparada"]) == (2, 1, 1)
    assert (resumo["candidatos_referencia"], resumo["candidatos_comparada"]) == (3, 3)
    assert resumo["confianca_diferenca_max"] == pytest.approx(0.02)
    assert resumo["confianca_diferenca_p95"] == pytest.approx(0.0195)
    assert resumo["centro_distancia_max_px"] == pytest.approx(2.0)
    assert resumo["centro_distancia_p95_px"] == pytest.approx(1.95)
    assert resumo["escolha_mudou"] == 0
    assert resumo["dentro_do_criterio"] is False


def test_criterio_passa_so_com_par_dentro_dos_limites_e_nenhuma_escolha_mudada() -> None:
    def um(ref: BallDetection, cmp: BallDetection) -> bool:
        return compara(
            [LadoDoContato("a", 0, (ref,), ref)], [LadoDoContato("a", 0, (cmp,), cmp)]
        ).dentro_do_criterio

    assert um(_bola(100, 100, 0.50), _bola(100.5, 100, 0.505)) is True
    assert um(_bola(100, 100, 0.50), _bola(101.5, 100, 0.50)) is False
    assert um(_bola(100, 100, 0.50), _bola(100, 100, 0.53)) is False
    assert um(_bola(100, 100), _bola(100, 100, origem="interpolated")) is False


def test_sem_par_nenhum_nao_e_paridade() -> None:
    """Dois lados sem candidato concordam em tudo, e nao provam nada sobre o runtime."""
    lados = [LadoDoContato("a", 0, (), None)]

    relatorio = compara(lados, lados)

    assert relatorio.to_dict()["pareados"] == 0
    assert relatorio.dentro_do_criterio is False


def test_contato_sem_quadro_impede_paridade_mesmo_com_outro_contato_pareado() -> None:
    ponto = _bola(100, 100)
    lados = [
        LadoDoContato("medido", 8, (ponto,), ponto),
        LadoDoContato("sem-quadro", None, (), None),
    ]

    resumo = compara(lados, lados).to_dict()

    assert resumo["pareados"] == 1
    assert resumo["sem_quadro_no_contato"] == 1
    assert resumo["escolha_mudou"] == 0
    assert resumo["dentro_do_criterio"] is False


def test_mudanca_do_ponto_escolhido() -> None:
    ponto = _bola(100, 100)

    assert escolha_mudou(None, None, RAIO_PAR_PX) is False
    assert escolha_mudou(None, ponto, RAIO_PAR_PX) is True
    assert escolha_mudou(ponto, None, RAIO_PAR_PX) is True
    assert escolha_mudou(ponto, _bola(103, 100), RAIO_PAR_PX) is False
    assert escolha_mudou(ponto, _bola(105, 100), RAIO_PAR_PX) is True
    assert escolha_mudou(ponto, _bola(100, 100, origem="interpolated"), RAIO_PAR_PX) is True


def test_comparacao_recusa_contatos_diferentes_e_raio_invalido() -> None:
    a = [LadoDoContato("a", 8, (), None)]

    with pytest.raises(ParidadeError, match="mesmos quadros"):
        compara(a, [LadoDoContato("a", 12, (), None)])
    with pytest.raises(ParidadeError, match="mesmos quadros"):
        compara(a, [LadoDoContato("b", 8, (), None)])
    with pytest.raises(ParidadeError, match="raio"):
        compara(a, a, raio_px=0.0)


def test_lado_le_o_quadro_do_contato_com_os_candidatos_e_o_ponto_do_modulo() -> None:
    meio_x, meio_y = PREPROCESS.width / 2.0, PREPROCESS.height / 2.0
    bola = YoloBall(_Detector(Box(meio_x - 6, meio_y - 6, meio_x + 6, meio_y + 6)))
    bola.load()
    clip = Clip("c", "c.mp4", 30.0, 16, 1920, 1080)

    lado = lado_do_contato(bola, clip, _lote((0, 4, 8, 12)), contato=9)

    assert lado.frame_index == 8
    (candidato,) = lado.candidatos
    assert lado.escolhido is not None
    assert lado.escolhido.source == "detected"
    assert lado.escolhido.box == candidato.box
    assert (candidato.box.x1 + candidato.box.x2) / 2 == pytest.approx(960.0, abs=1.0)


def test_lado_sem_quadro_perto_do_contato_fica_sem_candidato() -> None:
    bola = _BolaDoMesmoPeso(20.0)
    clip = Clip("c", "c.mp4", 30.0, 100, 64, 64)

    lado = lado_do_contato(bola, clip, _lote((0, 4), PEQUENO), contato=90)

    assert (lado.frame_index, lado.candidatos, lado.escolhido) == (None, (), None)


def test_a_segunda_configuracao_nao_le_do_cache_as_deteccoes_da_primeira() -> None:
    """O `.pt` em fp32 e o plano dele casam na chave do cache, e a paridade sairia perfeita."""
    clip = Clip("c", "c.mp4", 30.0, 8, 64, 64)
    lote = _lote((0, 4), PEQUENO)

    primeira = lado_do_contato(_BolaDoMesmoPeso(20.0), clip, lote, contato=4)
    segunda = lado_do_contato(_BolaDoMesmoPeso(30.0), clip, lote, contato=4)

    resumo = compara([primeira], [segunda]).to_dict()
    assert resumo["pareados"] == 0
    assert resumo["escolha_mudou"] == 1
