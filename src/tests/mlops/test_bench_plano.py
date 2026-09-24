"""O banco de planos: tempo por passada e concordancia das deteccoes com o plano de referencia."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from cli import app
from edge.runtime import RuntimeModel
from mlops.bench_plano import (
    BenchError,
    Deteccao,
    compara,
    concordancia,
    deteccoes,
    entrada_do_plano,
    por_centro,
    prepara,
)

ANCORAS = 6


def _plano(pasta: Path, nome: str, forma: list[int] | None = None, entradas: int = 1) -> Path:
    plano = pasta / f"{nome}.engine"
    plano.write_bytes(b"plano")
    forma = forma or [1, 3, 64, 96]
    relatorio = {"inputs": [{"name": "images", "shape": forma}] * entradas, "precisao": nome}
    plano.with_suffix(".run.json").write_text(json.dumps(relatorio))
    return plano


def _saida(caixas: list[tuple[float, float, float, float, float, float]]) -> np.ndarray[Any, Any]:
    """Saida crua de um detector de duas classes: por ancora, cx, cy, w, h, score0, score1."""
    matriz = np.zeros((1, 6, ANCORAS), dtype=np.float32)
    for i, linha in enumerate(caixas):
        matriz[0, :, i] = linha
    return matriz


def test_a_entrada_vem_do_relatorio_de_construcao(tmp_path: Path) -> None:
    assert entrada_do_plano(_plano(tmp_path, "fp32")) == ("images", (1, 3, 64, 96))


def test_plano_sem_relatorio_ou_com_varias_entradas_e_recusado(tmp_path: Path) -> None:
    with pytest.raises(BenchError, match="build_engine"):
        entrada_do_plano(tmp_path / "solto.engine")
    with pytest.raises(BenchError, match="2 entradas"):
        entrada_do_plano(_plano(tmp_path, "duplo", entradas=2))


def test_o_letterbox_cabe_na_entrada_e_normaliza() -> None:
    quadro = np.full((108, 192, 3), 255, dtype=np.uint8)

    lote = prepara(quadro, 64, 96)

    assert lote.shape == (1, 3, 64, 96)
    assert lote.dtype == np.float32
    assert float(lote.min()) >= 0.0 and float(lote.max()) <= 1.0
    assert float(lote[0, :, 0, 0].mean()) == pytest.approx(114 / 255, abs=1e-3)


def test_a_decodificacao_suprime_duplicata_e_separa_classes() -> None:
    saida = _saida(
        [
            (50, 50, 20, 20, 0.9, 0.0),
            (51, 50, 20, 20, 0.8, 0.0),
            (10, 10, 4, 4, 0.0, 0.7),
            (80, 80, 4, 4, 0.1, 0.1),
        ]
    )

    achadas = deteccoes(saida)

    assert len(achadas) == 2
    assert achadas[0].caixa == pytest.approx((40.0, 40.0, 60.0, 60.0))
    assert {d.classe for d in achadas} == {0, 1}


def test_a_cabeca_de_pose_traz_os_pontos() -> None:
    saida = np.zeros((1, 4 + 1 + 2 * 3, 2), dtype=np.float32)
    saida[0, :, 0] = [30, 30, 10, 10, 0.9, 25, 26, 0.9, 35, 36, 0.2]

    achadas = deteccoes(saida, pontos=2)

    assert len(achadas) == 1
    assert achadas[0].pontos is not None
    assert achadas[0].pontos.shape == (2, 3)


def test_saida_menor_que_os_pontos_pedidos_e_recusada() -> None:
    with pytest.raises(BenchError, match="pontos"):
        deteccoes(np.zeros((1, 6, 2), dtype=np.float32), pontos=1)


def test_a_concordancia_casa_por_iou_e_classe() -> None:
    pontos_a = np.array([[10, 10, 0.9], [20, 20, 0.9]], dtype=np.float32)
    pontos_b = np.array([[13, 14, 0.9], [20, 20, 0.1]], dtype=np.float32)
    referencia = [
        Deteccao((0, 0, 10, 10), 0.9, 0, pontos_a),
        Deteccao((50, 50, 60, 60), 0.8, 1),
    ]
    outra = [
        Deteccao((0, 0, 10, 11), 0.9, 0, pontos_b),
        Deteccao((50, 50, 60, 60), 0.8, 0),
        Deteccao((90, 90, 95, 95), 0.5, 1),
    ]

    casados, so_ref, so_outra, distancias = concordancia(referencia, outra)

    assert (casados, so_ref, so_outra) == (1, 1, 2)
    assert distancias == pytest.approx([5.0])


def test_bola_pequena_deslocada_casa_pelo_centro_e_nao_pelo_iou() -> None:
    referencia = [Deteccao((100, 100, 108, 108), 0.9, 0)]
    deslocada = [Deteccao((103, 100, 111, 108), 0.8, 0)]
    longe = [Deteccao((110, 100, 118, 108), 0.8, 0)]

    assert concordancia(referencia, deslocada)[0] == 0
    assert por_centro(referencia, deslocada, raio=4.0) == 1
    assert por_centro(referencia, longe, raio=4.0) == 0
    assert por_centro(referencia, [Deteccao((103, 100, 111, 108), 0.8, 1)], raio=4.0) == 0


def _carregador(respostas: Mapping[str, np.ndarray[Any, Any]]) -> Any:
    def carrega(plano: Path) -> RuntimeModel:
        resposta = respostas[plano.stem]

        def modelo(entradas: Mapping[str, Any]) -> Mapping[str, Any]:
            assert set(entradas) == {"images"}
            return {"output0": resposta}

        return modelo

    return carrega


def test_o_banco_mede_cada_plano_e_compara_com_a_referencia(tmp_path: Path) -> None:
    planos = {"fp32": _plano(tmp_path, "fp32"), "int8": _plano(tmp_path, "int8")}
    respostas = {
        "fp32": _saida([(50, 50, 20, 20, 0.9, 0.0), (10, 10, 4, 4, 0.0, 0.7)]),
        "int8": _saida([(50, 50, 20, 20, 0.9, 0.0)]),
    }
    quadros = [np.zeros((108, 192, 3), dtype=np.uint8)] * 3
    passos = iter(float(i) for i in range(1000))

    relatorio = compara(
        planos,
        quadros,
        _carregador(respostas),
        referencia="fp32",
        aquecimento=1,
        relogio=lambda: next(passos) / 1000,
    )

    int8: Any = relatorio["planos"]
    assert relatorio["quadros"] == 3
    assert int8["fp32"]["ms_por_passada"] == [1.0, 1.0, 1.0]
    assert int8["fp32"]["deteccoes_por_quadro"] == 2
    assert int8["int8"]["contra_referencia"]["revocacao"] == 0.5
    assert int8["int8"]["contra_referencia"]["precisao"] == 1.0
    assert int8["int8"]["contra_referencia"]["revocacao_centro"] == 0.5
    assert int8["int8"]["contra_referencia"]["precisao_centro"] == 1.0
    assert "contra_referencia" not in int8["fp32"]


def test_referencia_ausente_ou_sem_quadro_e_recusado(tmp_path: Path) -> None:
    planos = {"fp16": _plano(tmp_path, "fp16")}
    with pytest.raises(BenchError, match="referencia"):
        compara(planos, [np.zeros((4, 4, 3), dtype=np.uint8)], _carregador({}), referencia="fp32")
    with pytest.raises(BenchError, match="quadro"):
        compara(planos, [], _carregador({}), referencia="fp16")


def test_o_comando_recusa_plano_sem_nome() -> None:
    resultado = CliRunner().invoke(app, ["bench-planos", "--plano", "sem-igual.engine"])

    assert resultado.exit_code == 1
    assert "nome=caminho" in resultado.output
