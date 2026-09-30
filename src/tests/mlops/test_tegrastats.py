"""O resumo de memoria e energia de uma execucao, lido do log do `tegrastats`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cli import app
from mlops.tegrastats import TegrastatsError, le, resume

MODELO = (
    "09-24-2026 14:08:{s:02d} RAM {ram}/7486MB (lfb 179x4MB) SWAP {swap}/4096MB (cached 42MB) "
    "CPU [92%@1728,13%@1728] GR3D_FREQ {gpu}% cpu@49C/49C soc2@47.75C/47.75C "
    "gpu@48.5C/48.5C tj@{tj}C/{tj}C soc1@47.9C/47.9C VDD_IN {vdd}mW/9999mW/99999mW "
    "VDD_CPU_GPU_CV 1344mW/1344mW/1344mW VDD_SOC 1386mW/1386mW/1386mW"
)


def _log(tmp_path: Path, amostras: list[tuple[int, int, int, int, float, int]]) -> Path:
    caminho = tmp_path / "tegrastats.log"
    linhas = [
        MODELO.format(s=s, ram=ram, swap=swap, gpu=gpu, tj=tj, vdd=vdd)
        for s, ram, swap, gpu, tj, vdd in amostras
    ]
    caminho.write_text("\n".join(["cabecalho que o tegrastats nao escreve", *linhas]) + "\n")
    return caminho


def test_a_energia_integra_a_potencia_instantanea(tmp_path: Path) -> None:
    caminho = _log(
        tmp_path,
        [(0, 2000, 0, 0, 49.0, 4000), (1, 5000, 10, 99, 60.5, 12000), (3, 3000, 0, 0, 55.0, 6000)],
    )

    resumo = resume(le(caminho), clipes=2)

    assert resumo["duracao_s"] == 3
    assert resumo["energia_j"] == pytest.approx(8.0 + 18.0)
    assert resumo["energia_por_clipe_j"] == pytest.approx(13.0)
    assert resumo["segundos_por_clipe"] == pytest.approx(1.5)
    assert resumo["ram_pico_mb"] == 5000
    assert resumo["ram_total_mb"] == 7486
    assert resumo["swap_pico_mb"] == 10
    assert resumo["potencia_pico_w"] == 12.0
    assert resumo["potencia_media_w"] == pytest.approx(26.0 / 3, abs=0.01)
    assert resumo["temperatura_pico_c"] == 60.5
    assert resumo["gpu_ocupada"] == pytest.approx(1 / 3, abs=0.001)


def test_sem_clipes_o_resumo_nao_tem_por_clipe(tmp_path: Path) -> None:
    resumo = resume(le(_log(tmp_path, [(0, 1, 0, 0, 1.0, 1000), (1, 1, 0, 0, 1.0, 1000)])))

    assert "energia_por_clipe_j" not in resumo


def test_log_curto_ou_ausente_e_recusado(tmp_path: Path) -> None:
    with pytest.raises(TegrastatsError, match="nao existe"):
        le(tmp_path / "nada.log")
    with pytest.raises(TegrastatsError, match="menos de duas"):
        le(_log(tmp_path, [(0, 1, 0, 0, 1.0, 1000)]))
    with pytest.raises(TegrastatsError, match="duas amostras"):
        resume(())


def test_clipes_nao_positivo_e_recusado(tmp_path: Path) -> None:
    amostras = le(_log(tmp_path, [(0, 1, 0, 0, 1.0, 1000), (1, 1, 0, 0, 1.0, 1000)]))
    with pytest.raises(TegrastatsError, match="positivo"):
        resume(amostras, clipes=0)


def test_o_comando_imprime_e_grava_o_resumo(tmp_path: Path) -> None:
    caminho = _log(tmp_path, [(0, 2000, 0, 0, 49.0, 4000), (2, 3000, 0, 50, 50.0, 8000)])
    saida = tmp_path / "resumo.json"

    resultado = CliRunner().invoke(
        app, ["tegrastats", str(caminho), "--clipes", "1", "--saida", str(saida)]
    )

    assert resultado.exit_code == 0, resultado.output
    assert json.loads(saida.read_text())["energia_por_clipe_j"] == pytest.approx(12.0)


def test_o_comando_explica_log_invalido(tmp_path: Path) -> None:
    resultado = CliRunner().invoke(app, ["tegrastats", str(tmp_path / "nada.log")])

    assert resultado.exit_code == 1
    assert "nao existe" in resultado.output


def test_sem_swap_o_tegrastats_omite_o_campo_e_o_leitor_aceita(tmp_path: Path) -> None:
    caminho = tmp_path / "sem-swap.log"
    linha = (
        "09-24-2026 15:54:{s} RAM 2656/7486MB (lfb 106x4MB) CPU [0%@1728,83%@1728] "
        "GR3D_FREQ 0% cpu@49.4C/49.8C gpu@49.1C/49.1C tj@49.4C/49.8C "
        "VDD_IN 4944mW/4457mW/4944mW VDD_SOC 1384mW/1247mW/1384mW"
    )
    caminho.write_text(linha.format(s="49") + "\n" + linha.format(s="50") + "\n")

    resumo = resume(le(caminho))

    assert resumo["swap_pico_mb"] == 0
    assert resumo["energia_j"] == pytest.approx(4.9)


@pytest.mark.parametrize("seconds", [(2, 0), (0, 2, 1, 3)])
def test_summary_rejects_any_backwards_interval(tmp_path: Path, seconds: tuple[int, ...]) -> None:
    path = _log(tmp_path, [(second, 2000, 0, 0, 50.0, 4000) for second in seconds])

    with pytest.raises(TegrastatsError, match="ordem"):
        resume(le(path))


def test_summary_rejects_zero_total_duration(tmp_path: Path) -> None:
    path = _log(tmp_path, [(0, 2000, 0, 0, 50.0, 4000)] * 2)

    with pytest.raises(TegrastatsError, match="duracao.*positiva"):
        resume(le(path))


def test_repeated_seconds_are_allowed_when_total_duration_is_positive(tmp_path: Path) -> None:
    path = _log(tmp_path, [(second, 2000, 0, 0, 50.0, 4000) for second in (0, 0, 2)])

    summary = resume(le(path))

    assert summary["duracao_s"] == 2
    assert summary["energia_j"] == 8.0
    assert summary["potencia_media_w"] == 4.0


@pytest.mark.parametrize(
    ("original", "invalid"),
    [("09-24-2026", "09-31-2026"), ("tj@50.0C", "tj@50..0C")],
)
def test_invalid_recognized_sample_reports_file_and_original_line(
    tmp_path: Path, original: str, invalid: str
) -> None:
    path = _log(tmp_path, [(0, 2000, 0, 0, 50.0, 4000), (1, 2000, 0, 0, 50.0, 4000)])
    path.write_text(path.read_text().replace(original, invalid, 1))

    with pytest.raises(TegrastatsError, match=r"tegrastats\.log:2: amostra invalida"):
        le(path)


@pytest.mark.parametrize("seconds", [(2, 0), (0, 0), (0, 1)])
@pytest.mark.parametrize("existing_report", [False, True])
def test_cli_rejects_invalid_log_without_creating_or_overwriting_report(
    tmp_path: Path, seconds: tuple[int, ...], existing_report: bool
) -> None:
    path = _log(tmp_path, [(second, 2000, 0, 0, 50.0, 4000) for second in seconds])
    if seconds == (0, 1):
        path.write_text(path.read_text().replace("09-24-2026", "09-31-2026", 1))
    report = tmp_path / "summary.json"
    previous = b'{"previous": "valid report"}\n'
    if existing_report:
        report.write_bytes(previous)

    result = CliRunner().invoke(app, ["tegrastats", str(path), "--saida", str(report)])

    assert result.exit_code == 1, result.output
    if seconds == (2, 0):
        assert "ordem cronologica" in result.output
    elif seconds == (0, 0):
        assert "duracao total precisa ser positiva" in result.output
    else:
        assert "tegrastats.log:2: amostra invalida" in result.output
    if existing_report:
        assert report.read_bytes() == previous
    else:
        assert not report.exists()
