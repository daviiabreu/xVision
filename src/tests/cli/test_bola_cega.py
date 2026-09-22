"""Os comandos da segunda passada cega da bola, pelo lado de quem usa a CLI."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cli import app
from edge.paths import VAR_ENV
from mlops.ball_trajectory import BallTrajectoryLabel
from mlops.bola_cega import Marca, grava, registro

AGORA = datetime(2026, 9, 22, 12, 0, tzinfo=timezone.utc)


def rotulo(frame_index: int) -> BallTrajectoryLabel:
    return BallTrajectoryLabel(
        clip_id="c0",
        match_id="m0",
        video_sha256="a" * 64,
        frame_index=frame_index,
        timestamp_s=frame_index / 60,
        center_xy=(100.0, 100.0),
        apparent_diameter_px=10.0,
        visibility="visible",
        trajectory_segment=0,
        is_keyframe=True,
        motion_blur=False,
        camera_cut=False,
        annotator="davi",
        reviewed_by=None,
        quality_flags=("human-annotated",),
    )


@pytest.fixture
def labels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(VAR_ENV, str(tmp_path))
    base = tmp_path / "labels"
    (base / "model-pilots-v1").mkdir(parents=True)
    (base / "model-pilots-v1" / "queue.json").write_text(
        json.dumps({"ball": [{"clip_id": "c0", "contact_frame": 60}]})
    )
    for quadro, x in ((60, 100.0), (63, 100.0)):
        grava(
            base / "ball-trajectory-v1.jsonl",
            registro(
                rotulo(quadro), Marca("visible", (x, 100.0), 10.0, False, False), "davi", AGORA
            ),
        )
    for quadro, x in ((60, 102.0), (63, 110.0)):
        grava(
            base / "ball-trajectory-v1-cega.jsonl",
            registro(
                rotulo(quadro), Marca("visible", (x, 100.0), 10.0, False, False), "davi", AGORA
            ),
        )
    return base


@pytest.mark.usefixtures("labels")
def test_a_concordancia_sai_no_terminal_e_em_json(tmp_path: Path) -> None:
    saida = tmp_path / "concordancia.json"

    resultado = CliRunner().invoke(app, ["label", "ball-agreement", "--saida", str(saida)])

    assert resultado.exit_code == 0, resultado.output
    assert "2 quadros comparados, 1 de contato" in resultado.output
    assert "mediana 0.600" in resultado.output
    assert "fora" in resultado.output
    assert json.loads(saida.read_text())["centro_max_diametros"] == pytest.approx(1.0)


def test_sem_segunda_passada_a_concordancia_explica_e_sai_com_erro(labels: Path) -> None:
    (labels / "ball-trajectory-v1-cega.jsonl").unlink()

    resultado = CliRunner().invoke(app, ["label", "ball-agreement"])

    assert resultado.exit_code == 1
    assert "nao existe" in resultado.output


@pytest.mark.usefixtures("labels")
def test_anotar_sem_dizer_quem_e_recusado() -> None:
    resultado = CliRunner().invoke(app, ["label", "ball-blind", "--sem-navegador"])

    assert resultado.exit_code == 1
    assert "--annotator" in resultado.output


def test_anotar_sem_primeira_passada_e_recusado(labels: Path) -> None:
    (labels / "ball-trajectory-v1.jsonl").unlink()

    resultado = CliRunner().invoke(
        app, ["label", "ball-blind", "--annotator", "davi", "--sem-navegador"]
    )

    assert resultado.exit_code == 1
    assert "nao existe" in resultado.output
