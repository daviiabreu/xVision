"""A procedencia que nao cabe no evento congelado, num arquivo irmao."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlops.run import RunRecord, hardware, runtimes, write_run


def _registro() -> RunRecord:
    return RunRecord(
        run_id="20260824-014211-a1b2c3",
        started_at="2026-08-24T01:42:11.500Z",
        git_sha="9179287",
        git_dirty=False,
        config_sha256="7d31",
        dataset={"clips": 604, "population": "todos-604"},
        modules={
            "pose": {
                "name": "stub-pose",
                "version": "0.1.0",
                "status": "ok",
                "reason": None,
                "duration_ms": 4.0,
            }
        },
        features=(("marked", "0.2.0"),),
        host="orin-nano-01",
        python="3.10.14",
        platform="linux-aarch64",
        hardware={
            "machine": "aarch64",
            "cpu_count": 6,
            "tegra_release": "# R36 (release), REVISION: 3.0",
            "gpu": "Orin",
        },
        runtime={"torch": "2.3.0", "tensorrt": "10.3.0", "cuda": "12.2"},
    )


def test_the_record_carries_what_lets_another_person_repeat_the_run(tmp_path: Path) -> None:
    escrito = write_run(tmp_path, _registro())

    dados = json.loads(escrito.read_text(encoding="utf-8"))
    assert dados["git_sha"] == "9179287"
    assert dados["git_dirty"] is False
    assert dados["config_sha256"] == "7d31"


def test_every_stage_carries_status_reason_and_duration(tmp_path: Path) -> None:
    """Este e um projeto de borda: o numero que ele mais precisa reportar e ms por clipe."""
    dados = json.loads(write_run(tmp_path, _registro()).read_text(encoding="utf-8"))

    pose = dados["modules"]["pose"]
    assert set(pose) >= {"status", "reason", "duration_ms"}
    assert pose["duration_ms"] == 4.0


def test_the_record_never_touches_the_frozen_event_schema(tmp_path: Path) -> None:
    """`shot_event.v1.json` tem additionalProperties false e um digest cravado em teste.

    A saida de menor resistencia seria enfiar `git_sha` dentro de `versions`, que valida porque
    `versions` e mapa de string, e em duas semanas o evento vira saco de gato sem tipo.
    """
    from contracts.schema import load_schema

    dados = json.loads(write_run(tmp_path, _registro()).read_text(encoding="utf-8"))
    evento = load_schema()

    procedencia = {"git_sha", "git_dirty", "config_sha256", "host", "python", "platform"}
    assert procedencia <= set(dados)
    assert procedencia & set(evento["properties"]) == set()
    assert evento["additionalProperties"] is False


def test_two_runs_in_the_same_second_do_not_share_a_directory() -> None:
    """Reproduzido 3 de 3 em concorrencia: `events.jsonl` e `run.json` da mesma pasta
    descreviam execucoes diferentes, em silencio e com exit zero."""
    from mlops.run import new_run_id

    assert len({new_run_id() for _ in range(50)}) == 50


def test_the_run_id_matches_the_shape_the_record_documents() -> None:
    import re

    from mlops.run import new_run_id

    assert re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{6}", new_run_id())


def test_a_maquina_que_produziu_o_numero_fica_escrita(tmp_path: Path) -> None:
    """`platform` sozinho devolve `linux-aarch64` numa Jetson, igual a qualquer ARM com Linux.

    O defeito replantado nao levanta nada e foi apontado na revisao da Sprint 3: os tempos por
    etapa existiam e a maquina que os produziu nao estava escrita em lugar nenhum, entao ninguem
    conseguia repetir a medicao sem a apresentacao oral do lado.
    """
    dados = json.loads(write_run(tmp_path, _registro()).read_text(encoding="utf-8"))

    assert dados["hardware"]["gpu"] == "Orin"
    assert dados["hardware"]["tegra_release"].startswith("# R36")
    assert dados["runtime"]["tensorrt"] == "10.3.0"
    assert dados["runtime"]["cuda"] == "12.2"


def test_o_schema_sobe_quando_o_registro_cresce(tmp_path: Path) -> None:
    """Consumidor que le `run.json` precisa saber que os campos novos existem a partir de quando."""
    dados = json.loads(write_run(tmp_path, _registro()).read_text(encoding="utf-8"))

    assert dados["schema_version"] == "1.1"


def test_runtimes_le_versao_sem_importar_o_pacote() -> None:
    """Ler metadado custa um stat; importar torch custa segundos numa execucao de stubs."""
    import sys

    antes = set(sys.modules)

    achadas = runtimes()

    assert "numpy" in achadas
    assert not {m for m in set(sys.modules) - antes if m.split(".")[0] in {"torch", "tensorrt"}}


def test_hardware_nao_inventa_gpu_quando_nao_ha_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ausencia de GPU e informacao; campo vazio ao lado dos outros parece dado que ninguem leu."""
    import sys

    monkeypatch.setitem(sys.modules, "torch", None)

    medido = hardware()

    assert "gpu" not in medido
    assert medido["machine"]
