"""A procedencia que nao cabe no evento congelado, num arquivo irmao."""

from __future__ import annotations

import json
from pathlib import Path

from mlops.run import RunRecord, write_run


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
