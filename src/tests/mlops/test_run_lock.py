from __future__ import annotations

import multiprocessing
from collections.abc import Generator
from contextlib import contextmanager
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path

import pytest

from mlops.checkpoint import CheckpointError
from mlops.run_lock import exclusive_run


def _hold_run(destination: Path, connection: Connection) -> None:
    try:
        with exclusive_run(destination):
            connection.send("locked")
            connection.recv()
    finally:
        connection.close()


@contextmanager
def _holding_process(destination: Path) -> Generator[tuple[BaseProcess, Connection], None, None]:
    context = multiprocessing.get_context("fork")
    parent, child = context.Pipe()
    process = context.Process(target=_hold_run, args=(destination, child))
    process.start()
    child.close()
    try:
        assert parent.poll(10), "o processo nao adquiriu a trava"
        assert parent.recv() == "locked"
        yield process, parent
    finally:
        if process.is_alive():
            process.kill()
        process.join(10)
        parent.close()
        process.close()


def test_concurrent_process_is_refused_before_modifying_run(tmp_path: Path) -> None:
    target = tmp_path / "events.jsonl.parcial"
    target.write_text("saved events\n")

    with _holding_process(tmp_path):
        with pytest.raises(CheckpointError, match="em uso"), exclusive_run(tmp_path):
            target.write_text("overwritten events\n")
        assert target.read_text() == "saved events\n"


@pytest.mark.parametrize("abrupt", [False, True])
def test_process_exit_releases_lock_without_removing_its_file(tmp_path: Path, abrupt: bool) -> None:
    with _holding_process(tmp_path) as (process, connection):
        if abrupt:
            process.kill()
        else:
            connection.send("release")
        process.join(10)
        assert not process.is_alive()
        assert (tmp_path / ".run.lock").exists()

        with exclusive_run(tmp_path):
            (tmp_path / "resumed").touch()

    assert (tmp_path / "resumed").exists()


def test_exception_releases_lock_for_a_new_attempt(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="interrupted"), exclusive_run(tmp_path):
        raise RuntimeError("interrupted")

    with exclusive_run(tmp_path):
        (tmp_path / "resumed").touch()

    assert (tmp_path / "resumed").exists()


def test_different_runs_can_execute_concurrently(tmp_path: Path) -> None:
    with _holding_process(tmp_path / "first"), exclusive_run(tmp_path / "second"):
        (tmp_path / "second" / "events.jsonl.parcial").touch()

    assert (tmp_path / "second" / "events.jsonl.parcial").exists()
