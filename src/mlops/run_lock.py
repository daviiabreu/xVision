"""Exclusividade por diretorio durante a escrita e a retomada de uma execucao."""

from __future__ import annotations

import fcntl
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from mlops.checkpoint import CheckpointError


@contextmanager
def exclusive_run(destination: Path) -> Generator[None, None, None]:
    """O SO libera a trava ao fechar o processo, mesmo quando ele e interrompido.

    O arquivo permanece no diretorio: remove-lo permitiria que dois processos travassem
    inodes diferentes. Sua existencia nao indica uma execucao ativa; somente flock indica.
    """
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / ".run.lock").open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise CheckpointError(f"{destination}: diretorio de execucao em uso") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
