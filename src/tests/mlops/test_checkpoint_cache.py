"""Checkpoint compatibility and validation at the persisted cache-policy boundary."""

import json
from pathlib import Path

import pytest

from mlops.checkpoint import Checkpoint, CheckpointError, read_checkpoint, write_checkpoint


def test_legacy_checkpoint_means_cache_enabled(tmp_path: Path) -> None:
    path = write_checkpoint(
        tmp_path, Checkpoint(config_sha256="fixture", population=0, started_at="fixture")
    )
    document = json.loads(path.read_text())
    del document["cache_enabled"]
    path.write_text(json.dumps(document))

    assert read_checkpoint(tmp_path).cache_enabled is True


@pytest.mark.parametrize("policy", [None, 0, 1, "false", {}, []])
def test_checkpoint_rejects_non_boolean_cache_policy(tmp_path: Path, policy: object) -> None:
    path = write_checkpoint(
        tmp_path, Checkpoint(config_sha256="fixture", population=0, started_at="fixture")
    )
    document = json.loads(path.read_text())
    document["cache_enabled"] = policy
    path.write_text(json.dumps(document))

    with pytest.raises(CheckpointError, match="cache_enabled"):
        read_checkpoint(tmp_path)
