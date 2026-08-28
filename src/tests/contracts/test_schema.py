"""Testes de contrato: o evento vale contra o JSON Schema, e o schema esta preso a versao."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Any, get_args

import pytest

from contracts import (
    SCHEMA_PATH,
    SCHEMA_VERSION,
    ContractError,
    FeatureStatus,
    SchemaValidationError,
    ShotEvent,
    load_schema,
)
from contracts.demo import build_demo_event
from contracts.events import _DATE_TIME_PROFILE  # pyright: ignore[reportPrivateUsage]
from contracts.validation import validate_shot_event

PINNED_VERSION = "1.0"
PINNED_DIGEST = "2cef64e155c096e1ea7714ffca65dcd0f885e990f4d8a6a5e0ced600b330da75"

IGNORED_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".claude", "var"}
"""Pastas que nao sao o repositorio, e que fariam a varredura achar copia que nao existe no git.

`.claude` guarda worktree temporario de agente, que e uma checagem completa do repo e portanto tem
o proprio `shot_event.v1.json`; `var` guarda cache e execucao. As duas sao ignoradas pelo git, entao
copia dentro delas nao e o defeito que este teste procura, que e schema duplicado e versionado."""

_RAIZES_DO_SDIST = {
    "api",
    "cli",
    "contracts",
    "edge",
    "mlops",
    "xvision_players_yolo",
    ".gitignore",
    "pyproject.toml",
    "PKG-INFO",
}


@pytest.fixture
def event() -> ShotEvent:
    """Definida aqui, e nao num `conftest.py` da pasta.

    Medido: com os arquivos passados na ordem `contracts/...`, algo fora de `contracts/`, e
    `contracts/...` de novo, o conftest da pasta deixa de ser registrado no reentrar e quatro
    testes morrem com `fixture 'event' not found`. A suite so e verde na ordem em que ela e
    normalmente coletada, e qualquer fatiamento de CI ou selecao manual de arquivos derruba
    testes por motivo que nao tem nada a ver com o codigo.

    Uma linha de import custa menos que depender dessa ordem.
    """
    return build_demo_event()


def test_valid_event_passes_schema(event: ShotEvent) -> None:
    validate_shot_event(event.to_dict())


def test_schema_rejects_missing_shot_id(event: ShotEvent) -> None:
    payload: dict[str, Any] = event.to_dict()
    del payload["shot_id"]

    with pytest.raises(SchemaValidationError, match="shot_id"):
        validate_shot_event(payload)


def test_schema_rejects_unknown_field(event: ShotEvent) -> None:
    payload: dict[str, Any] = event.to_dict()
    payload["expected_goals"] = 0.21

    with pytest.raises(SchemaValidationError, match="expected_goals"):
        validate_shot_event(payload)


def test_schema_rejects_unknown_schema_version(event: ShotEvent) -> None:
    payload: dict[str, Any] = event.to_dict()
    payload["schema_version"] = "2.0"

    with pytest.raises(SchemaValidationError, match="schema_version"):
        validate_shot_event(payload)


def test_schema_validation_error_is_a_contract_error() -> None:
    assert issubclass(SchemaValidationError, ContractError)


def test_schema_version_is_pinned() -> None:
    schema = load_schema()

    assert SCHEMA_VERSION == PINNED_VERSION
    assert schema["properties"]["schema_version"]["const"] == PINNED_VERSION
    assert hashlib.sha256(SCHEMA_PATH.read_bytes()).hexdigest() == PINNED_DIGEST


def test_feature_status_espelha_o_enum_do_schema() -> None:
    schema = load_schema()

    assert set(get_args(FeatureStatus)) == set(
        schema["$defs"]["featureValue"]["properties"]["status"]["enum"]
    )


def repository_root() -> Path:
    root = Path(__file__).resolve().parents[3]
    assert (root / ".gitlab-ci.yml").is_file(), f"{root} nao e a raiz do repositorio"
    return root


def test_schema_file_has_no_copy_in_the_repository() -> None:
    repository = repository_root()

    copies = [
        path for path in repository.rglob(SCHEMA_PATH.name) if IGNORED_DIRS.isdisjoint(path.parts)
    ]

    assert copies == [SCHEMA_PATH]


@pytest.fixture(scope="module")
def artefatos(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    destino = tmp_path_factory.mktemp("dist")
    saida = subprocess.run(
        ["uv", "build", "--offline", "--out-dir", str(destino)],
        cwd=repository_root() / "src",
        capture_output=True,
        text=True,
    )
    assert saida.returncode == 0, f"uv build falhou:\n{saida.stderr}"

    wheels = list(destino.glob("*.whl"))
    sdists = list(destino.glob("*.tar.gz"))
    assert len(wheels) == 1 and len(sdists) == 1, f"esperava um de cada, veio {wheels} {sdists}"
    return wheels[0], sdists[0]


def test_schema_file_is_shipped_in_the_wheel(artefatos: tuple[Path, Path]) -> None:
    wheel_path, _ = artefatos
    membro = f"contracts/{SCHEMA_PATH.name}"

    with zipfile.ZipFile(wheel_path) as wheel:
        nomes = wheel.namelist()
        assert membro in nomes, f"{membro} ficou fora do wheel; presentes: {nomes}"
        conteudo = wheel.read(membro)

    assert hashlib.sha256(conteudo).hexdigest() == PINNED_DIGEST, (
        "o schema dentro do wheel diverge do schema do repositorio"
    )


def test_o_sdist_leva_so_o_pacote(artefatos: tuple[Path, Path]) -> None:
    _, sdist_path = artefatos

    with tarfile.open(sdist_path) as sdist:
        nomes = [nome.split("/", 1)[-1] for nome in sdist.getnames()]

    intrusos = [nome for nome in nomes if nome.split("/", 1)[0] not in _RAIZES_DO_SDIST]
    assert not intrusos, f"o sdist leva o que nao e do pacote: {sorted(intrusos)[:10]}"


def test_schema_sem_as_regras_do_tipo_falha_legivel(tmp_path: Path) -> None:
    destino = tmp_path / "contracts"
    shutil.copytree(SCHEMA_PATH.parent, destino)
    quebrado = json.loads((destino / SCHEMA_PATH.name).read_text(encoding="utf-8"))
    del quebrado["properties"]["shot_id"]["pattern"]
    (destino / SCHEMA_PATH.name).write_text(json.dumps(quebrado), encoding="utf-8")

    saida = subprocess.run(
        [sys.executable, "-c", "import contracts"],
        cwd=tmp_path,
        env={"PYTHONPATH": str(tmp_path), "PATH": os.environ.get("PATH", "")},
        capture_output=True,
        text=True,
    )

    assert saida.returncode != 0
    assert "nao traz as regras que o tipo cobra" in saida.stderr


def test_importing_the_package_does_not_require_jsonschema() -> None:
    codigo = "import contracts, sys; print('jsonschema' in sys.modules)"
    saida = subprocess.run(
        [sys.executable, "-c", codigo], capture_output=True, text=True, check=True
    )

    assert saida.stdout.strip() == "False", (
        "importar contracts carregou jsonschema no caminho de quem so monta e serializa"
    )


def test_o_padrao_de_started_at_e_o_mesmo_no_schema_e_no_tipo() -> None:
    do_schema = load_schema()["$defs"]["pipelineMetadata"]["properties"]["started_at"]["pattern"]

    assert do_schema.endswith("$")
    assert do_schema.removesuffix("$") + r"\Z" == _DATE_TIME_PROFILE.pattern
