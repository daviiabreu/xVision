"""A procedencia que nao cabe no evento congelado, num arquivo irmao.

Guarda o sha do commit e se a arvore estava suja, a identidade do dado, nome e versao de cada
modulo, o sha256 do `config/pipeline.toml`, o host e o status, motivo e duracao de cada etapa. E o
que permite dizer que a versao 0.3.1 do detector mudou um numero, e o que permite outra pessoa
repetir.

Arquivo irmao, e nao campo no evento, porque `shot_event.v1.json` tem `additionalProperties:
false` nos tres niveis e um digest cravado em teste. Crescer o evento custaria bump de versao, os
dois validadores, o `from_dict` e a nuvem, e nao desbloqueia uma pessoa sequer. A saida de menor
resistencia seria enfiar `git_sha` dentro de `versions`, que valida porque `versions` e mapa de
string, e em duas semanas o evento vira saco de gato sem tipo.

`duration_ms` por etapa esta aqui de proposito: este e um projeto de borda com Jetson, e o numero
que ele mais precisa reportar e milissegundo por clipe no dispositivo.

Nao calcula metrica, nao compara execucoes e nao sobe para servidor nenhum. Nao entra no git,
porque `var/` e ignorado, entao predicao sobre clipe do parceiro nunca chega ao repositorio por
construcao.
"""

from __future__ import annotations

import json
import platform as plataforma
import secrets
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
RUN_FILE = "run.json"
EVENTS_FILE = "events.jsonl"


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    started_at: str
    git_sha: str
    git_dirty: bool
    config_sha256: str
    dataset: Mapping[str, object]
    modules: Mapping[str, Mapping[str, object]]
    features: Sequence[tuple[str, str]]
    host: str
    python: str
    platform: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "git_sha": self.git_sha,
            "git_dirty": self.git_dirty,
            "config_sha256": self.config_sha256,
            "dataset": dict(self.dataset),
            "modules": {k: dict(v) for k, v in self.modules.items()},
            "features": [list(par) for par in self.features],
            "host": self.host,
            "python": self.python,
            "platform": self.platform,
        }


def write_run(destino: Path, registro: RunRecord) -> Path:
    destino.mkdir(parents=True, exist_ok=True)
    caminho = destino / RUN_FILE
    parcial = caminho.with_suffix(".json.parcial")
    parcial.write_text(json.dumps(registro.to_dict(), indent=1) + "\n", encoding="utf-8")
    parcial.replace(caminho)
    return caminho


def new_run_id(agora: datetime | None = None) -> str:
    """Segundo sozinho colide: duas execucoes no mesmo segundo escreviam na mesma pasta, e o
    `events.jsonl` e o `run.json` dela passavam a descrever execucoes diferentes, com exit zero."""
    instante = agora or datetime.now(timezone.utc)
    return f"{instante.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"


def git_state() -> tuple[str, bool]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        sujo = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return "", False
    return sha, sujo


def environment() -> tuple[str, str, str]:
    return (
        plataforma.node(),
        ".".join(str(p) for p in sys.version_info[:3]),
        f"{plataforma.system().lower()}-{plataforma.machine()}",
    )
