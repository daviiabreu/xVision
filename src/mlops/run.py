"""A procedencia que nao cabe no evento congelado, num arquivo irmao.

Guarda o sha do commit e se a arvore estava suja, a identidade do dado, nome e versao de cada
modulo, o sha256 do `config/pipeline.toml` e o `[decode].mode` que ele declara, o host e o status,
motivo e duracao de cada etapa. E o que permite dizer que a versao 0.3.1 do detector mudou um
numero, e o que permite outra pessoa repetir.

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
import os
import platform as plataforma
import secrets
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, cast

SCHEMA_VERSION = "1.3"
RUN_FILE = "run.json"
EVENTS_FILE = "events.jsonl"


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    started_at: str
    git_sha: str
    git_dirty: bool
    config_sha256: str
    decode: str
    dataset: Mapping[str, object]
    modules: Mapping[str, Mapping[str, object]]
    features: Sequence[tuple[str, str]]
    host: str
    python: str
    platform: str
    hardware: Mapping[str, object]
    runtime: Mapping[str, str]
    cache_enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "git_sha": self.git_sha,
            "git_dirty": self.git_dirty,
            "config_sha256": self.config_sha256,
            "decode": self.decode,
            "cache_enabled": self.cache_enabled,
            "dataset": dict(self.dataset),
            "modules": {k: dict(v) for k, v in self.modules.items()},
            "features": [list(par) for par in self.features],
            "host": self.host,
            "python": self.python,
            "platform": self.platform,
            "hardware": dict(self.hardware),
            "runtime": dict(self.runtime),
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


PACOTES = (
    "torch",
    "torchvision",
    "ultralytics",
    "opencv-python-headless",
    "numpy",
    "onnxruntime",
    "onnxruntime-gpu",
    "tensorrt",
    "tensorrt-cu13",
    "tensorrt-cu12",
    "timm",
)
"""Os pacotes cujo numero de versao muda o resultado ou o tempo, e nenhum a mais."""

TEGRA = Path("/etc/nv_tegra_release")
"""Onde o L4T declara a propria versao na Jetson. Ausente em qualquer outra maquina."""


def runtimes() -> dict[str, str]:
    """A versao de cada pacote que decide o numero, lida do metadado sem importar nada.

    `importlib.metadata.version` le o `.dist-info` do ambiente, entao ela custa um stat e nao
    carrega torch. Isso importa numa execucao de stubs, que hoje nunca paga o import e nao deve
    passar a pagar so para preencher um relatorio.

    Pacote ausente fica de fora em vez de entrar vazio: nao ter TensorRT instalado e informacao, e
    string vazia ao lado das outras parece versao que ninguem leu.

    CUDA e cuDNN so entram quando `torch` JA esta carregado. Importa-lo aqui custaria segundos numa
    execucao que nunca precisou dele, e a versao de CUDA nao existe fora do torch.
    """
    achadas: dict[str, str] = {}
    for nome in PACOTES:
        try:
            achadas[nome] = metadata.version(nome)
        except metadata.PackageNotFoundError:
            continue
    torch = cast("Any", sys.modules.get("torch"))
    if torch is None:
        return achadas
    cuda = getattr(getattr(torch, "version", None), "cuda", None)
    if isinstance(cuda, str):
        achadas["cuda"] = cuda
    try:
        cudnn = torch.backends.cudnn.version()
    except (AttributeError, OSError, RuntimeError):
        cudnn = None
    if isinstance(cudnn, int):
        achadas["cudnn"] = str(cudnn)
    return achadas


def hardware() -> dict[str, object]:
    """A maquina que produziu o numero, no detalhe que permite outra pessoa repetir.

    `platform` sozinho devolve `linux-aarch64` numa Jetson, que e indistinguivel de qualquer ARM
    com Linux. Um tempo por etapa sem a placa ao lado nao e reproduzivel, e foi exatamente o que
    faltou nas medicoes da Sprint 3: o numero existe, a maquina que o produziu nao esta escrita em
    lugar nenhum.

    A GPU so e consultada quando `torch` ja esta carregado, pelo mesmo motivo de `runtimes`.
    """
    dados: dict[str, object] = {
        "machine": plataforma.machine(),
        "processor": plataforma.processor(),
        "cpu_count": os.cpu_count(),
        "release": plataforma.release(),
    }
    try:
        if TEGRA.is_file():
            dados["tegra_release"] = TEGRA.read_text(encoding="utf-8").strip().splitlines()[0]
    except (OSError, IndexError):
        pass
    torch = cast("Any", sys.modules.get("torch"))
    if torch is None:
        return dados
    try:
        if torch.cuda.is_available():
            dados["gpu"] = str(torch.cuda.get_device_name(0))
            dados["gpu_memory_mb"] = round(
                float(torch.cuda.get_device_properties(0).total_memory) / 1e6
            )
    except (AttributeError, AssertionError, OSError, RuntimeError):
        pass
    return dados
