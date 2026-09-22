"""A configuracao de CPU e a de producao sem as diretivas que exigem CUDA, e nada alem disso.

Ela existe para a evidencia operacional com os modulos reais ser reproduzivel sem a Jetson. O risco
proprio dela e apodrecer calada: alguem muda um limiar em `pipeline.models.toml`, esquece da copia,
e a evidencia passa a descrever uma configuracao que ninguem roda. E exatamente o defeito que a
revisao da Sprint 3 apontou no comentario do `players`, que ficou para tras um dia depois de a bola
mudar de precisao.

O teste le as duas e exige que a diferenca seja so a declarada abaixo.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):  # pragma: no cover - o piso e 3.10
    import tomllib
else:
    import tomli as tomllib

CONFIG = Path(__file__).resolve().parents[1] / "config"
PRODUCAO = CONFIG / "pipeline.models.toml"
CPU = CONFIG / "pipeline.models.cpu.toml"

SO_CUDA: dict[str, set[str]] = {
    "players": {"quantize"},
    "ball": {"quantize", "letterbox"},
}
"""As chaves que a configuracao de CPU tira, por especie. Tudo o mais tem de ser igual."""


def _le(caminho: Path) -> dict[str, Any]:
    return tomllib.loads(caminho.read_text(encoding="utf-8"))


def test_fora_das_diretivas_de_cuda_as_duas_configuracoes_sao_iguais() -> None:
    producao, cpu = _le(PRODUCAO), _le(CPU)

    esperado = {**producao, "modules": dict(producao["modules"])}
    for especie, chaves in SO_CUDA.items():
        esperado["modules"][especie] = {
            k: v for k, v in producao["modules"][especie].items() if k not in chaves
        }

    assert cpu == esperado, (
        "pipeline.models.cpu.toml divergiu de pipeline.models.toml fora das diretivas de CUDA. "
        "Quem mudou a producao precisa repetir a mudanca na copia de CPU."
    )


def test_a_configuracao_de_cpu_nao_declara_nada_que_exija_cuda() -> None:
    cpu = _le(CPU)

    proibidas = [
        f"{especie}.{chave}"
        for especie, secao in cpu["modules"].items()
        for chave, valor in secao.items()
        if (chave == "quantize" and valor == "fp16") or (chave == "letterbox" and valor == "cuda")
    ]

    assert not proibidas, f"a configuracao de CPU declara diretiva que exige GPU: {proibidas}"


def test_a_producao_ainda_declara_o_que_a_copia_tira() -> None:
    """Se a producao parar de declarar fp16, a lista acima vira mentira e precisa ser revista."""
    producao = _le(PRODUCAO)

    for especie, chaves in SO_CUDA.items():
        faltando = chaves - set(producao["modules"][especie])
        assert not faltando, f"{especie} deixou de declarar {faltando} em producao"
