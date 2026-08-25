"""Quem pode importar quem, verificado, e nao confiado a disciplina.

A borda vai para a Jetson e nunca le CSV, entao ela nao importa `mlops`. O contrato nao importa
ninguem, porque e o acordo entre as pontas. `cli` e `api` existem para juntar as outras, e por
isso sao as unicas que importam mais de uma: o `cli` usa `edge` e `mlops`, e a `api` usa so `edge`,
porque ela roda o pipeline e nao le CSV. As duas sao pontos de entrada, entao ninguem as importa.

Com um pacote so, o resolvedor de dependencias nao separa mais as areas, entao a fronteira entre
elas deixa de ser fato da instalacao e passa a ser verificada aqui.

O que este teste **nao** cobre e dependencia de terceiro. A raiz declara cinco dependencias que
toda area carrega, e o extra `api` separa apenas o que o servico HTTP precisa: nada impede a
borda importar `fastapi` se alguem instalar o extra. Separar de verdade exigiria um pacote por
area, e o preco disso ja foi pago uma vez e revertido.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1]

PROIBIDO = {
    "edge": {"mlops", "cli", "api"},
    "contracts": {"edge", "mlops", "cli", "api"},
    "mlops": {"cli", "api"},
    "cli": {"api"},
    "api": {"mlops", "cli"},
}


def _raizes_importadas(modulo: Path) -> set[str]:
    raizes: set[str] = set()
    for no in ast.walk(ast.parse(modulo.read_text(encoding="utf-8"))):
        if isinstance(no, ast.Import):
            raizes.update(alias.name.split(".")[0] for alias in no.names)
        elif isinstance(no, ast.ImportFrom) and no.level == 0 and no.module:
            raizes.add(no.module.split(".")[0])
    return raizes


@pytest.mark.parametrize("area", sorted(PROIBIDO))
def test_a_area_nao_importa_quem_nao_deve(area: str) -> None:
    modulos = sorted((SRC / area).rglob("*.py"))

    assert modulos, f"{SRC / area} nao tem modulo nenhum, o teste perderia o sentido"
    for modulo in modulos:
        invasores = _raizes_importadas(modulo) & PROIBIDO[area]
        assert not invasores, f"{modulo.relative_to(SRC)} importa {sorted(invasores)}"
