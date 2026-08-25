"""A topologia declarada nao pertence a provedor nenhum, e isso e o que a mudanca compra.

Terraform nao tem recurso generico de nuvem: `railway_project`, `aws_ecs_service` e
`google_cloud_run_v2_service` pertencem a providers. O que da para deixar neutro e a topologia,
quais servicos existem, o que cada um expoe e como se ligam. Escolher a nuvem passa a mexer em
uma amarracao, e nao nesta declaracao.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

INFRA = Path(__file__).resolve().parents[2] / "infra"
PROVEDORES = ("railway_", "aws_", "google_", "azurerm_", "kubernetes_", "docker_", "heroku_")


def _hcl() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted(INFRA.glob("*.tf")))


@pytest.mark.parametrize("prefixo", PROVEDORES)
def test_no_resource_belongs_to_a_cloud_provider(prefixo: str) -> None:
    for tipo in re.findall(r'^resource\s+"([a-z0-9_]+)"', _hcl(), re.MULTILINE):
        assert not tipo.startswith(prefixo), f"{tipo} amarra a topologia a um provedor"


def test_the_configuration_needs_no_provider_block() -> None:
    """Sem provider, `terraform plan` roda no CI sem credencial e sem escolha ainda nao feita."""
    assert "required_providers" not in _hcl()
    assert not re.search(r'^provider\s+"', _hcl(), re.MULTILINE)


def test_every_service_declares_what_a_deployment_needs_to_know() -> None:
    texto = _hcl()

    for campo in ("role", "port", "health_path", "public", "needs"):
        assert campo in texto, f"a topologia nao declara {campo}"


def test_stores_are_separate_from_services() -> None:
    """Servico se recria e banco nao, entao misturar os dois num mapa so convida a recriar dado."""
    texto = _hcl()

    assert 'variable "stores"' in texto
    assert 'resource "terraform_data" "store"' in texto
