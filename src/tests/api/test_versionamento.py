"""O prefixo de versao so vale se ninguem puder publicar rota fora dele por descuido.

Versionar e barato de declarar e caro de manter: basta uma rota nova registrada no app em vez do
router para o contrato passar a ter duas portas, uma versionada e outra nao, e a segunda so
aparece quando alguem publica a v2 e ela nao muda. Por isso a regra e verificada, e nao confiada.
"""

from __future__ import annotations

import pytest

from api import PREFIXO, VERSAO, VERSAO_MAIOR, create_app

from .apoio import cliente_autenticado

OPERACAO = frozenset({"/", "/bruno.zip", "/health"})


def caminhos() -> set[str]:
    return set(create_app().openapi()["paths"])


def test_every_business_route_lives_under_the_version_prefix() -> None:
    fora = {c for c in caminhos() if c not in OPERACAO and not c.startswith(f"{PREFIXO}/")}

    assert not fora, f"rota de contrato publicada fora de {PREFIXO}: {sorted(fora)}"


@pytest.mark.parametrize("caminho", sorted(OPERACAO))
def test_the_operational_routes_stay_outside_the_version(caminho: str) -> None:
    """Um orquestrador que consulta saude nao deveria saber em que versao o contrato esta.

    Com `/health` dentro do prefixo, publicar a v2 obrigaria a reconfigurar o healthcheck de toda
    implantacao, por um motivo que nada tem a ver com o contrato.
    """
    assert caminho in caminhos()
    assert not caminho.startswith(f"{PREFIXO}/")


def test_the_documented_version_and_the_url_prefix_cannot_disagree() -> None:
    """Dois numeros de versao em dois lugares e a forma mais silenciosa de mentir sobre contrato."""
    documento = create_app().openapi()

    assert documento["info"]["version"] == VERSAO
    assert VERSAO.split(".")[0] == str(VERSAO_MAIOR)
    assert f"/v{VERSAO_MAIOR}" == PREFIXO


def test_the_upload_url_the_service_mints_points_at_the_versioned_route() -> None:
    """O front faz PUT no endereco que recebe: se ele vier sem versao, o PUT cai em 404."""
    with cliente_autenticado(create_app()) as cliente:
        corpo = cliente.post(
            f"{PREFIXO}/videos/upload-url",
            json={"fileName": "c.mp4", "contentType": "video/mp4"},
        ).json()

    assert f"{PREFIXO}/videos/upload/" in corpo["uploadUrl"]


def test_an_unversioned_call_finds_the_way_instead_of_a_bare_404() -> None:
    """Chamar sem versao e o erro mais provavel de quem integra, e a raiz responde onde ir."""
    with cliente_autenticado(create_app()) as cliente:
        assert cliente.post("/videos/upload-url", json={}).status_code == 404

        raiz = cliente.get("/").json()

    assert raiz["atual"] == PREFIXO
    assert PREFIXO in raiz["versoes"]
    assert raiz["docs"] == "/docs"
    assert raiz["bruno"] == "/bruno.zip"
