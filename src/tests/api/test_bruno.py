"""A colecao do Bruno e documentacao executavel, e documentacao envelhece sozinha.

O `openapi.json` ja e conferido contra o codigo, entao o que falta cobrar e a outra ponta: rota
nova sem requisicao correspondente deixa quem usa a colecao achando que viu a API inteira.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from api import PREFIXO, create_app

COLECAO = Path(__file__).resolve().parents[2] / "api" / "docs" / "xvision"

REQUISICAO = {
    ("/", "get"): "versoes.bru",
    ("/health", "get"): "health.bru",
    ("/bruno.zip", "get"): "bruno-download.bru",
    (f"{PREFIXO}/auth/register", "post"): "auth-register.bru",
    (f"{PREFIXO}/auth/login", "post"): "auth-login.bru",
    (f"{PREFIXO}/auth/password/forgot", "post"): "auth-forgot-password.bru",
    (f"{PREFIXO}/auth/password/reset", "post"): "auth-reset-password.bru",
    (f"{PREFIXO}/auth/password/change", "post"): "auth-change-password.bru",
    (f"{PREFIXO}/auth/me", "get"): "auth-me.bru",
    (f"{PREFIXO}/auth/me", "patch"): "auth-update-email.bru",
    (f"{PREFIXO}/auth/me/statistics", "get"): "auth-statistics.bru",
    (f"{PREFIXO}/videos", "get"): "listar-videos.bru",
    (f"{PREFIXO}/videos/upload-url", "post"): "pedir-url-de-envio.bru",
    (PREFIXO + "/videos/upload/{token}", "put"): "enviar-video.bru",
    (PREFIXO + "/videos/status/{video_id}", "get"): "consultar-status.bru",
    (PREFIXO + "/videos/{video_id}", "get"): "buscar-resultado.bru",
    (PREFIXO + "/videos/{video_id}/render", "get"): "buscar-render.bru",
}


def rotas() -> set[tuple[str, str]]:
    documento = create_app().openapi()
    return {
        (caminho, metodo) for caminho, metodos in documento["paths"].items() for metodo in metodos
    }


def test_the_collection_covers_every_route_the_service_answers() -> None:
    faltando = rotas() - set(REQUISICAO)

    assert not faltando, f"rota sem requisicao no Bruno: {sorted(faltando)}"


def test_the_collection_has_no_request_for_a_route_that_does_not_exist() -> None:
    sobrando = set(REQUISICAO) - rotas()

    assert not sobrando, f"requisicao apontando para rota que saiu: {sorted(sobrando)}"


@pytest.mark.parametrize(("rota", "arquivo"), sorted(REQUISICAO.items()))
def test_every_request_exists_and_declares_the_method_of_its_route(
    rota: tuple[str, str], arquivo: str
) -> None:
    caminho = COLECAO / arquivo

    assert caminho.exists(), f"{arquivo} nao existe na colecao"
    texto = caminho.read_text(encoding="utf-8")
    assert re.search(rf"^{rota[1]} \{{", texto, re.MULTILINE), (
        f"{arquivo} nao declara o metodo {rota[1].upper()}"
    )


def test_the_environment_declares_what_the_requests_read() -> None:
    """Requisicao que le variavel inexistente falha com URL vazia, que confunde mais que ajuda."""
    ambiente = (COLECAO / "environments" / "local.bru").read_text(encoding="utf-8")

    for variavel in ("baseUrl", "resetToken", "videoPath"):
        assert f"{variavel}:" in ambiente
