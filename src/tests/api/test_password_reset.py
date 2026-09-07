from __future__ import annotations

import asyncio
import time
import zipfile
from io import BytesIO
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi import BackgroundTasks, Request
from fastapi.testclient import TestClient

from api import (
    PREFIXO,
    _rota_esqueci_senha,  # pyright: ignore[reportPrivateUsage]
    auth,
    create_app,
    recuperacao,
    store,
    zerar_tentativas,
)
from api.schemas import EsqueciSenhaRequest

EMAIL = "pessoa@exemplo.com"
SENHA_ANTIGA = "uma-senha-antiga-de-verdade"
SENHA_NOVA = "uma-senha-nova-de-verdade"


def _configurar_email(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(recuperacao.SMTP_HOST_ENV, "smtp.exemplo.local")
    monkeypatch.setenv(recuperacao.SMTP_FROM_ENV, "xvision@exemplo.local")
    monkeypatch.setenv(recuperacao.PUBLIC_URL_ENV, "https://xvision.exemplo.local")


def test_password_recovery_reports_when_delivery_is_not_configured() -> None:
    with TestClient(create_app()) as cliente:
        resposta = cliente.post(f"{PREFIXO}/auth/password/forgot", json={"email": EMAIL})

    assert resposta.status_code == 503
    assert "nao foi configurada" in resposta.json()["detail"]


def test_a_reset_link_changes_the_password_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configurar_email(monkeypatch)
    enviados: list[tuple[str, str]] = []

    def registrar_envio(destino: str, link: str) -> None:
        enviados.append((destino, link))

    monkeypatch.setattr(recuperacao, "_enviar", registrar_envio)

    with TestClient(create_app()) as cliente:
        assert (
            cliente.post(
                f"{PREFIXO}/auth/register", json={"email": EMAIL, "senha": SENHA_ANTIGA}
            ).status_code
            == 201
        )
        pedido = cliente.post(f"{PREFIXO}/auth/password/forgot", json={"email": EMAIL})

        assert pedido.status_code == 202
        assert pedido.json()["mensagem"].startswith("Se o email")
        assert enviados[0][0] == EMAIL
        token = parse_qs(urlparse(enviados[0][1]).query)["token"][0]
        entrada_antes = cliente.post(
            f"{PREFIXO}/auth/login", json={"email": EMAIL, "senha": SENHA_ANTIGA}
        )
        sessao_antiga = {"Authorization": f"Bearer {entrada_antes.json()['access_token']}"}

        troca = cliente.post(
            f"{PREFIXO}/auth/password/reset",
            json={"token": token, "senha": SENHA_NOVA},
        )
        repetida = cliente.post(
            f"{PREFIXO}/auth/password/reset",
            json={"token": token, "senha": "outra-senha-nova-de-verdade"},
        )
        antiga = cliente.post(f"{PREFIXO}/auth/login", json={"email": EMAIL, "senha": SENHA_ANTIGA})
        nova = cliente.post(f"{PREFIXO}/auth/login", json={"email": EMAIL, "senha": SENHA_NOVA})
        sessao_revogada = cliente.get(f"{PREFIXO}/auth/me", headers=sessao_antiga)

    assert troca.status_code == 200
    assert repetida.status_code == 400
    assert antiga.status_code == 401
    assert nova.status_code == 200
    assert sessao_revogada.status_code == 401


def test_an_unknown_email_gets_the_same_public_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configurar_email(monkeypatch)
    enviados: list[str] = []

    def registrar_destino(destino: str, _: str) -> None:
        enviados.append(destino)

    monkeypatch.setattr(recuperacao, "_enviar", registrar_destino)

    with TestClient(create_app()) as cliente:
        resposta = cliente.post(
            f"{PREFIXO}/auth/password/forgot", json={"email": "ninguem@exemplo.com"}
        )

    assert resposta.status_code == 202
    assert resposta.json()["mensagem"].startswith("Se o email")
    assert enviados == []


@pytest.mark.parametrize("change", ["email", "password", "revocation"])
def test_a_queued_recovery_cannot_outlive_a_credential_change(
    monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    _configurar_email(monkeypatch)
    principal = store.criar_principal("usuario", EMAIL, senha_hash=auth.hash_de_senha(SENHA_ANTIGA))
    tarefas = BackgroundTasks()
    request = Request({"type": "http", "headers": [], "client": ("127.0.0.1", 12345)})
    _rota_esqueci_senha(EsqueciSenhaRequest(email=EMAIL), request, tarefas)
    if change == "email":
        store.atualizar_nome_principal(principal.id, "novo@exemplo.com")
    elif change == "password":
        store.atualizar_senha_principal(principal.id, auth.hash_de_senha(SENHA_NOVA))
    else:
        store.revogar_principal(principal.id)
    enviados: list[str] = []

    def registrar_destino(email: str, _: str) -> None:
        enviados.append(email)

    monkeypatch.setattr(recuperacao, "_enviar", registrar_destino)

    asyncio.run(tarefas())

    assert enviados == []
    with store.conexao() as conn:
        assert conn.execute("SELECT COUNT(*) FROM redefinicoes_senha").fetchone()[0] == 0


def test_an_expired_token_does_not_change_the_password() -> None:
    principal = store.criar_principal("usuario", EMAIL, senha_hash=auth.hash_de_senha(SENHA_ANTIGA))
    token = auth.novo_token()
    store.criar_redefinicao_senha(principal, auth.hash_de_token(token), int(time.time()) - 1)

    with TestClient(create_app()) as cliente:
        resposta = cliente.post(
            f"{PREFIXO}/auth/password/reset",
            json={"token": token, "senha": SENHA_NOVA},
        )

    assert resposta.status_code == 400


def test_password_recovery_is_rate_limited(monkeypatch: pytest.MonkeyPatch) -> None:
    zerar_tentativas()
    _configurar_email(monkeypatch)
    monkeypatch.setenv("XVISION_REDEFINICAO_TENTATIVAS", "1")

    with TestClient(create_app()) as cliente:
        primeira = cliente.post(
            f"{PREFIXO}/auth/password/forgot", json={"email": "ninguem@exemplo.com"}
        )
        segunda = cliente.post(
            f"{PREFIXO}/auth/password/forgot", json={"email": "outra@exemplo.com"}
        )

    assert primeira.status_code == 202
    assert segunda.status_code == 429


def test_trusted_proxy_keeps_rate_limits_separate_by_forwarded_ip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zerar_tentativas()
    _configurar_email(monkeypatch)
    monkeypatch.setenv("XVISION_TRUST_PROXY", "1")
    monkeypatch.setenv("XVISION_REDEFINICAO_TENTATIVAS", "1")

    with TestClient(create_app()) as cliente:
        primeira = cliente.post(
            f"{PREFIXO}/auth/password/forgot",
            json={"email": "ninguem@exemplo.com"},
            headers={"X-Forwarded-For": "198.51.100.10"},
        )
        segunda = cliente.post(
            f"{PREFIXO}/auth/password/forgot",
            json={"email": "outra@exemplo.com"},
            headers={"X-Forwarded-For": "198.51.100.11"},
        )

    assert primeira.status_code == 202
    assert segunda.status_code == 202


@pytest.mark.parametrize(
    ("variavel", "valor"),
    [
        (recuperacao.SMTP_PORT_ENV, "porta"),
        (recuperacao.TTL_ENV, "agora"),
        (recuperacao.PUBLIC_URL_ENV, "endereco-sem-esquema"),
    ],
)
def test_invalid_email_delivery_configuration_refuses_to_start(
    monkeypatch: pytest.MonkeyPatch, variavel: str, valor: str
) -> None:
    _configurar_email(monkeypatch)
    monkeypatch.setenv(variavel, valor)

    with (
        pytest.raises(RuntimeError, match="recuperacao de senha invalida"),
        TestClient(create_app()),
    ):
        pass


def test_the_downloaded_bruno_collection_contains_the_requests() -> None:
    with TestClient(create_app()) as cliente:
        resposta = cliente.get("/bruno.zip")

    assert resposta.status_code == 200
    assert resposta.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(BytesIO(resposta.content)) as arquivo:
        nomes = set(arquivo.namelist())
    assert "xvision/bruno.json" in nomes
    assert "xvision/pedir-url-de-envio.bru" in nomes
    assert "xvision/auth-reset-password.bru" in nomes
