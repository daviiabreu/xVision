from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from api import PREFIXO, auth, create_app, store

from .apoio import cliente_autenticado

SENHA_ATUAL = "uma-senha-atual-bem-longa"
SENHA_NOVA = "uma-senha-nova-ainda-melhor"


@pytest.fixture
def cliente() -> Iterator[TestClient]:
    with TestClient(create_app()) as aberto:
        yield aberto


def _criar_e_entrar(
    cliente: TestClient, email: str = "conta@exemplo.com"
) -> tuple[int, dict[str, str]]:
    criado = cliente.post(f"{PREFIXO}/auth/register", json={"email": email, "senha": SENHA_ATUAL})
    entrada = cliente.post(f"{PREFIXO}/auth/login", json={"email": email, "senha": SENHA_ATUAL})
    return criado.json()["id"], {"Authorization": f"Bearer {entrada.json()['access_token']}"}


def test_a_conta_troca_o_email_sem_perder_os_videos(cliente: TestClient) -> None:
    principal_id, headers = _criar_e_entrar(cliente)
    video = store.criar("jogada.mp4", "video/mp4", principal_id=principal_id)

    resposta = cliente.patch(
        f"{PREFIXO}/auth/me",
        headers=headers,
        json={"email": "  NOVO@exemplo.com ", "senha_atual": SENHA_ATUAL},
    )

    assert resposta.status_code == 200
    assert resposta.json() == {"id": principal_id, "tipo": "usuario", "nome": "novo@exemplo.com"}
    assert cliente.get(f"{PREFIXO}/auth/me", headers=headers).json()["nome"] == "novo@exemplo.com"
    video_persistido = store.ler(video.id)
    assert video_persistido is not None
    assert video_persistido.principal_id == principal_id
    assert (
        cliente.post(
            f"{PREFIXO}/auth/login",
            json={"email": "conta@exemplo.com", "senha": SENHA_ATUAL},
        ).status_code
        == 401
    )
    assert (
        cliente.post(
            f"{PREFIXO}/auth/login",
            json={"email": "novo@exemplo.com", "senha": SENHA_ATUAL},
        ).status_code
        == 200
    )


def test_o_email_nao_muda_com_senha_errada_ou_endereco_em_uso(cliente: TestClient) -> None:
    _, headers = _criar_e_entrar(cliente)
    cliente.post(
        f"{PREFIXO}/auth/register",
        json={"email": "ocupado@exemplo.com", "senha": SENHA_ATUAL},
    )

    senha_errada = cliente.patch(
        f"{PREFIXO}/auth/me",
        headers=headers,
        json={"email": "outro@exemplo.com", "senha_atual": "senha-incorreta-bem-longa"},
    )
    repetido = cliente.patch(
        f"{PREFIXO}/auth/me",
        headers=headers,
        json={"email": "ocupado@exemplo.com", "senha_atual": SENHA_ATUAL},
    )

    assert senha_errada.status_code == 400
    assert senha_errada.json()["detail"] == "a senha atual esta incorreta"
    assert repetido.status_code == 409
    assert cliente.get(f"{PREFIXO}/auth/me", headers=headers).json()["nome"] == "conta@exemplo.com"


def test_a_conta_troca_a_senha_depois_de_confirmar_a_atual(cliente: TestClient) -> None:
    _, headers = _criar_e_entrar(cliente)
    token_antigo = headers["Authorization"].removeprefix("Bearer ")

    resposta = cliente.post(
        f"{PREFIXO}/auth/password/change",
        headers=headers,
        json={"senha_atual": SENHA_ATUAL, "nova_senha": SENHA_NOVA},
    )

    assert resposta.status_code == 200
    assert resposta.json()["mensagem"] == "Senha atualizada com sucesso."
    token_novo = resposta.json()["access_token"]
    assert token_novo != token_antigo
    assert cliente.get(f"{PREFIXO}/auth/me", headers=headers).status_code == 401
    assert (
        cliente.get(
            f"{PREFIXO}/auth/me",
            headers={"Authorization": f"Bearer {token_novo}"},
        ).status_code
        == 200
    )
    assert (
        cliente.post(
            f"{PREFIXO}/auth/login",
            json={"email": "conta@exemplo.com", "senha": SENHA_ATUAL},
        ).status_code
        == 401
    )
    assert (
        cliente.post(
            f"{PREFIXO}/auth/login",
            json={"email": "conta@exemplo.com", "senha": SENHA_NOVA},
        ).status_code
        == 200
    )


@pytest.mark.parametrize("mudanca", ["email", "senha"])
def test_alterar_credenciais_invalida_links_de_redefinicao_abertos(
    cliente: TestClient, mudanca: str
) -> None:
    principal_id, headers = _criar_e_entrar(cliente)
    token = auth.novo_token()
    principal = store.ler_principal(principal_id)
    assert principal is not None
    store.criar_redefinicao_senha(
        principal,
        auth.hash_de_token(token),
        2_000_000_000,
    )

    if mudanca == "email":
        alteracao = cliente.patch(
            f"{PREFIXO}/auth/me",
            headers=headers,
            json={"email": "novo@exemplo.com", "senha_atual": SENHA_ATUAL},
        )
    else:
        alteracao = cliente.post(
            f"{PREFIXO}/auth/password/change",
            headers=headers,
            json={"senha_atual": SENHA_ATUAL, "nova_senha": SENHA_NOVA},
        )

    redefinicao = cliente.post(
        f"{PREFIXO}/auth/password/reset",
        json={"token": token, "senha": "uma-terceira-senha-bem-longa"},
    )

    assert alteracao.status_code == 200
    assert redefinicao.status_code == 400


def test_a_nova_senha_precisa_ser_valida_e_diferente(cliente: TestClient) -> None:
    _, headers = _criar_e_entrar(cliente)

    curta = cliente.post(
        f"{PREFIXO}/auth/password/change",
        headers=headers,
        json={"senha_atual": SENHA_ATUAL, "nova_senha": "curta"},
    )
    igual = cliente.post(
        f"{PREFIXO}/auth/password/change",
        headers=headers,
        json={"senha_atual": SENHA_ATUAL, "nova_senha": SENHA_ATUAL},
    )

    assert curta.status_code == 422
    assert igual.status_code == 400
    assert "diferente" in igual.json()["detail"]


def test_as_estatisticas_contam_todos_os_estados_da_conta(cliente: TestClient) -> None:
    principal_id, headers = _criar_e_entrar(cliente)
    store.criar("pendente.mp4", "video/mp4", principal_id=principal_id)
    processando = store.criar("processando.mp4", "video/mp4", principal_id=principal_id)
    finalizado = store.criar("finalizado.mp4", "video/mp4", principal_id=principal_id)
    falhou = store.criar("falhou.mp4", "video/mp4", principal_id=principal_id)
    store.marcar(processando.id, store.PROCESSANDO)
    store.marcar(finalizado.id, store.PROCESSANDO)
    store.marcar(finalizado.id, store.FINALIZADO, resultados=[])
    store.marcar(falhou.id, store.FALHOU, motivo="falha de teste")

    resposta = cliente.get(f"{PREFIXO}/auth/me/statistics", headers=headers)

    assert resposta.status_code == 200
    assert resposta.json() == {
        "total": 4,
        "pendentes": 1,
        "em_processamento": 1,
        "finalizadas": 1,
        "falhas": 1,
    }


@pytest.mark.parametrize(
    ("metodo", "caminho", "corpo"),
    [
        ("patch", "/auth/me", {"email": "novo@exemplo.com", "senha_atual": SENHA_ATUAL}),
        (
            "post",
            "/auth/password/change",
            {"senha_atual": SENHA_ATUAL, "nova_senha": SENHA_NOVA},
        ),
    ],
)
def test_token_de_servico_nao_edita_uma_conta(
    metodo: str, caminho: str, corpo: dict[str, str]
) -> None:
    with cliente_autenticado(create_app()) as servico:
        resposta = servico.request(metodo, PREFIXO + caminho, json=corpo)

    assert resposta.status_code == 403
