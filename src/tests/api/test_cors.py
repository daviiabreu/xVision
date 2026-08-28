"""O 413 do teto de corpo tem de sair com cabecalho de CORS, senao o navegador nao le a recusa.

O middleware de tamanho recusa antes de a rota existir, e `add_middleware` empilha ao contrario:
quem se registra por ultimo fica por fora. Com o CORS por dentro, a resposta de 413 saia sem
`access-control-allow-origin`, o `fetch` do frontend rejeitava sem status e a tela dizia falha de
conexao para um arquivo grande demais. O caso so aparece quando os dois tetos divergem, que e o
que a variavel de ambiente permite.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from api import PREFIXO, create_app

from .apoio import cliente_autenticado

ORIGEM = "http://localhost:3000"


@pytest.fixture
def cliente(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("XVISION_API_MAX_UPLOAD_MB", "1")
    monkeypatch.setenv("XVISION_API_MAX_JSON_MB", "1")
    with cliente_autenticado(create_app()) as aberto:
        yield aberto


def test_o_413_do_teto_de_corpo_sai_com_cors(cliente: TestClient) -> None:
    resposta = cliente.put(
        f"{PREFIXO}/videos/upload/qualquer",
        content=b"x" * (2 * 1024 * 1024),
        headers={"Origin": ORIGEM, "Content-Type": "video/mp4"},
    )

    assert resposta.status_code == 413
    assert resposta.headers.get("access-control-allow-origin") == ORIGEM


def test_a_recusa_de_credencial_tambem_sai_com_cors(cliente: TestClient) -> None:
    resposta = cliente.get(
        f"{PREFIXO}/auth/me",
        headers={"Origin": ORIGEM, "Authorization": "Bearer nao-vale"},
    )

    assert resposta.status_code == 401
    assert resposta.headers.get("access-control-allow-origin") == ORIGEM


def test_o_retry_after_prometido_no_contrato_chega_ao_navegador(cliente: TestClient) -> None:
    """Sem `expose_headers`, o `fetch` do frontend nao enxerga o cabecalho que o 429 promete."""
    resposta = cliente.put(
        f"{PREFIXO}/videos/upload/qualquer",
        content=b"x" * (2 * 1024 * 1024),
        headers={"Origin": ORIGEM, "Content-Type": "video/mp4"},
    )

    expostos = resposta.headers.get("access-control-expose-headers", "")

    assert "Retry-After" in expostos, f"o navegador so le o que esta em {expostos!r}"
