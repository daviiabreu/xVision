"""O cliente autenticado que os testes da API usam, importado de proposito e nao herdado.

Um `conftest.py` nesta pasta serviria a mesma coisa sem o import, e foi exatamente esse tipo de
heranca invisivel que fez a suite depender da ordem de coleta. Uma linha de import custa menos.
"""

from __future__ import annotations

import httpx2
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import PREFIXO, auth, store

SERVICO = "servico-de-teste"


def cliente_autenticado(app: FastAPI, nome: str = SERVICO) -> TestClient:
    """Um principal de servico por chamada, com o token no cabecalho.

    Servico, e nao usuario, porque a maioria dos testes nao tem nada a ver com senha: usar conta
    faria cada um deles depender do `scrypt`, que e lento de proposito.
    """
    token = auth.novo_token()
    store.criar_principal("servico", nome, token_hash=auth.hash_de_token(token))
    cliente = TestClient(app)
    cliente.headers["Authorization"] = f"Bearer {token}"
    return cliente


def enviar(
    cliente: TestClient, video: bytes, *, nome: str = "c.mp4", tipo: str = "video/mp4"
) -> httpx2.Response:
    """O caminho de envio que existe: reserva o endereco e manda o binario cru nele.

    Muito teste daqui usava o antigo `POST /videos/processamento` por ser uma chamada so, e nao
    por ser base64. Este auxiliar mantem esses testes com uma linha e sem esconder que o envio de
    verdade sao duas requisicoes.
    """
    reserva = cliente.post(
        f"{PREFIXO}/videos/upload-url", json={"fileName": nome, "contentType": tipo}
    )
    if reserva.status_code != 200:
        return reserva
    return cliente.put(reserva.json()["uploadUrl"], content=video, headers={"Content-Type": tipo})
