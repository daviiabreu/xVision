"""O cliente autenticado que os testes da API usam, importado de proposito e nao herdado.

Um `conftest.py` nesta pasta serviria a mesma coisa sem o import, e foi exatamente esse tipo de
heranca invisivel que fez a suite depender da ordem de coleta. Uma linha de import custa menos.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import auth, store

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
