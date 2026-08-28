"""O contrato e conferido batendo na rota, e nao comparando duas listas escritas a mao.

O teste anterior fazia `esperados <= respostas`, com as duas listas vindas do proprio codigo e
sem nenhuma requisicao. Ele era estruturalmente incapaz de ver um codigo que a rota devolve e o
contrato nao declara, e de fato nao viu: o `PUT` respondia 415 sem declarar, enquanto o README
afirmava que havia teste para isso.

Aqui cada recusa e provocada de verdade, e o codigo observado tem de estar declarado. Um cliente
gerado a partir do `openapi.json` so trata o que esta la.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api import PREFIXO, create_app
from edge.testing.make_clip import make_synthetic_clip

from .apoio import cliente_autenticado

MP4 = "video/mp4"


@pytest.fixture
def cliente() -> Iterator[TestClient]:
    with cliente_autenticado(create_app()) as aberto:
        yield aberto


@pytest.fixture
def declarado() -> dict[tuple[str, str], set[int]]:
    documento = create_app().openapi()
    return {
        (caminho, metodo): {int(codigo) for codigo in corpo["responses"]}
        for caminho, metodos in documento["paths"].items()
        for metodo, corpo in metodos.items()
    }


def _endereco(cliente: TestClient) -> tuple[str, int]:
    corpo = cliente.post(
        f"{PREFIXO}/videos/upload-url", json={"fileName": "c.mp4", "contentType": MP4}
    ).json()
    return corpo["uploadUrl"], corpo["id"]


def _provocacoes(cliente: TestClient, tmp_path: Path) -> list[tuple[str, str, int]]:
    """Uma requisicao por recusa que o servico sabe dar, mais o caminho feliz de cada rota."""
    feliz, _ = _endereco(cliente)
    vazio, _ = _endereco(cliente)
    nao_video, _ = _endereco(cliente)
    bytes_do_video = make_synthetic_clip(tmp_path / "s.mp4").read_bytes()
    upload = PREFIXO + "/videos/upload/{token}"
    reserva = f"{PREFIXO}/videos/upload-url"

    return [
        (
            reserva,
            "post",
            cliente.post(reserva, json={"fileName": "c", "contentType": MP4}).status_code,
        ),
        (
            reserva,
            "post",
            cliente.post(reserva, json={"fileName": "c", "contentType": "text/plain"}).status_code,
        ),
        (reserva, "post", cliente.post(reserva, json={}).status_code),
        (
            upload,
            "put",
            cliente.put(f"{PREFIXO}/videos/upload/nao-existe", content=b"x").status_code,
        ),
        (upload, "put", cliente.put(vazio, content=b"").status_code),
        (upload, "put", cliente.put(nao_video, content=b"nao sou video").status_code),
        (upload, "put", cliente.put(feliz, content=bytes_do_video).status_code),
        (upload, "put", cliente.put(feliz, content=bytes_do_video).status_code),
        (
            PREFIXO + "/videos/status/{video_id}",
            "get",
            cliente.get(f"{PREFIXO}/videos/status/999").status_code,
        ),
        (PREFIXO + "/videos/{video_id}", "get", cliente.get(f"{PREFIXO}/videos/999").status_code),
    ]


def test_no_route_answers_a_code_the_contract_does_not_declare(
    cliente: TestClient, tmp_path: Path, declarado: dict[tuple[str, str], set[int]]
) -> None:
    fora = [
        (caminho, metodo, codigo)
        for caminho, metodo, codigo in _provocacoes(cliente, tmp_path)
        if codigo not in declarado[(caminho, metodo)]
    ]

    assert not fora, f"codigo devolvido e nao declarado no openapi.json: {fora}"


def test_the_probe_really_exercises_more_than_the_happy_path(
    cliente: TestClient, tmp_path: Path
) -> None:
    """Sem esta guarda, uma sonda que so recebesse 200 passaria no teste acima sem provar nada."""
    codigos = {codigo for _, _, codigo in _provocacoes(cliente, tmp_path)}

    assert {400, 404, 409, 415, 422} <= codigos, (
        f"a sonda nao provocou recusa suficiente: {codigos}"
    )
    assert 202 in codigos, "a sonda nao exercitou o caminho feliz de envio"
