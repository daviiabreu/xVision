"""O endereco que o servico cunha nao pode vir do cabecalho que o cliente mandou.

`uploadUrl` e `render_url` sairam absolutos, montados com `str(request.base_url)`, que o Starlette
deriva do `Host`. Quem escolhe o `Host` e quem chama, entao o corpo da resposta carregava um dado
de fora: atras de proxy que reescreve o cabecalho o navegador recebia o nome interno do servico, e
com TLS terminado acima recebia `http://`, que o navegador recusa como conteudo misto. Nenhum
teste cobria isso, e por isso a troca para caminho relativo passou sem nada ficar vermelho.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api import PREFIXO, create_app, render, store
from edge.paths import var_dir

from .apoio import cliente_autenticado

FORJADO = "servico-interno.svc.cluster.local:9999"


@pytest.fixture(autouse=True)
def isolamento(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("XVISION_VAR_DIR", str(tmp_path))
    yield


@pytest.fixture
def cliente() -> Iterator[TestClient]:
    with cliente_autenticado(create_app()) as aberto:
        yield aberto


def _reservar(cliente: TestClient, host: str | None = None) -> dict[str, object]:
    resposta = cliente.post(
        f"{PREFIXO}/videos/upload-url",
        json={"fileName": "c.mp4", "contentType": "video/mp4"},
        headers={"Host": host} if host else None,
    )

    assert resposta.status_code == 200
    return resposta.json()


def test_o_endereco_de_envio_sai_relativo(cliente: TestClient) -> None:
    corpo = _reservar(cliente)

    assert str(corpo["uploadUrl"]).startswith(f"{PREFIXO}/videos/upload/")


@pytest.mark.parametrize("host", [FORJADO, "outro.exemplo.com"])
def test_o_host_do_cliente_nao_entra_no_endereco_de_envio(cliente: TestClient, host: str) -> None:
    corpo = _reservar(cliente, host)

    assert host not in str(corpo["uploadUrl"])
    assert "://" not in str(corpo["uploadUrl"])


def test_o_endereco_do_render_sai_relativo_e_sem_o_host_do_cliente(
    cliente: TestClient, tmp_path: Path
) -> None:
    corpo = _reservar(cliente)
    video_id = int(str(corpo["id"]))
    store.marcar(video_id, "Em processamento")
    store.marcar(video_id, "Finalizado")
    destino = render.render_path(video_id)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(b"nao e video de verdade, e o teste so olha o endereco")

    resultado = cliente.get(f"{PREFIXO}/videos/{video_id}", headers={"Host": FORJADO})

    assert resultado.status_code == 200
    endereco = str(resultado.json()["render_url"])
    assert endereco == f"{PREFIXO}/videos/{video_id}/render"
    assert FORJADO not in endereco
    assert var_dir() == tmp_path
