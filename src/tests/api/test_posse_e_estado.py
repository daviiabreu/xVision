"""A posse do endereco de envio e a maquina de estados, que e o que impede resultado trocado.

Medido no servico anterior: dois `PUT` no mesmo token passavam os dois, e com a fila ocupada isso
acontecia 100% das vezes mesmo sem simultaneidade, porque a janela entre ler o estado e gravar os
bytes e a fila inteira. Bastava um cliente repetir depois de um timeout. O segundo envio apagava
o arquivo que o worker estava lendo, marcava `Falhou`, e depois o worker gravava `Finalizado` por
cima: o registro terminava contando a historia errada, com o motivo culpando o arquivo de quem
enviou.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api import PREFIXO, create_app, execution, store
from edge.testing.make_clip import make_synthetic_clip

from .apoio import cliente_autenticado

MP4 = "video/mp4"


@pytest.fixture
def cliente() -> Iterator[TestClient]:
    with cliente_autenticado(create_app()) as aberto:
        yield aberto


@pytest.fixture
def video(tmp_path: Path) -> bytes:
    return make_synthetic_clip(tmp_path / "s.mp4").read_bytes()


def _endereco(cliente: TestClient) -> tuple[str, int]:
    corpo = cliente.post(
        f"{PREFIXO}/videos/upload-url", json={"fileName": "c.mp4", "contentType": MP4}
    ).json()
    return corpo["uploadUrl"], corpo["id"]


def test_only_one_upload_can_claim_a_given_address() -> None:
    """A posse sai de um UPDATE condicional, entao ler e depois gravar nao tem janela."""
    criado = store.criar("c.mp4", MP4)

    primeiro = store.reivindicar(criado.token)
    segundo = store.reivindicar(criado.token)

    assert primeiro is not None
    assert primeiro.id == criado.id
    assert segundo is None, "o segundo envio tomou posse de um endereco ja reivindicado"


def test_an_unknown_address_claims_nothing() -> None:
    assert store.reivindicar("token-que-ninguem-emitiu") is None


def test_the_second_upload_is_refused_and_does_not_touch_the_first(
    cliente: TestClient, video: bytes
) -> None:
    """O segundo `PUT` nao pode apagar o arquivo que o worker do primeiro esta lendo."""
    url, identificador = _endereco(cliente)

    primeiro = cliente.put(url, content=video)
    segundo = cliente.put(url, content=video)

    assert primeiro.status_code == 202, primeiro.text
    assert segundo.status_code == 409, segundo.text
    lido = store.ler(identificador)
    assert lido is not None
    assert lido.estado in {store.PENDENTE, store.PROCESSANDO, store.FINALIZADO}
    assert lido.estado != store.FALHOU, "o reenvio derrubou o trabalho que ja estava em curso"


@pytest.mark.parametrize("terminal", [store.FINALIZADO, store.FALHOU])
def test_a_terminal_state_cannot_be_left(terminal: str) -> None:
    """Sem isto, o worker gravava `Finalizado` por cima de um `Falhou` ja registrado."""
    criado = store.criar("c.mp4", MP4)
    store.marcar(criado.id, store.PROCESSANDO)
    store.marcar(criado.id, terminal)

    for destino in store.ESTADOS:
        with pytest.raises(store.TransicaoInvalida, match=terminal):
            store.marcar(criado.id, destino)


def test_a_state_cannot_be_skipped_backwards() -> None:
    criado = store.criar("c.mp4", MP4)
    store.marcar(criado.id, store.PROCESSANDO)

    with pytest.raises(store.TransicaoInvalida):
        store.marcar(criado.id, store.PENDENTE)


def test_marking_a_video_that_does_not_exist_is_refused() -> None:
    with pytest.raises(store.StoreError, match="4242"):
        store.marcar(4242, store.PROCESSANDO)


def test_an_interrupted_upload_leaves_no_file_and_no_waiting_job(
    cliente: TestClient, tmp_path: Path
) -> None:
    """Envio recusado no meio nao deixa arquivo no disco nem registro esperando para sempre."""
    url, identificador = _endereco(cliente)

    resposta = cliente.put(url, content=b"")

    assert resposta.status_code == 400
    lido = store.ler(identificador)
    assert lido is not None
    assert lido.estado == store.FALHOU
    assert lido.motivo is not None
    assert list((tmp_path / "var" / "uploads").glob("*")) == []


def test_the_reason_a_video_failed_reaches_whoever_sent_it(cliente: TestClient) -> None:
    """O campo existia, era documentado no OpenAPI, e nenhuma rota conseguia devolve-lo."""
    url, identificador = _endereco(cliente)
    cliente.put(url, content=b"nao sou um video")

    status = cliente.get(f"{PREFIXO}/videos/status/{identificador}")
    resultado = cliente.get(f"{PREFIXO}/videos/{identificador}")

    assert status.json()["video_status"] == store.FALHOU
    assert status.json()["motivo"], "o status nao diz por que falhou"
    assert resultado.status_code == 200, "estado terminal nao pode virar 409 dizendo que falta"
    assert resultado.json()["motivo"], "o resultado nao diz por que falhou"
    assert "ainda nao terminou" not in resultado.text


def test_an_address_that_was_never_used_is_not_treated_as_a_crash() -> None:
    """Endereco emitido e nao usado nao e trabalho interrompido: e cliente que ainda nao mandou.

    O que separa os dois e a posse, e nao o estado. Sem essa distincao, a recuperacao de queda
    marcaria `Falhou` todo endereco que alguem pediu e ainda vai usar.
    """
    emitido = store.criar("c.mp4", MP4)

    assert emitido.id not in {v.id for v in store.recuperaveis()}

    store.reivindicar(emitido.token)

    assert emitido.id in {v.id for v in store.recuperaveis()}


def test_a_video_interrupted_mid_processing_is_recovered(tmp_path: Path, video: bytes) -> None:
    """A queda de verdade: o registro fica em `Em processamento` e o arquivo sobrevive."""
    criado = store.criar("c.mp4", MP4)
    store.reivindicar(criado.token)
    store.marcar(criado.id, store.PROCESSANDO)
    enviados = tmp_path / "var" / "uploads"
    enviados.mkdir(parents=True, exist_ok=True)
    (enviados / f"{criado.id}.mp4").write_bytes(video)

    assert criado.id in {v.id for v in store.recuperaveis()}

    with TestClient(create_app()):
        execution.aguardar()

    lido = store.ler(criado.id)
    assert lido is not None
    assert lido.estado in store.TERMINAIS, "a queda no meio do processamento nao foi recuperada"
