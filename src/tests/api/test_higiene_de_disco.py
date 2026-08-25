"""Nenhum caminho de recusa pode deixar arquivo no disco, e nenhuma mensagem pode vazar caminho.

Nove mutacoes passaram na suite da API com 99,4% de cobertura de linha. Estas sao as que doem em
producao: envio recusado que deixa ate 128 MB no disco sem nenhum registro apontando para ele,
base64 corrompido aceito em silencio, e o caminho absoluto do servidor voltando ao cliente dentro
do motivo da falha. Cobertura de linha nao viu nenhuma delas, porque as linhas rodavam.
"""

from __future__ import annotations

import base64
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
def uploads(tmp_path: Path) -> Path:
    return tmp_path / "var" / "uploads"


def _endereco(cliente: TestClient) -> str:
    return cliente.post(
        f"{PREFIXO}/videos/upload-url", json={"fileName": "c.mp4", "contentType": MP4}
    ).json()["uploadUrl"]


def _restos(uploads: Path) -> list[str]:
    return sorted(p.name for p in uploads.glob("*")) if uploads.is_dir() else []


def test_a_refused_upload_leaves_nothing_on_disk(
    cliente: TestClient, uploads: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sem o apagamento, cada 413 deixa ate 128 MB parados e repetir enche o disco da borda."""
    monkeypatch.setenv("XVISION_API_MAX_UPLOAD_MB", "1")

    resposta = cliente.put(_endereco(cliente), content=b"x" * (2 * 1024 * 1024))

    assert resposta.status_code == 413
    assert _restos(uploads) == []


def test_a_file_that_does_not_decode_leaves_nothing_on_disk(
    cliente: TestClient, uploads: Path
) -> None:
    resposta = cliente.put(_endereco(cliente), content=b"nao sou um video" * 100)

    assert resposta.status_code == 415
    assert _restos(uploads) == []


def test_the_upload_really_lands_on_disk_before_it_is_cleaned(
    cliente: TestClient, uploads: Path, tmp_path: Path
) -> None:
    """A guarda dos tres testes acima: diretorio vazio nao pode ser a prova de que se limpou.

    Sem isto, mudar o sufixo do arquivo enviado faria os outros passarem com o arquivo intacto
    noutro nome. Aqui o arquivo e observado existindo antes de o worker rodar.
    """
    video = make_synthetic_clip(tmp_path / "s.mp4").read_bytes()
    monkeypatch_alvo: list[str] = []

    original = execution.processar_video

    def espiar(identificador: int, caminho: Path, config: Path, banco: Path | None = None) -> None:
        monkeypatch_alvo.append(caminho.name)
        assert caminho.exists(), "o arquivo enviado nao chegou ao disco"
        original(identificador, caminho, config, banco)

    execution.processar_video = espiar  # pyright: ignore[reportAttributeAccessIssue]
    try:
        cliente.put(_endereco(cliente), content=video)
        execution.aguardar()
    finally:
        execution.processar_video = original  # pyright: ignore[reportAttributeAccessIssue]

    assert monkeypatch_alvo, "o worker nem foi chamado"
    assert _restos(uploads) == []


def test_base64_with_invalid_characters_is_refused_and_not_silently_trimmed(
    cliente: TestClient,
) -> None:
    """Com `validate=False` o decodificador descarta o lixo e devolve bytes truncados.

    O video corrompido entraria na fila e falharia depois, sem ninguem saber que a corrupcao
    aconteceu no transporte. O teste que existia usava entrada que falha por preenchimento, e
    essa levanta nos dois modos.
    """
    valido = base64.b64encode(b"conteudo qualquer para completar o bloco").decode()
    envenenado = valido[:8] + "*!@#" + valido[8:]

    resposta = cliente.post(
        f"{PREFIXO}/videos/processamento", json={"file": envenenado, "type": MP4}
    )

    assert resposta.status_code == 400
    assert "base64" in resposta.json()["detail"]


def test_the_worker_never_writes_the_server_path_into_the_reason(
    cliente: TestClient, tmp_path: Path
) -> None:
    """Existe teste para o gemeo desta funcao na fronteira HTTP, e nao existia para a do worker.

    O `motivo` e gravado no banco e agora chega ao cliente pelas duas rotas de leitura, entao um
    caminho absoluto ali vaza o diretorio do autor e o layout da implantacao.
    """
    quebrado = tmp_path / "nao-e-video.mp4"
    quebrado.write_bytes(b"cabecalho invalido" * 200)
    video = store.criar("c.mp4", MP4)

    execution.processar_video(video.id, quebrado, Path("config/pipeline.toml"))

    lido = store.ler(video.id)
    assert lido is not None
    assert lido.estado == store.FALHOU
    assert lido.motivo is not None
    assert str(tmp_path) not in lido.motivo
    assert "nao-e-video" not in lido.motivo
    assert execution.SEM_CAMINHO in lido.motivo


def test_an_oversized_body_is_refused_by_the_header_and_never_parsed(
    cliente: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Conferir depois do pydantic materializar a string protegia a metade menor do problema.

    Medido no servico anterior: um POST de 300 MB devolvia 413 e levava o processo de 65 MB para
    1176 MB de RSS. A recusa correta chegava, e ela nao ajudava em nada.

    O corpo aqui e JSON invalido de proposito: se o middleware deixasse passar, o pydantic
    responderia 422, e nao 413. O codigo distingue quem recusou.
    """
    monkeypatch.setenv("XVISION_API_MAX_JSON_MB", "1")
    corpo = b"{isto nem e json valido" + b"A" * (2 * 1024 * 1024)

    resposta = cliente.post(
        f"{PREFIXO}/videos/processamento",
        content=corpo,
        headers={"Content-Type": "application/json"},
    )

    assert resposta.status_code == 413, "o corpo foi lido antes de ser recusado"
    assert "upload-url" in resposta.json()["detail"], (
        "a recusa nao diz por onde mandar arquivo grande"
    )
    assert len(resposta.content) < 1000, "a resposta cresceu junto com o pedido"


def test_the_upload_route_has_its_own_larger_ceiling(
    cliente: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Os dois caminhos tem tetos diferentes, e o middleware precisa saber qual deles usar."""
    monkeypatch.setenv("XVISION_API_MAX_JSON_MB", "1")
    monkeypatch.setenv("XVISION_API_MAX_UPLOAD_MB", "8")

    resposta = cliente.put(_endereco(cliente), content=b"x" * (2 * 1024 * 1024))

    assert resposta.status_code != 413, "o PUT foi cortado pelo teto do caminho base64"


def test_the_service_refuses_work_it_cannot_do_instead_of_queueing_forever(
    cliente: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Aceitar trabalho sem teto e prometer resultado que nunca vem.

    Medido no servico anterior: 200 envios aceitos em 0,28 s e 186 arquivos simultaneos em
    `var/uploads`. Com 128 MB por arquivo, sao ate 23 GB de disco em transito, e nada devolvia
    503 nem 429.
    """
    monkeypatch.setenv("XVISION_API_MAX_FILA", "1")
    video = make_synthetic_clip(tmp_path / "s.mp4").read_bytes()

    codigos = [cliente.put(_endereco(cliente), content=video).status_code for _ in range(6)]

    assert 503 in codigos, f"a fila aceitou tudo sem teto: {codigos}"
    recusa = next(
        r
        for r in (cliente.put(_endereco(cliente), content=video) for _ in range(6))
        if r.status_code == 503
    )
    assert recusa.headers.get("Retry-After") == "30", "o 503 nao diz quando voltar"


def test_a_refused_job_does_not_leave_its_file_behind(
    cliente: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, uploads: Path
) -> None:
    monkeypatch.setenv("XVISION_API_MAX_FILA", "1")
    video = make_synthetic_clip(tmp_path / "s.mp4").read_bytes()

    for _ in range(6):
        cliente.put(_endereco(cliente), content=video)
    execution.aguardar()

    assert _restos(uploads) == []
