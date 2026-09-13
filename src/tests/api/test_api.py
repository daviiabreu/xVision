"""O contrato que o frontend consome, verificado contra o que o codigo gera.

O envio e um caminho so: `POST /videos/upload-url` reserva o endereco e o `PUT` manda o binario
cru nele, que e o que `src/interface/src/services/videoService.ts` chama. Houve um segundo, que
recebia o video em base64 dentro do JSON, e saiu quando a interface fechou no `PUT` e ele ficou
sem cliente. Os testes que existiam por causa dele sairam junto; os que so precisavam de uma
chamada so passaram a usar `apoio.enviar`, que faz as duas numa linha.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api import PREFIXO, create_app, execution, store
from edge.testing.make_clip import make_synthetic_clip

from .apoio import cliente_autenticado, enviar

OPENAPI = Path(__file__).resolve().parents[2] / "api" / "docs" / "openapi.json"
MP4 = "video/mp4"


@pytest.fixture(autouse=True)
def isolamento(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Cada teste com o seu banco e o seu `var/`, senao um vaza estado no outro."""
    monkeypatch.setenv("XVISION_VAR_DIR", str(tmp_path / "var"))
    yield
    execution.aguardar()


@pytest.fixture
def cliente() -> TestClient:
    return cliente_autenticado(create_app())


@pytest.fixture
def bytes_do_video(tmp_path: Path) -> bytes:
    return make_synthetic_clip(tmp_path / "s.mp4").read_bytes()


def _terminar(cliente: TestClient, identificador: int) -> dict[str, Any]:
    execution.aguardar()
    resposta = cliente.get(f"{PREFIXO}/videos/{identificador}")
    assert resposta.status_code == 200, resposta.text
    return dict(resposta.json())


def test_health_answers_before_anything_else(cliente: TestClient) -> None:
    """E a rota que o orquestrador consulta para decidir se a implantacao subiu."""
    resposta = cliente.get("/health")

    assert resposta.status_code == 200
    assert resposta.json()["status"] == "ok"


def test_the_upload_url_route_answers_what_the_frontend_reads(cliente: TestClient) -> None:
    """`videoService.ts` le exatamente `uploadUrl` da resposta e faz PUT nele."""
    resposta = cliente.post(
        f"{PREFIXO}/videos/upload-url", json={"fileName": "clipe.mp4", "contentType": MP4}
    )

    assert resposta.status_code == 200, resposta.text
    corpo = resposta.json()
    assert isinstance(corpo["id"], int)
    assert corpo["uploadUrl"].endswith(f"{PREFIXO}/videos/upload/{store.ler(corpo['id']).token}")  # pyright: ignore[reportOptionalMemberAccess]


def test_the_binary_upload_runs_the_pipeline_and_finishes(
    cliente: TestClient, bytes_do_video: bytes
) -> None:
    """O caminho inteiro que o frontend percorre hoje, do pedido de URL ao resultado."""
    reserva = cliente.post(
        f"{PREFIXO}/videos/upload-url", json={"fileName": "clipe.mp4", "contentType": MP4}
    ).json()

    envio = cliente.put(reserva["uploadUrl"], content=bytes_do_video)

    assert envio.status_code == 202, envio.text
    assert envio.json() == {"status": 202, "id": reserva["id"]}
    corpo = _terminar(cliente, reserva["id"])
    assert corpo["video_status"] == store.FINALIZADO
    assert corpo["nome_arquivo"] == "clipe.mp4"
    nomes = {r["nome_processo"] for r in corpo["resultados"]}
    assert {"players", "ball", "field", "shot", "tracking"} <= nomes
    concluidas = [r for r in corpo["resultados"] if r["situacao"] == "ok"]
    assert concluidas
    assert all(r["duracao_ms"] >= 0 for r in corpo["resultados"])
    assert all(str(r["frame"]).startswith("data:image/jpeg;base64,") for r in concluidas)


def test_the_status_route_walks_the_four_states_the_frontend_declares(
    cliente: TestClient, bytes_do_video: bytes
) -> None:
    identificador = enviar(cliente, bytes_do_video).json()["id"]
    execution.aguardar()

    resposta = cliente.get(f"{PREFIXO}/videos/status/{identificador}")

    assert resposta.status_code == 200
    assert resposta.json() == {
        "status": 200,
        "video_status": store.FINALIZADO,
        "motivo": None,
        "etapa_atual": None,
        "etapas_concluidas": 10,
        "etapas_total": 10,
        "progresso": 100,
        "posicao_fila": None,
        "reconsultar_em_s": None,
    }


def test_the_result_is_refused_while_the_video_has_not_finished(cliente: TestClient) -> None:
    """Devolver lista vazia faria o frontend desenhar tela de resultado sem resultado."""
    identificador = cliente.post(
        f"{PREFIXO}/videos/upload-url", json={"fileName": "c.mp4", "contentType": MP4}
    ).json()["id"]

    resposta = cliente.get(f"{PREFIXO}/videos/{identificador}")

    assert resposta.status_code == 409
    assert store.PENDENTE in resposta.json()["detail"]


@pytest.mark.parametrize("status", [True, False])
def test_results_saved_before_stage_metadata_remain_readable(
    cliente: TestClient, status: bool
) -> None:
    identificador = cliente.post(
        f"{PREFIXO}/videos/upload-url", json={"fileName": "legacy.mp4", "contentType": MP4}
    ).json()["id"]
    legado: dict[str, Any] = {
        "nome_processo": "ball",
        "status": status,
        "frame": None,
        "metricas": [],
    }
    store.marcar(identificador, store.PROCESSANDO)
    store.marcar(identificador, store.FINALIZADO, resultados=[legado])

    resposta = cliente.get(f"{PREFIXO}/videos/{identificador}")

    assert resposta.status_code == 200
    resultado = resposta.json()["resultados"][0]
    assert resultado["status"] is status
    assert resultado["situacao"] == ("ok" if status else "failed")
    assert resultado["duracao_ms"] == 0
    persistido = store.ler(identificador)
    assert persistido is not None and persistido.resultados == [legado]


def test_an_unknown_id_and_an_unknown_token_are_both_refused(cliente: TestClient) -> None:
    assert cliente.get(f"{PREFIXO}/videos/status/9999").status_code == 404
    assert cliente.get(f"{PREFIXO}/videos/9999").status_code == 404
    assert cliente.put(f"{PREFIXO}/videos/upload/nao-existe", content=b"x").status_code == 404


def test_the_same_upload_url_cannot_be_used_twice(
    cliente: TestClient, bytes_do_video: bytes
) -> None:
    """O token autoriza gravar uma vez: sem isso, reenviar sobrescreve trabalho em andamento."""
    reserva = cliente.post(
        f"{PREFIXO}/videos/upload-url", json={"fileName": "c.mp4", "contentType": MP4}
    ).json()
    cliente.put(reserva["uploadUrl"], content=bytes_do_video)
    execution.aguardar()

    repetido = cliente.put(reserva["uploadUrl"], content=bytes_do_video)

    assert repetido.status_code == 409


def test_what_is_not_even_a_video_is_refused_at_the_door(cliente: TestClient) -> None:
    """A triagem barata recusa na reserva o que nem video diz ser, antes de gastar upload."""
    for tipo in ("application/pdf", "text/plain", "image/png"):
        resposta = cliente.post(
            f"{PREFIXO}/videos/upload-url", json={"fileName": "x", "contentType": tipo}
        )

        assert resposta.status_code == 415, f"{tipo} passou pela triagem: {resposta.text}"


def test_a_container_other_than_mp4_is_not_refused_by_the_type_alone(cliente: TestClient) -> None:
    """Lista fixa de container recusava `.mov`, que o decodificador desta imagem abre.

    O `.mov` e o caso do iPhone, que grava HEVC nesse container: recusar por lista seria recusar
    o formato mais provavel de um envio feito pelo celular.
    """
    for tipo in ("video/quicktime", "video/x-matroska", "video/webm"):
        resposta = cliente.post(
            f"{PREFIXO}/videos/upload-url", json={"fileName": "c", "contentType": tipo}
        )

        assert resposta.status_code == 200, f"{tipo} recusado pelo tipo: {resposta.text}"


def test_a_renamed_file_claiming_to_be_a_video_is_caught_by_the_decoder(
    cliente: TestClient,
) -> None:
    """`contentType` e afirmacao do cliente: um `.txt` renomeado chega declarando `video/mp4`."""
    resposta = enviar(cliente, b"isto e texto, nao video" * 100)

    assert resposta.status_code == 415
    assert "nao abriu como video" in resposta.json()["detail"]


def test_an_empty_body_is_refused(cliente: TestClient) -> None:
    """Endereco reservado e `PUT` sem bytes e envio interrompido, e nao arquivo vazio aceito."""
    assert enviar(cliente, b"").status_code == 400


def test_a_binary_over_the_ceiling_is_cut_mid_stream(
    cliente: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XVISION_API_MAX_UPLOAD_MB", "1")
    reserva = cliente.post(
        f"{PREFIXO}/videos/upload-url", json={"fileName": "c.mp4", "contentType": MP4}
    ).json()

    resposta = cliente.put(reserva["uploadUrl"], content=b"x" * (2 * 1024 * 1024))

    assert resposta.status_code == 413


def test_a_validation_error_never_echoes_the_body_back(cliente: TestClient) -> None:
    """O pydantic devolve no campo `input` o corpo que recusou, e nada garante que ele seja curto.

    O caso medido, um envio de 10 MB voltando inteiro, morreu com a rota que carregava o video
    dentro do JSON. O que sobra continua ecoando: `fileName` tem teto de 255 caracteres e nada
    impede um cliente de mandar 5 MB de nome, que sem o handler voltariam na recusa.
    """
    nome = "n" * (5 * 1024 * 1024)

    resposta = cliente.post(
        f"{PREFIXO}/videos/upload-url", json={"fileName": nome, "contentType": MP4}
    )

    assert resposta.status_code == 422
    assert nome not in resposta.text
    assert len(resposta.content) < 1000, "a resposta cresceu junto com o pedido"
    assert all("input" not in item for item in resposta.json()["detail"])


def test_a_file_that_is_not_a_video_is_refused_without_leaking_the_server_path(
    cliente: TestClient,
) -> None:
    """A mensagem da borda cita o caminho absoluto, e ele nao pode chegar ao cliente."""
    resposta = enviar(cliente, b"nao sou um video")

    assert resposta.status_code == 415
    detalhe = resposta.json()["detail"]
    assert "/Users/" not in detalhe
    assert "var/uploads" not in detalhe
    assert ".mp4" not in detalhe, "nem o nome do temporario sai daqui"


def test_an_undecodable_upload_does_not_leave_a_job_waiting_forever(cliente: TestClient) -> None:
    """O registro ja existe quando a recusa acontece, e `Pendente` faria o front
    consultar para sempre um trabalho que ninguem vai pegar.
    """
    reserva = cliente.post(
        f"{PREFIXO}/videos/upload-url", json={"fileName": "c.mp4", "contentType": MP4}
    ).json()

    assert cliente.put(reserva["uploadUrl"], content=b"tambem nao sou video").status_code == 415

    lido = store.ler(reserva["id"])
    assert lido is not None
    assert lido.estado == store.FALHOU


def test_the_uploaded_file_is_removed_after_the_pipeline_runs(
    cliente: TestClient, bytes_do_video: bytes, tmp_path: Path
) -> None:
    """Sem isso, cada video enviado fica no disco da borda para sempre."""
    assert enviar(cliente, bytes_do_video).status_code == 202, "o envio nem chegou a acontecer"
    execution.aguardar()

    assert list((tmp_path / "var" / "uploads").glob("*.mp4")) == []


def test_the_browser_is_allowed_to_call_from_the_frontend_origin(cliente: TestClient) -> None:
    """Sem CORS o navegador bloqueia a chamada e ninguem descobre ate integrar."""
    resposta = cliente.options(
        f"{PREFIXO}/videos/upload-url",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert resposta.status_code == 200
    assert resposta.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_an_unknown_origin_is_not_allowed(cliente: TestClient) -> None:
    resposta = cliente.options(
        f"{PREFIXO}/videos/upload-url",
        headers={"Origin": "http://exemplo.invalido", "Access-Control-Request-Method": "POST"},
    )

    assert "access-control-allow-origin" not in resposta.headers


def test_a_video_stuck_in_processing_is_reprocessed_when_the_file_survived(
    tmp_path: Path, bytes_do_video: bytes
) -> None:
    """Trocar o nome do estado nao e recuperar o trabalho.

    A versao anterior devolvia o video para `Pendente` e parava ali. Como ninguem pega trabalho
    pendente sozinho, o frontend passava a girar em `Pendente` em vez de girar em
    `Em processamento`, e o arquivo enviado ficava no disco para sempre.
    """
    video = store.criar("c.mp4", MP4)
    store.reivindicar(video.token)
    store.marcar(video.id, store.PROCESSANDO)
    enviados = tmp_path / "var" / "uploads"
    enviados.mkdir(parents=True, exist_ok=True)
    (enviados / f"{video.id}.mp4").write_bytes(bytes_do_video)

    with TestClient(create_app()):
        execution.aguardar()

    lido = store.ler(video.id)
    assert lido is not None
    assert lido.estado == store.FINALIZADO, "a queda nao foi recuperada: o clipe nao foi refeito"
    assert not (enviados / f"{video.id}.mp4").exists(), "o arquivo enviado ficou no disco"


def test_a_video_stuck_in_processing_fails_with_a_reason_when_the_file_is_gone() -> None:
    """Sem o arquivo nao ha o que reprocessar, e fingir `Pendente` seria consulta eterna."""
    video = store.criar("c.mp4", MP4)
    store.reivindicar(video.token)
    store.marcar(video.id, store.PROCESSANDO)

    with TestClient(create_app()):
        pass

    lido = store.ler(video.id)
    assert lido is not None
    assert lido.estado == store.FALHOU
    assert lido.motivo is not None
    assert "caiu" in lido.motivo


def test_the_versioned_openapi_matches_what_the_code_generates() -> None:
    """O arquivo versionado existe para a revisao ver o contrato no diff, e envelhece sozinho."""
    import json

    assert OPENAPI.exists()
    assert json.loads(OPENAPI.read_text(encoding="utf-8")) == create_app().openapi()


def test_every_documented_refusal_carries_the_same_shape(cliente: TestClient) -> None:
    """Recusa com forma diferente por rota faz o frontend precisar de um parser por endpoint."""
    recusas = [
        cliente.post(
            f"{PREFIXO}/videos/upload-url", json={"fileName": "c", "contentType": "text/plain"}
        ),
        enviar(cliente, b""),
        cliente.get(f"{PREFIXO}/videos/9999"),
        cliente.put(f"{PREFIXO}/videos/upload/inexistente", content=b"x"),
    ]

    for resposta in recusas:
        assert resposta.status_code >= 400, f"isto nao e recusa: {resposta.status_code}"
        assert set(resposta.json()) == {"detail"}
        assert isinstance(resposta.json()["detail"], str)
