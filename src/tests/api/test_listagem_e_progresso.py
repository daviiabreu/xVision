"""Lista persistida, fila e progresso, que substituem o indice local do frontend."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api import PREFIXO, create_app, execution, render, store
from contracts import Perception
from edge.testing.make_clip import make_synthetic_clip

from .apoio import cliente_autenticado


@pytest.fixture(autouse=True)
def isolamento(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("XVISION_VAR_DIR", str(tmp_path / "var"))
    execution.religar()
    yield
    execution.aguardar()


def _reservar(cliente: TestClient, nome: str) -> int:
    resposta = cliente.post(
        f"{PREFIXO}/videos/upload-url",
        json={"fileName": nome, "contentType": "video/mp4"},
    )
    assert resposta.status_code == 200, resposta.text
    return int(resposta.json()["id"])


def test_a_lista_e_persistida_por_conta_e_paginada_do_mais_recente() -> None:
    primeiro = cliente_autenticado(create_app(), "primeiro")
    segundo = cliente_autenticado(create_app(), "segundo")
    antigo = _reservar(primeiro, "antigo.mp4")
    novo = _reservar(primeiro, "novo.mp4")
    _reservar(segundo, "de-outra-conta.mp4")

    pagina = primeiro.get(f"{PREFIXO}/videos", params={"limite": 1, "deslocamento": 0})
    proxima = primeiro.get(f"{PREFIXO}/videos", params={"limite": 1, "deslocamento": 1})

    assert pagina.status_code == 200
    assert pagina.json()["total"] == 2
    assert pagina.json()["pendentes"] == 2
    assert pagina.json()["em_processamento"] == 0
    assert pagina.json()["finalizadas"] == 0
    assert pagina.json()["falhas"] == 0
    assert [item["id"] for item in pagina.json()["videos"]] == [novo]
    assert [item["id"] for item in proxima.json()["videos"]] == [antigo]
    assert pagina.json()["videos"][0]["nome_arquivo"] == "novo.mp4"


def test_uma_reserva_sem_upload_expira_em_vez_de_ficar_pendente_para_sempre(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XVISION_UPLOAD_RESERVATION_TTL_S", "1")

    with cliente_autenticado(create_app()) as cliente:
        reserva = cliente.post(
            f"{PREFIXO}/videos/upload-url",
            json={"fileName": "abandonado.mp4", "contentType": "video/mp4"},
        )
        time.sleep(1.1)
        status = cliente.get(f"{PREFIXO}/videos/status/{reserva.json()['id']}")
        envio_atrasado = cliente.put(reserva.json()["uploadUrl"], content=b"atrasado")

    assert status.status_code == 200
    assert status.json()["video_status"] == store.FALHOU
    assert "expirou" in status.json()["motivo"]
    assert status.json()["reconsultar_em_s"] is None
    assert envio_atrasado.status_code == 409


def test_o_progresso_publicado_vem_do_estado_persistido() -> None:
    cliente = cliente_autenticado(create_app())
    identificador = _reservar(cliente, "clipe.mp4")
    store.marcar(identificador, store.PROCESSANDO)
    store.marcar_progresso(identificador, "pose", 4, 9)

    corpo = cliente.get(f"{PREFIXO}/videos/status/{identificador}").json()

    assert corpo["etapa_atual"] == "pose"
    assert corpo["etapas_concluidas"] == 4
    assert corpo["etapas_total"] == 9
    assert corpo["progresso"] == 44
    assert corpo["reconsultar_em_s"] == 2


def test_render_and_evidence_follow_model_progress_from_the_same_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    updates: list[tuple[str, int, int]] = []
    persist_progress = store.marcar_progresso

    def observe_progress(
        identificador: int, etapa: str, concluidas: int, total: int, caminho: Path | None = None
    ) -> None:
        persist_progress(identificador, etapa, concluidas, total, caminho)
        updates.append((etapa, concluidas, total))

    rendered: list[Perception] = []

    def observe_render(
        video_id: int, clipe: Path, config: render.RenderConfig, *, perception: Perception
    ) -> Path:
        state = store.ler(video_id)
        assert state is not None and state.etapa_atual == "render"
        assert clipe.is_file() and perception.clip_id == f"video-{video_id}"
        rendered.append(perception)
        return render.render_path(video_id)

    monkeypatch.setattr(store, "marcar_progresso", observe_progress)
    monkeypatch.setattr(render, "render", observe_render)
    video = store.criar("same-run.mp4", "video/mp4")
    clip = make_synthetic_clip(tmp_path / "same-run.mp4")

    execution.processar_video(
        video.id, clip, Path(__file__).resolve().parents[2] / "config" / "pipeline.toml"
    )

    assert len(rendered) == 1
    assert [stage for stage, _, _ in updates][-2:] == ["render", "evidence"]
    assert [completed for _, completed, _ in updates] == list(range(len(updates)))
    state = store.ler(video.id)
    assert state is not None and state.estado == store.FINALIZADO
    assert state.etapas_concluidas == state.etapas_total == len(updates)


def test_a_posicao_da_fila_reflete_a_ordem_real_do_executor() -> None:
    cliente = cliente_autenticado(create_app())
    executando = _reservar(cliente, "executando.mp4")
    primeiro = _reservar(cliente, "primeiro.mp4")
    segundo = _reservar(cliente, "segundo.mp4")
    liberar = threading.Event()
    iniciou = threading.Event()

    def bloquear() -> None:
        iniciou.set()
        liberar.wait(5)

    execution.submeter(bloquear, chave=executando)
    assert iniciou.wait(2)
    execution.submeter(lambda: None, chave=primeiro)
    execution.submeter(lambda: None, chave=segundo)

    try:
        assert cliente.get(f"{PREFIXO}/videos/status/{primeiro}").json()["posicao_fila"] == 1
        assert cliente.get(f"{PREFIXO}/videos/status/{segundo}").json()["posicao_fila"] == 2
        assert cliente.get(f"{PREFIXO}/videos/status/{executando}").json()["posicao_fila"] is None
    finally:
        liberar.set()


@pytest.mark.parametrize(
    ("etapa", "concluidas", "total"),
    [("", 0, 1), ("pose", -1, 1), ("pose", 2, 1), ("pose", 0, 0)],
)
def test_progresso_invalido_nao_e_gravado(etapa: str, concluidas: int, total: int) -> None:
    video = store.criar("c.mp4", "video/mp4")
    store.marcar(video.id, store.PROCESSANDO)

    with pytest.raises(store.StoreError):
        store.marcar_progresso(video.id, etapa, concluidas, total)
