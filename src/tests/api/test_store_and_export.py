"""O estado dos videos e o gerador do contrato versionado, cobertos onde a rota nao chega."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from api import execution, export, store
from api.schemas import tipo_de


@pytest.fixture(autouse=True)
def isolamento(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XVISION_VAR_DIR", str(tmp_path / "var"))


def test_a_state_outside_the_contract_is_refused_by_the_store() -> None:
    """O enum do frontend tem quatro valores, e gravar um quinto quebraria o consumidor."""
    video = store.criar("c.mp4", "video/mp4")

    with pytest.raises(store.StoreError, match="nao e um dos estados"):
        store.marcar(video.id, "Quase pronto")


def test_reading_a_video_that_does_not_exist_answers_nothing() -> None:
    assert store.ler(4242) is None
    assert store.ler_por_token("token-que-ninguem-emitiu") is None


def test_the_type_of_a_boolean_is_not_the_type_of_a_number() -> None:
    """O contrato do evento aceita tres tipos e o frontend so distingue dois."""
    assert tipo_de(True) == "boolean"
    assert tipo_de(1.5) == "float"


def test_an_unexpected_failure_still_leaves_a_reason_the_frontend_can_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Excecao que sobe de um worker morre sem leitor e deixaria o video parado para sempre."""
    video = store.criar("c.mp4", "video/mp4")
    arquivo = tmp_path / "quebrado.mp4"
    arquivo.write_bytes(b"nao importa")

    def explodir(*_: object, **__: object) -> None:
        raise RuntimeError("o disco pegou fogo")

    monkeypatch.setattr("api.execution.probe", explodir)
    execution.processar_video(video.id, arquivo, Path("config/pipeline.toml"))

    lido = store.ler(video.id)
    assert lido is not None
    assert lido.estado == store.FALHOU
    assert lido.motivo == execution.FALHA_INTERNA
    assert "disco pegou fogo" not in (lido.motivo or ""), "detalhe interno nao vaza para fora"
    assert not arquivo.exists(), "o arquivo enviado sai do disco mesmo quando o pipeline quebra"


def test_a_shut_down_queue_refuses_new_work_instead_of_starting_a_new_pool() -> None:
    """Recriar o pool sob demanda anulava o desligamento.

    O teste anterior so afirmava que o pool novo era outro objeto, e nunca submetia tarefa: ele
    passava tanto com o desligamento funcionando quanto com ele anulado. Depois de `desligar`, uma
    tarefa que chegasse levantava uma thread que ninguem mais desliga.
    """
    execution.submeter(lambda: None)
    execution.aguardar()
    execution.desligar()

    with pytest.raises(execution.FilaCheia, match="desligando"):
        execution.submeter(lambda: None)

    execution.religar()
    tarefa = execution.submeter(lambda: None)
    execution.aguardar()

    assert tarefa.done()


def test_the_worker_count_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Uma fila de um e o padrao porque memoria e o recurso escasso na borda."""
    monkeypatch.delenv(execution.WORKERS_ENV, raising=False)
    assert execution.workers() == 1

    monkeypatch.setenv(execution.WORKERS_ENV, "4")
    assert execution.workers() == 4

    monkeypatch.setenv(execution.WORKERS_ENV, "0")
    assert execution.workers() == 1, "zero worker nunca processaria nada"


def test_the_export_writes_the_contract_the_code_generates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Quem gera o arquivo versionado precisa de teste, senao so o arquivo velho e conferido."""
    destino = tmp_path / "openapi.json"
    monkeypatch.setattr(export, "DESTINO", destino)

    export.main()

    assert json.loads(destino.read_text(encoding="utf-8"))["info"]["title"] == "xVision"
    assert str(destino) in capsys.readouterr().out


def test_a_fresh_database_lands_on_the_latest_version() -> None:
    store.criar("c.mp4", "video/mp4")

    with store.conexao() as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == len(store.MIGRACOES)


def test_a_migration_added_later_reaches_a_database_that_already_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """E o caso que `CREATE TABLE IF NOT EXISTS` sozinho nao resolve, e o motivo disto existir.

    Sem versao, o banco de quem ja rodou o servico fica no esquema antigo para sempre e a coluna
    nova so falta em producao, nunca no teste de quem escreveu a migracao.
    """
    video = store.criar("c.mp4", "video/mp4")
    monkeypatch.setattr(
        store, "MIGRACOES", (*store.MIGRACOES, "ALTER TABLE videos ADD COLUMN origem TEXT;")
    )

    with store.conexao() as conn:
        colunas = {linha[1] for linha in conn.execute("PRAGMA table_info(videos)")}
        assert "origem" in colunas
        assert conn.execute("PRAGMA user_version").fetchone()[0] == len(store.MIGRACOES)

    assert store.ler(video.id) is not None, "a migracao nao pode perder a linha que ja existia"


def test_running_twice_does_not_apply_the_same_migration_again() -> None:
    """`ALTER TABLE` repetido levanta, entao reaplicar quebraria a segunda conexao do processo."""
    store.criar("c.mp4", "video/mp4")

    for _ in range(3):
        with store.conexao() as conn:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == len(store.MIGRACOES)


def test_the_queue_ceiling_holds_under_concurrent_submissions() -> None:
    """Conferir e enfileirar precisam acontecer juntos, ou o teto nao e teto.

    Medido sem o cadeado: com teto de 4 e 16 clientes numa barreira, os 16 foram aceitos e nenhum
    recusado, porque o handler base64 e sincrono e roda no pool de threads do Starlette. A
    docstring vende esse teto como o que impede ate 23 GB de disco em transito.
    """
    import threading

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv(execution.FILA_ENV, "4")
    largada = threading.Barrier(16)
    aceitos: list[bool] = []
    trava = threading.Lock()
    segurar = threading.Event()

    def esperar_a_largada() -> None:
        segurar.wait(5.0)

    def tentar() -> None:
        largada.wait()
        try:
            execution.submeter(esperar_a_largada)
            resultado = True
        except execution.FilaCheia:
            resultado = False
        with trava:
            aceitos.append(resultado)

    try:
        threads = [threading.Thread(target=tentar) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=20)
        assert sum(aceitos) <= 4, f"o teto de 4 aceitou {sum(aceitos)} trabalhos"
        assert sum(aceitos) >= 1, "o teto recusou tudo, o que tambem seria defeito"
    finally:
        segurar.set()
        execution.aguardar()
        monkeypatch.undo()


def test_the_ceiling_check_and_the_enqueue_happen_inside_the_same_lock() -> None:
    """Estrutural, porque prova de corrida e instavel por construcao.

    Conferir o tamanho e enfileirar em passos separados deixa uma janela: medido sem o cadeado,
    com teto de 4 e 16 clientes numa barreira, os 16 foram aceitos. O teste de concorrencia acima
    pega isso as vezes; este pega sempre. `store.reivindicar`, vinte linhas ao lado, resolve a
    mesma classe de corrida com UPDATE condicional e escreve o porque.
    """
    fonte = Path(execution.__file__).read_text(encoding="utf-8")
    arvore = ast.parse(fonte)
    funcao = next(
        no for no in ast.walk(arvore) if isinstance(no, ast.FunctionDef) and no.name == "submeter"
    )
    com_cadeado = [no for no in funcao.body if isinstance(no, ast.With)]

    assert com_cadeado, "submeter enfileira sem cadeado, e o teto deixa de ser teto"
    dentro = ast.dump(com_cadeado[0])
    assert "teto_da_fila" in dentro, "a conferencia do teto ficou fora do cadeado"
    assert "submit" in dentro, "o enfileiramento ficou fora do cadeado"
