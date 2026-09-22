"""Retomada de uma execucao interrompida do catalogo.

Cada teste aqui replanta um defeito que passa com codigo de saida zero: clipe processado duas vezes
no `events.jsonl`, `run.json` com tempo de etapa cobrindo so o ultimo trecho, ou retomada de uma
execucao com outra configuracao produzindo um arquivo unico com metade dos eventos de cada uma.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import cli
from cli import app
from mlops.checkpoint import CHECKPOINT_FILE, read_checkpoint

runner = CliRunner()
CONFIG = Path("config/pipeline.toml")


class _Interrompe(Exception): ...


@dataclass(frozen=True)
class _Registro:
    shot_id: str
    fold: int = 0
    match_id: str = "900"


@dataclass(frozen=True)
class _Evento:
    clip_id: str

    def to_dict(self) -> dict[str, str]:
        return {"clip_id": self.clip_id}


@dataclass(frozen=True)
class _Etapa:
    kind: str
    name: str
    version: str
    duration_ms: float
    status: str = "ok"
    reason: str | None = None


@dataclass(frozen=True)
class _Resultado:
    event: _Evento
    stages: tuple[_Etapa, ...]
    shot_source: str | None
    ball_support: str | None = "detected"
    shot_candidates: tuple[Mapping[str, object], ...] = ()
    shot_candidate_pool: tuple[int, ...] = ()


def _prepara(
    monkeypatch: pytest.MonkeyPatch,
    quantos: int,
    para_em: int | None = None,
    deslocamento: int = 0,
) -> list[str]:
    """Catalogo sintetico de `quantos` clipes; `para_em` levanta na enesima chamada.

    `deslocamento` troca os identificadores mantendo a contagem, que e o caso que a guarda de
    selecao existe para recusar.
    """
    registros = [_Registro(str(i + deslocamento)) for i in range(quantos)]
    vistos: list[str] = []

    def processa(registro: Any, _config: Path) -> _Resultado:
        if para_em is not None and len(vistos) == para_em:
            raise _Interrompe
        vistos.append(registro.shot_id)
        return _Resultado(
            event=_Evento(registro.shot_id),
            stages=(_Etapa("ball", "yolo-ball", "0.1.0", 10.0),),
            shot_source="ball-break",
            shot_candidates=({"frame_index": 10, "sample_index": 12, "score": 0.5},),
            shot_candidate_pool=(10, 40, 70),
        )

    def rotulos(_: Path) -> object:
        return object()

    def catalogo() -> list[_Registro]:
        return registros

    def populacao(itens: Any, _tabela: Any) -> list[Any]:
        return list(itens)

    monkeypatch.setattr(cli, "read_labels", rotulos)
    monkeypatch.setattr(cli, "load_catalog", catalogo)
    monkeypatch.setattr(cli, "_populacao", populacao)
    monkeypatch.setattr(cli, "_processa_registro", processa)
    return vistos


def _executa(saida: Path, *extra: str, config: Path = CONFIG) -> Any:
    """Chama pela casca, e nao pela funcao interna: e o caminho que a pessoa usa."""
    return runner.invoke(
        app,
        ["run", "--all", "--config", str(config), "--out", str(saida), *extra],
        catch_exceptions=False,
    )


def _run_id(saida: Path) -> Path:
    (unico,) = [p for p in saida.iterdir() if p.is_dir()]
    return unico


def _eventos(execucao: Path) -> list[str]:
    return [
        json.loads(linha)["clip_id"]
        for linha in (execucao / "events.jsonl").read_text().splitlines()
        if linha.strip()
    ]


def test_a_execucao_completa_apaga_o_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Checkpoint que sobra num diretorio completo sinaliza execucao incompleta que nao existe."""
    _prepara(monkeypatch, 3)
    saida = tmp_path / "runs"

    _executa(saida)

    execucao = _run_id(saida)
    assert _eventos(execucao) == ["0", "1", "2"]
    assert not (execucao / CHECKPOINT_FILE).exists()
    assert not (execucao / "events.jsonl.parcial").exists()


def test_a_retomada_completa_a_execucao_sem_repetir_clipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """O defeito replantado e o clipe repetido: sem `done`, a retomada recomeca do primeiro."""
    _prepara(monkeypatch, 25, para_em=12)
    saida = tmp_path / "runs"

    with pytest.raises(_Interrompe):
        _executa(saida)
    execucao = _run_id(saida)
    parcial = read_checkpoint(execucao)
    assert parcial.ok == 10, "o checkpoint responde pelos dez primeiros, nao pelos doze"

    _prepara(monkeypatch, 25)
    _executa(saida, "--resume", str(execucao))

    assert _eventos(execucao) == [str(i) for i in range(25)]
    assert len(_eventos(execucao)) == len(set(_eventos(execucao)))


def test_a_retomada_preserva_os_agregados_do_trecho_anterior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sem persistir tempo por etapa, o `run.json` descreveria so o ultimo trecho e passaria."""
    _prepara(monkeypatch, 25, para_em=12)
    saida = tmp_path / "runs"

    with pytest.raises(_Interrompe):
        _executa(saida)
    execucao = _run_id(saida)
    _prepara(monkeypatch, 25)
    _executa(saida, "--resume", str(execucao))

    registro = json.loads((execucao / "run.json").read_text())
    assert registro["dataset"]["clips"] == 25
    assert registro["dataset"]["resumed"] == 1
    assert registro["modules"]["ball"]["duration_ms"]["n"] == 25


def test_retomar_com_outra_configuracao_e_recusado(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Um `events.jsonl` com metade dos eventos de cada configuracao, e `run.json` declarando so
    uma delas, sai plausivel e descreve uma execucao que nunca existiu."""
    _prepara(monkeypatch, 25, para_em=12)
    saida = tmp_path / "runs"
    with pytest.raises(_Interrompe):
        _executa(saida)
    execucao = _run_id(saida)
    outra = tmp_path / "outra.toml"
    outra.write_text(CONFIG.read_text() + "\n# comentario que muda o hash\n", encoding="utf-8")

    _prepara(monkeypatch, 25)
    resultado = runner.invoke(
        app,
        ["run", "--all", "--config", str(outra), "--out", str(saida), "--resume", str(execucao)],
    )

    assert resultado.exit_code == 1
    assert "misturaria duas configuracoes" in resultado.output


def test_retomar_com_outra_populacao_e_recusado(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepara(monkeypatch, 25, para_em=12)
    saida = tmp_path / "runs"
    with pytest.raises(_Interrompe):
        _executa(saida)
    execucao = _run_id(saida)

    _prepara(monkeypatch, 30)
    resultado = runner.invoke(
        app,
        [
            "run",
            "--all",
            "--config",
            str(CONFIG),
            "--out",
            str(saida),
            "--resume",
            str(execucao),
        ],
    )

    assert resultado.exit_code == 1
    assert "mediria outra populacao" in resultado.output


def test_retomar_execucao_ja_completa_avisa_em_vez_de_reprocessar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepara(monkeypatch, 3)
    saida = tmp_path / "runs"
    _executa(saida)
    execucao = _run_id(saida)

    _prepara(monkeypatch, 3)
    resultado = runner.invoke(
        app,
        [
            "run",
            "--all",
            "--config",
            str(CONFIG),
            "--out",
            str(saida),
            "--resume",
            str(execucao),
        ],
    )

    assert resultado.exit_code == 0
    assert "ja esta completa" in resultado.output


def test_retomar_sem_checkpoint_e_erro_explicito(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepara(monkeypatch, 3)
    vazio = tmp_path / "sem-checkpoint"
    vazio.mkdir()

    resultado = runner.invoke(
        app, ["run", "--all", "--config", str(CONFIG), "--resume", str(vazio)]
    )

    assert resultado.exit_code == 1
    assert "nao tem checkpoint para retomar" in resultado.output


def test_resume_sem_all_avisa_em_vez_de_ignorar(tmp_path: Path) -> None:
    """Ignorar a opcao faria a pessoa acreditar que retomou quando processou um clipe do zero."""
    resultado = runner.invoke(
        app, ["run", "--clip", str(tmp_path / "c.mp4"), "--resume", str(tmp_path)]
    )

    assert resultado.exit_code == 1
    assert "--resume vale para --all" in resultado.output


def test_o_progresso_aparece_durante_a_execucao(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sem progresso, uma execucao de horas nao se distingue de uma travada."""
    _prepara(monkeypatch, 12)

    resultado = runner.invoke(
        app, ["run", "--all", "--config", str(CONFIG), "--out", str(tmp_path / "runs")]
    )

    assert resultado.exit_code == 0
    assert "10/12 clipes" in resultado.output
    assert "s por clipe" in resultado.output
    assert "faltam cerca de" in resultado.output


def test_a_procedencia_por_clipe_sai_ao_lado_dos_eventos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sem este arquivo, `eval-instant` mede junto contato detectado e ponto medio preenchido."""
    from mlops.instant import OBSERVATIONS_FILE

    _prepara(monkeypatch, 3)
    saida = tmp_path / "runs"

    _executa(saida)

    linhas = [
        json.loads(x)
        for x in (_run_id(saida) / OBSERVATIONS_FILE).read_text().splitlines()
        if x.strip()
    ]
    assert linhas == [
        {
            "clip_id": str(i),
            "shot_source": "ball-break",
            "ball_support": "detected",
            "candidates": [{"frame_index": 10, "sample_index": 12, "score": 0.5}],
            "candidate_pool": [10, 40, 70],
        }
        for i in range(3)
    ]


def test_a_procedencia_sobrevive_a_retomada(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mlops.instant import OBSERVATIONS_FILE

    _prepara(monkeypatch, 25, para_em=12)
    saida = tmp_path / "runs"
    with pytest.raises(_Interrompe):
        _executa(saida)
    execucao = _run_id(saida)

    _prepara(monkeypatch, 25)
    _executa(saida, "--resume", str(execucao))

    linhas = [
        json.loads(x) for x in (execucao / OBSERVATIONS_FILE).read_text().splitlines() if x.strip()
    ]
    assert len(linhas) == 25
    assert {x["clip_id"] for x in linhas} == {str(i) for i in range(25)}
    assert all(
        x["candidates"] == [{"frame_index": 10, "sample_index": 12, "score": 0.5}] for x in linhas
    ), (
        "candidato dos clipes anteriores a queda precisa sobreviver ao checkpoint, senao a "
        "taxonomia sai medida sobre metade do conjunto com cara de medida sobre ele inteiro"
    )


def test_retomar_com_outros_clipes_na_mesma_contagem_e_recusado(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A contagem sozinha nao identifica a selecao, e o revisor reproduziu o pior caso.

    Interromper um `--dobra ajuste --limit 40` e retomar com `--dobra reservado --limit 40`
    terminava com codigo zero, consumia o pacote reservado dentro de uma execucao que comecou como
    ajuste, e gravava um `run.json` declarando quarenta clipes com sessenta eventos.
    """
    _prepara(monkeypatch, 25, para_em=12)
    saida = tmp_path / "runs"
    with pytest.raises(_Interrompe):
        _executa(saida)
    execucao = _run_id(saida)

    # outros vinte e cinco clipes, mesma contagem
    _prepara(monkeypatch, 25, deslocamento=100)
    resultado = runner.invoke(
        app,
        ["run", "--all", "--config", str(CONFIG), "--out", str(saida), "--resume", str(execucao)],
    )

    assert resultado.exit_code == 1
    assert "selecionou outros clipes" in resultado.output


def test_resume_rejects_changed_feature_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from edge.features.type_of_play import TypeOfPlay

    _prepara(monkeypatch, 25, para_em=12)
    saida = tmp_path / "runs"
    with pytest.raises(_Interrompe):
        _executa(saida)
    execucao = _run_id(saida)
    parcial = (execucao / "events.jsonl.parcial").read_bytes()
    monkeypatch.setattr(TypeOfPlay, "spec", replace(TypeOfPlay.spec, version="9.9.9"))
    vistos = _prepara(monkeypatch, 25)

    resultado = _executa(saida, "--resume", str(execucao))

    assert resultado.exit_code == 1
    assert "duas versoes de feature" in resultado.output
    assert not vistos
    assert (execucao / "events.jsonl.parcial").read_bytes() == parcial


@pytest.mark.parametrize("campo", ["features_sha256", "selection_sha256"])
def test_resume_rejects_checkpoint_without_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, campo: str
) -> None:
    _prepara(monkeypatch, 25, para_em=12)
    saida = tmp_path / "runs"
    with pytest.raises(_Interrompe):
        _executa(saida)
    execucao = _run_id(saida)
    caminho = execucao / CHECKPOINT_FILE
    documento = json.loads(caminho.read_text())
    documento.pop(campo, None)
    caminho.write_text(json.dumps(documento))
    parcial = (execucao / "events.jsonl.parcial").read_bytes()
    vistos = _prepara(monkeypatch, 25)

    resultado = _executa(saida, "--resume", str(execucao))

    assert resultado.exit_code == 1
    assert "sem identidade" in resultado.output
    assert not vistos
    assert (execucao / "events.jsonl.parcial").read_bytes() == parcial


def test_retomar_com_outra_versao_de_feature_e_recusado(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """O `config_sha256` cobre os bytes do TOML, e a versao da feature mora no codigo.

    Sem esta guarda, subir `type_of_play` de `0.1.0` para `0.2.0` e retomar uma execucao comecada
    antes passava por todas as verificacoes, e o `events.jsonl` saia com eventos das duas versoes.
    O `run.json` declarava so a atual, porque ele le a versao do codigo em execucao, e
    `mlops.evaluate` seleciona feature por nome e nao por versao, entao ele media as duas juntas.
    """
    _prepara(monkeypatch, 25, para_em=12)
    saida = tmp_path / "runs"
    with pytest.raises(_Interrompe):
        _executa(saida)
    execucao = _run_id(saida)

    estado = read_checkpoint(execucao)
    assert estado.features_sha256, "o checkpoint precisa gravar a versao das features"

    def outras_versoes(_config: Path) -> str:
        return "0" * 64

    _prepara(monkeypatch, 25)
    monkeypatch.setattr(cli, "_feature_digest", outras_versoes)
    resultado = runner.invoke(
        app,
        ["run", "--all", "--config", str(CONFIG), "--out", str(saida), "--resume", str(execucao)],
    )

    assert resultado.exit_code == 1
    assert "duas versoes de feature" in resultado.output


def test_a_versao_das_features_sai_do_config_e_nao_de_um_literal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """O digest tem que mudar quando a versao de uma feature muda, senao a guarda nunca dispara."""
    from edge.features import type_of_play

    antes = cli._feature_digest(CONFIG)  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(
        type_of_play.TypeOfPlay,
        "spec",
        replace(type_of_play.TypeOfPlay.spec, version="9.9.9"),
    )
    depois = cli._feature_digest(CONFIG)  # pyright: ignore[reportPrivateUsage]

    assert antes != depois
