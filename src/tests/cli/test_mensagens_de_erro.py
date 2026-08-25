"""As mensagens que alguem le quando erra a mao, lidas por um teste pelo menos uma vez.

Sao 33 linhas de `typer.echo(..., err=True)` que nenhum teste tinha exercitado. Nao e sobre
cobertura: e que texto que ninguem leu tende a estar errado, a apontar para uma opcao que mudou
de nome, ou a nao dizer o que fazer em seguida. Cada afirmacao abaixo cobra as tres coisas: que
sai pelo canal de erro, que o codigo de saida nao e zero, e que a mensagem diz o proximo passo.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from cli import app

runner = CliRunner()

_ARTEFATO = (
    "--kind",
    "players",
    "--name",
    "m",
    "--version",
    "1",
    "--entrypoint",
    "peso.bin",
)


def _erro(*argumentos: str) -> str:
    resultado = runner.invoke(app, list(argumentos))
    assert resultado.exit_code != 0, f"{argumentos} devia falhar e saiu 0: {resultado.output}"
    assert "Traceback" not in resultado.output
    return resultado.output


def test_run_without_a_clip_says_which_option_to_pass() -> None:
    saida = _erro("run")

    assert "--clip" in saida
    assert "--all" in saida


def test_run_with_a_config_that_does_not_exist_names_the_path_and_the_option(
    tmp_path: Path,
) -> None:
    ausente = tmp_path / "nao-existe.toml"

    saida = _erro("run", "--all", "--config", str(ausente))

    assert str(ausente) in saida
    assert "--config" in saida


@pytest.mark.parametrize(
    ("comando", "opcao"),
    [("data", "--labels"), ("data", "--split")],
)
def test_data_check_names_the_missing_file_and_its_option(
    comando: str, opcao: str, tmp_path: Path
) -> None:
    ausente = tmp_path / "sumiu.toml"

    saida = _erro(comando, "check", opcao, str(ausente))

    assert str(ausente) in saida
    assert opcao in saida


def test_eval_with_a_missing_truth_file_names_it(tmp_path: Path) -> None:
    execucao = tmp_path / "execucao"
    execucao.mkdir()
    (execucao / "events.jsonl").write_text("", encoding="utf-8")
    ausente = tmp_path / "gabarito.json"

    saida = _erro("eval", str(execucao), "--truth", str(ausente))

    assert str(ausente) in saida


def test_eval_with_an_unreadable_events_file_says_what_broke(tmp_path: Path) -> None:
    """Arquivo de eventos corrompido e o caso real de execucao interrompida no meio.

    O `--truth` e passado de proposito: sem ele o `eval` monta o gabarito a partir do dump do
    parceiro, que existe na maquina de quem trabalha e nao no runner, e o teste passaria a medir
    a ausencia do dump em vez da mensagem do arquivo corrompido. Foi o que aconteceu.
    """
    execucao = tmp_path / "execucao"
    execucao.mkdir()
    (execucao / "events.jsonl").write_text("{isto nao e json\n", encoding="utf-8")
    gabarito = tmp_path / "gabarito.json"
    gabarito.write_text("{}", encoding="utf-8")

    saida = _erro("eval", str(execucao), "--truth", str(gabarito))

    assert "events.jsonl" in saida


def test_artifact_refuses_a_path_that_is_not_a_directory(tmp_path: Path) -> None:
    arquivo = tmp_path / "um-arquivo.bin"
    arquivo.write_bytes(b"conteudo")

    saida = _erro("artifact", "add", str(arquivo), *_ARTEFATO)

    assert str(arquivo) in saida
    assert "diretorio" in saida


def test_artifact_refuses_an_unknown_kind_and_lists_the_known_ones(tmp_path: Path) -> None:
    (tmp_path / "peso.bin").write_bytes(b"conteudo")

    saida = _erro(
        "artifact",
        "add",
        str(tmp_path),
        "--kind",
        "jogadores",
        "--name",
        "m",
        "--version",
        "1",
        "--entrypoint",
        "peso.bin",
    )

    assert "jogadores" in saida
    assert "players" in saida, "a mensagem nao lista as especies conhecidas"


def test_artifact_refuses_an_empty_directory(tmp_path: Path) -> None:
    vazio = tmp_path / "sem-nada"
    vazio.mkdir()

    saida = _erro("artifact", "add", str(vazio), *_ARTEFATO)

    assert "vazio" in saida
