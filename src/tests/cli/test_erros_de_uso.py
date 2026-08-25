"""Erro de configuracao chega como uma linha, e nao como sessenta de traceback.

Cada uma destas excecoes carrega uma mensagem escrita para quem digitou errado, dizendo o que
esta errado e quais sao os valores aceitos. Com a lista curta de antes elas escapavam do `except`
e o Typer imprimia o traceback inteiro: a mensagem boa continuava la, enterrada, e quem estava
usando via um erro de programa em vez de um erro de uso.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from cli import app

runner = CliRunner()


@pytest.fixture
def clipe(tmp_path: Path) -> Path:
    from edge.testing.make_clip import make_synthetic_clip

    return make_synthetic_clip(tmp_path / "s.mp4")


CONFIGS = {
    "implementacao inexistente": (
        '[frames.coarse]\nstrategy = "uniform"\nfps = 5\n'
        '[modules.players]\nimpl = "nao.existe:Classe"\nframes = "coarse"\n'
    ),
    "especie desconhecida": (
        '[frames.coarse]\nstrategy = "uniform"\nfps = 5\n'
        '[modules.jogadores]\nimpl = "edge.modules.stubs:StubPlayers"\nframes = "coarse"\n'
    ),
    "conjunto de quadros nao declarado": (
        '[frames.coarse]\nstrategy = "uniform"\nfps = 5\n'
        '[modules.players]\nimpl = "edge.modules.stubs:StubPlayers"\nframes = "grosso"\n'
    ),
    "chave escrita errado": (
        '[frames.coarse]\nstrategy = "uniform"\nfpss = 5\n'
        '[modules.players]\nimpl = "edge.modules.stubs:StubPlayers"\nframes = "coarse"\n'
    ),
}


@pytest.mark.parametrize("caso", sorted(CONFIGS))
def test_a_broken_config_reports_one_line_and_not_a_traceback(
    caso: str, clipe: Path, tmp_path: Path
) -> None:
    config = tmp_path / "quebrado.toml"
    config.write_text(CONFIGS[caso], encoding="utf-8")

    resultado = runner.invoke(app, ["run", "--clip", str(clipe), "--config", str(config)])

    assert resultado.exit_code != 0, f"{caso} passou sem erro"
    assert "Traceback" not in resultado.output, f"{caso} vazou traceback: {resultado.output[:200]}"
    assert resultado.exception is None or isinstance(resultado.exception, SystemExit)


def test_a_missing_config_says_where_to_put_it(clipe: Path, tmp_path: Path) -> None:
    resultado = runner.invoke(
        app, ["run", "--clip", str(clipe), "--config", str(tmp_path / "nao-existe.toml")]
    )

    assert resultado.exit_code != 0
    assert "Traceback" not in resultado.output
