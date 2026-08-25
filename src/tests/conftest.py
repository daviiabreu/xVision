"""Nenhum teste escreve no `var/` de quem roda a suite, e isso nao e confiado a disciplina.

`edge.paths.var_dir()` e relativo ao diretorio de trabalho, entao qualquer teste que toque cache,
saida, execucoes ou o banco da API escreve no `var/` de verdade a menos que isole. Quatro
arquivos nao isolavam, e o efeito nao foi sujeira e sim instabilidade: a suite acumulava estado
entre execucoes e passava a falhar em funcao do que ficara para tras, um teste diferente a cada
rodada, todos verdes isolados. Numa das vezes ela pendurou.

Isolar arquivo por arquivo conserta os quatro de hoje e nao o quinto de amanha. Aqui e uma vez
so, para toda a suite, e o quinto nasce isolado sem ninguem lembrar disso.

O segredo de assinatura tambem entra aqui: sem ele o servico recusa a subir, de proposito, e
cada teste que instancia o app teria de lembrar de o declarar.

`XVISION_DATA_DIR` aponta para lugar nenhum de proposito. O dump da Footstats existe na
maquina de quem trabalha e nao no runner, entao teste que dependa dele passa aqui e quebra
la: aconteceu, e a mensagem de erro falava do dump e nao do teste. Escondendo o dump por
padrao, a suite fica hermetica e quem precisar dele de verdade aponta a variavel e pula
explicitamente quando ele faltar.

A fila da API tambem e drenada entre testes: o worker roda noutra thread e uma tarefa que
sobreviva ao teste que a criou escreveria no `var/` do teste seguinte, ou no real.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest


def _abrir_a_fila() -> None:
    """Teste que sobe o app fecha a fila ao sair, e o proximo a encontraria fechada.

    O ciclo de vida chama `desligar` no encerramento, de proposito. Como o estado e do modulo e
    nao do app, ele atravessa a fronteira do teste: sem reabrir aqui, a ordem de coleta passa a
    decidir quais testes conseguem enfileirar, que e a mesma classe de defeito que o isolamento
    de `var/` fechou.
    """
    try:
        from api import execution
    except ImportError:  # pragma: no cover - a suite roda sem o extra `api` instalado
        return
    execution.religar()


def _drenar_a_fila() -> None:
    try:
        from api import execution
    except ImportError:  # pragma: no cover - a suite roda sem o extra `api` instalado
        return
    execution.aguardar(timeout=30.0)


@pytest.fixture(autouse=True)
def var_isolado(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("XVISION_VAR_DIR", str(tmp_path / "var"))
    monkeypatch.setenv("XVISION_AUTH_SECRET", "segredo-da-suite-nao-usar-fora-daqui")
    monkeypatch.setenv("XVISION_DATA_DIR", str(tmp_path / "dump-que-nao-existe"))
    _abrir_a_fila()
    yield
    _drenar_a_fila()
