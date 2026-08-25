"""A ferramenta de emitir e revogar token, que e como o servico ganha o primeiro cliente."""

from __future__ import annotations

import pytest

from api import auth, store, tokens


def test_a_new_token_is_shown_once_and_stored_only_as_a_digest(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Perdido, o token nao e recuperavel, e e isso que o banco vazado nao entrega."""
    assert tokens.main(["criar", "borda"]) == 0

    mostrado = capsys.readouterr().out.strip()
    linha = store.principal_por_nome("borda")
    assert linha is not None
    assert store.principal_por_token(auth.hash_de_token(mostrado)) is not None
    with store.conexao() as conn:
        guardado = conn.execute(
            "SELECT token_hash FROM principais WHERE id = ?", (linha.id,)
        ).fetchone()
    assert guardado["token_hash"] != mostrado


def test_the_same_name_cannot_be_issued_twice(capsys: pytest.CaptureFixture[str]) -> None:
    tokens.main(["criar", "repetido"])
    capsys.readouterr()

    assert tokens.main(["criar", "repetido"]) == 1
    assert "ja esta em uso" in capsys.readouterr().out


def test_revoking_keeps_the_record(capsys: pytest.CaptureFixture[str]) -> None:
    """Apagar a linha deixaria os videos daquele servico sem dono, ilegiveis por qualquer um."""
    tokens.main(["criar", "temporario"])
    capsys.readouterr()

    assert tokens.main(["revogar", "temporario"]) == 0
    linha = store.principal_por_nome("temporario")
    assert linha is not None
    assert linha.revogado_em is not None


def test_revoking_twice_says_there_is_nothing_active(capsys: pytest.CaptureFixture[str]) -> None:
    tokens.main(["criar", "um"])
    tokens.main(["revogar", "um"])
    capsys.readouterr()

    assert tokens.main(["revogar", "um"]) == 1


def test_the_listing_shows_what_is_active_and_what_is_not(
    capsys: pytest.CaptureFixture[str],
) -> None:
    tokens.main(["criar", "vivo"])
    tokens.main(["criar", "morto"])
    tokens.main(["revogar", "morto"])
    capsys.readouterr()

    tokens.main(["listar"])

    saida = capsys.readouterr().out
    assert "ativo\tvivo" in saida
    assert "revogado\tmorto" in saida


def test_the_listing_of_an_empty_database_says_nothing_and_does_not_break(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert tokens.main(["listar"]) == 0
    assert capsys.readouterr().out == ""


def test_the_service_token_tool_never_touches_a_user_account(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`revogar davi@sou.inteli.edu.br` desativava a conta de uma pessoa e devolvia sucesso.

    O nome e unico entre os dois tipos, entao bastava colisao ou engano de digitacao para uma
    ferramenta de token de servico trancar alguem fora da propria conta. O que a ferramenta nao
    gerencia, ela nao apaga.
    """
    pessoa = store.criar_principal(
        "usuario", "davi@sou.inteli.edu.br", senha_hash=auth.hash_de_senha("uma-senha-longa")
    )
    capsys.readouterr()

    assert tokens.revogar("davi@sou.inteli.edu.br") is False
    assert tokens.main(["revogar", "davi@sou.inteli.edu.br"]) == 1

    lido = store.ler_principal(pessoa.id)
    assert lido is not None
    assert lido.revogado_em is None, "a ferramenta de token trancou uma conta de usuario"


def test_the_listing_never_shows_a_user_account(capsys: pytest.CaptureFixture[str]) -> None:
    """Listar usuario numa ferramenta de token confunde quem opera sobre o que ela gerencia."""
    store.criar_principal("usuario", "pessoa@x.com", senha_hash=auth.hash_de_senha("senha-longa"))
    tokens.main(["criar", "um-servico"])
    capsys.readouterr()

    tokens.main(["listar"])

    saida = capsys.readouterr().out
    assert "um-servico" in saida
    assert "pessoa@x.com" not in saida
