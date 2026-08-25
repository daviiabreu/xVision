"""Emitir e revogar token de servico, pela linha de comando.

Mora na area `api` e nao no `cli` porque a fronteira entre as areas proibe o `cli` de importar a
`api`, e essa proibicao existe por um motivo: quem instala a borda para a Jetson nao carrega
servidor web junto. Uma ferramenta operacional da API pertence a API.

O token e mostrado uma vez. O banco guarda so o resumo dele, entao nao ha de onde recupera-lo:
perdido, emite-se outro e revoga-se o antigo.
"""

from __future__ import annotations

import argparse

from api import auth, store


def criar(nome: str) -> str:
    token = auth.novo_token()
    store.criar_principal("servico", nome, token_hash=auth.hash_de_token(token))
    return token


def revogar(nome: str) -> bool:
    """So mexe em servico. Conta de usuario nao se tranca por ferramenta de token.

    Sem o filtro, `revogar davi@sou.inteli.edu.br` desativava a conta de uma pessoa e devolvia
    sucesso: o nome e unico entre os dois tipos, entao bastava colisao ou engano de digitacao. A
    ferramenta e de token de servico, e o que ela nao gerencia ela nao apaga.
    """
    linha = store.principal_por_nome(nome)
    if linha is None or linha.tipo != "servico":
        return False
    return store.revogar_principal(linha.id)


def listar() -> tuple[store.PrincipalRow, ...]:
    return tuple(
        linha
        for linha in (store.ler_principal(i) for i in range(1, _ultimo() + 1))
        if linha is not None and linha.tipo == "servico"
    )


def _ultimo() -> int:
    with store.conexao() as conn:
        linha = conn.execute("SELECT MAX(id) AS ultimo FROM principais").fetchone()
    return 0 if linha is None or linha["ultimo"] is None else int(linha["ultimo"])


def main(argumentos: list[str] | None = None) -> int:
    analisador = argparse.ArgumentParser(prog="python -m api.tokens")
    sub = analisador.add_subparsers(dest="comando", required=True)
    criacao = sub.add_parser("criar", help="Emite um token novo e o mostra uma vez.")
    criacao.add_argument("nome")
    revogacao = sub.add_parser("revogar", help="Fecha a porta de um token, sem apagar o registro.")
    revogacao.add_argument("nome")
    sub.add_parser("listar", help="Mostra os tokens emitidos e quais estao revogados.")

    argumentos_lidos = analisador.parse_args(argumentos)
    if argumentos_lidos.comando == "criar":
        try:
            print(criar(argumentos_lidos.nome))
        except store.NomeEmUso as erro:
            print(f"erro: {erro}")
            return 1
        return 0
    if argumentos_lidos.comando == "revogar":
        if revogar(argumentos_lidos.nome):
            print(f"{argumentos_lidos.nome} revogado")
            return 0
        print(f"erro: nao ha token ativo chamado {argumentos_lidos.nome!r}")
        return 1
    for linha in listar():
        situacao = "revogado" if linha.revogado_em else "ativo"
        print(f"{linha.id}\t{situacao}\t{linha.nome}")
    return 0


if __name__ == "__main__":  # pragma: no cover - so o `python -m` chega aqui
    raise SystemExit(main())
