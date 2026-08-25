"""O `src/README.md` promete comandos, e comando renomeado quebraria a promessa em silencio.

E o README do backend, e nao o da raiz. O da raiz e do projeto inteiro e oito pessoas editam:
mexer nele a partir daqui produz conflito de merge por um motivo que nada tem a ver com o
backend, entao esta branch nao o toca.

O criterio de fechamento da issue e alguem do grupo que nao escreveu o backend processar um clipe
seguindo so o manual, entao o manual precisa de algo que falhe quando ele fica errado.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cli import app

README = Path(__file__).resolve().parents[1] / "README.md"
REPOSITORIO = Path(__file__).resolve().parents[2]
runner = CliRunner()


GRUPOS = frozenset({"data", "cache", "artifact"})


def _comandos_citados() -> set[str]:
    """O README mostra o comando com argumento, entao o argumento e aparado aqui."""
    texto = README.read_text(encoding="utf-8")
    encontrados: set[str] = set()
    for linha in re.finditer(r"(?:uv run )?xvision ((?:[a-z-]+ ?)+)", texto):
        palavras = linha.group(1).split()
        encontrados.add(" ".join(palavras[:2]) if palavras[0] in GRUPOS else palavras[0])
    return encontrados


def test_the_readme_cites_at_least_the_path_the_issue_demands() -> None:
    citados = _comandos_citados()

    assert {"clip", "run", "data check", "eval"} <= citados


@pytest.mark.parametrize("comando", sorted(_comandos_citados()))
def test_every_command_the_readme_cites_exists(comando: str) -> None:
    resultado = runner.invoke(app, [*comando.split(), "--help"])

    assert resultado.exit_code == 0, f"o README cita `xvision {comando}`, que nao existe"


def test_the_readme_says_where_the_output_goes() -> None:
    """Metade do pedido da issue e "onde o resultado aparece"."""
    texto = README.read_text(encoding="utf-8")

    assert "var/runs/" in texto


def test_the_delivery_table_declares_what_is_partial() -> None:
    """Entrega parcial declarada como parcial pontua, e parcialidade escondida derruba."""
    texto = README.read_text(encoding="utf-8")

    assert "**parcial**" in texto
    assert "adiado" in texto


# Cada isencao com o motivo escrito, porque filtro generico foi o que tornou a primeira versao
# deste teste vazia. O que nao esta aqui tem de existir no disco.
NAO_SAO_CAMINHOS = {
    "feat/frontend": "e um ramo do git, e nao um caminho",
    "src/interface": "e onde o frontend entra quando a branch dele for mesclada",
    "interface": "idem, citado sem o prefixo na tabela de areas",
    "var": "o sistema cria ao rodar, e apagar e seguro por contrato",
    "var/cache": "idem",
    "var/out": "idem",
    "var/runs": "idem",
    "data/footstats": "material restrito do parceiro, nunca versionado",
}


def _caminhos_citados() -> set[str]:
    texto = README.read_text(encoding="utf-8")
    candidatos = {
        trecho
        for trecho in re.findall(r"`([^`\n]+)`", texto)
        if "/" in trecho and " " not in trecho and not trecho.startswith(("http", "<", "-"))
    }
    return {trecho.rstrip("/") for trecho in candidatos if not trecho.endswith(("*", "}"))}


def _existe(trecho: str) -> bool:
    """O README do backend cita caminho relativo a `src/` e tambem relativo a raiz."""
    return (REPOSITORIO / trecho).exists() or (README.parent / trecho).exists()


def test_the_path_check_is_not_looking_at_an_empty_set() -> None:
    """A primeira versao deste teste isentava todo caminho sem ponto no nome.

    O conjunto que ela verificava era vazio, incluindo `infra/railway/`, que e o caso citado na
    propria justificativa dela. Teste que nao olha nada passa sempre, entao o tamanho do conjunto
    vira asserção.
    """
    verificados = _caminhos_citados() - set(NAO_SAO_CAMINHOS)

    assert len(verificados) >= 10, f"so restaram {len(verificados)} caminhos verificados"


def test_every_repository_path_the_readme_cites_really_exists() -> None:
    """A revisao achou tres promessas falsas de uma vez: `infra/railway/`, o provider do Railway
    e um simbolo `EXTRACTORS` que nao existe em lugar nenhum. Caminho citado e verificavel, entao
    passa a ser verificado.
    """
    faltando = sorted(
        trecho
        for trecho in _caminhos_citados()
        if trecho not in NAO_SAO_CAMINHOS and not _existe(trecho)
    )

    assert not faltando, f"o README cita caminho que nao existe: {faltando}"


def test_every_module_the_readme_tells_you_to_run_is_importable() -> None:
    """O README manda rodar tres provas ao vivo, e `python -m` num modulo que saiu so falha
    na mao de quem seguiu a instrucao.
    """
    texto = README.read_text(encoding="utf-8")
    modulos = set(re.findall(r"python -m ([a-z_.]+)", texto))

    assert modulos, "o README parou de citar as provas ao vivo"
    for modulo in sorted(modulos):
        assert importlib.util.find_spec(modulo) is not None, f"{modulo} nao existe"


def test_the_readme_does_not_end_mid_sentence() -> None:
    """O arquivo terminava em `com dez valores distintos em`, sem ponto e sem quebra de linha.

    Ele e o `readme` declarado no `pyproject.toml`, entao o texto cortado vai para o metadado do
    pacote. Os testes que existiam conferiam que os comandos citados existem, e nunca que o texto
    esta inteiro: um `>` no lugar de `>>` numa escrita corta o fim sem quebrar nada mais.
    """
    texto = README.read_text(encoding="utf-8")

    assert texto.endswith("\n"), "o arquivo nao termina com quebra de linha"
    ultima = texto.rstrip().splitlines()[-1].rstrip()
    assert ultima.endswith((".", ":", "`", ")", "|")), (
        f"o README termina em {ultima[-40:]!r}, que nao parece fim de frase"
    )
