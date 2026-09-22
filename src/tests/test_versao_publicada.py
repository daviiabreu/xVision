"""A versao declarada do pacote precisa ser a ultima que foi publicada.

O defeito nao levanta nada e sobreviveu a tres releases: `releases/` ganhou notas da 0.2.0 e da
0.3.0, as tags foram criadas, o README passou a listar o historico, e `pyproject.toml` continuou
dizendo 0.1.0. Quem instalar a roda recebe `xvision 0.1.0` de uma arvore que e a 0.3.0, e qualquer
relatorio que cite a versao do pacote descreve uma entrega que nao existe.

O professor apontou isso na Sprint 3 junto de um pedido mais geral, mecanismo de fonte unica de
verdade. A fonte unica aqui e `releases/`: quem publica escreve as notas, e este teste cobra que a
versao do pacote acompanhe. Corrigir o numero sem a guarda so adiaria a proxima divergencia para a
release seguinte.

A versao mora em tres lugares, e os tres entram na guarda: `releases/`, `pyproject.toml` e
`uv.lock`. O lock precisa estar junto porque o CI roda `uv sync --frozen`, entao subir o
`pyproject.toml` sem relockar nao produz divergencia silenciosa, produz falha de instalacao.

Nao le tag do git de proposito. O teste roda em clone raso no CI, onde `git tag` vem vazio, e uma
guarda que passa por ausencia de dado e pior do que nenhuma.
"""

from __future__ import annotations

import re
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
RELEASES = REPOSITORY / "releases"
PYPROJECT = REPOSITORY / "src" / "pyproject.toml"
LOCK = REPOSITORY / "src" / "uv.lock"

VERSAO_DECLARADA = re.compile(r'^version\s*=\s*"(\d+)\.(\d+)\.(\d+)"', re.MULTILINE)
VERSAO_NO_LOCK = re.compile(r'^name = "xvision"\nversion = "(\d+)\.(\d+)\.(\d+)"', re.MULTILINE)
NOME_DE_VERSAO = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def _declarada() -> tuple[int, int, int]:
    assert (REPOSITORY / ".gitlab-ci.yml").is_file(), f"{REPOSITORY} nao e a raiz do repositorio"
    achado = VERSAO_DECLARADA.search(PYPROJECT.read_text(encoding="utf-8"))
    assert achado is not None, f"{PYPROJECT} nao declara `version` em x.y.z"
    return int(achado.group(1)), int(achado.group(2)), int(achado.group(3))


def _no_lock() -> tuple[int, int, int]:
    achado = VERSAO_NO_LOCK.search(LOCK.read_text(encoding="utf-8"))
    assert achado is not None, f"{LOCK} nao declara a versao de `xvision`"
    return int(achado.group(1)), int(achado.group(2)), int(achado.group(3))


def _publicadas() -> list[tuple[int, int, int]]:
    """As notas de release que tem nome de versao, ignorando rascunho como `draft-patch`."""
    achadas: list[tuple[int, int, int]] = []
    for arquivo in RELEASES.iterdir():
        nome = NOME_DE_VERSAO.match(arquivo.name)
        if arquivo.is_file() and nome is not None:
            achadas.append((int(nome.group(1)), int(nome.group(2)), int(nome.group(3))))
    return sorted(achadas)


def test_a_versao_do_pacote_e_a_ultima_publicada() -> None:
    publicadas = _publicadas()
    assert publicadas, f"{RELEASES} nao tem nenhuma nota com nome de versao"

    ultima = publicadas[-1]
    declarada = _declarada()

    assert declarada == ultima, (
        f"{PYPROJECT.relative_to(REPOSITORY)} declara "
        f"{'.'.join(map(str, declarada))} e a ultima release publicada e "
        f"{'.'.join(map(str, ultima))}. Quem publica release atualiza os dois."
    )


def test_toda_nota_de_release_tem_o_titulo_da_propria_versao() -> None:
    """Nome de arquivo e titulo precisam concordar, senao a fonte unica tem duas vozes."""
    for versao in _publicadas():
        nome = ".".join(map(str, versao))
        primeira = (RELEASES / nome).read_text(encoding="utf-8").splitlines()[0].strip()
        assert primeira == f"# Release {nome}", (
            f"releases/{nome} comeca com {primeira!r}, esperado '# Release {nome}'"
        )


def test_o_lock_carrega_a_mesma_versao_do_pacote() -> None:
    """O CI roda `uv sync --frozen`: lock atrasado nao e divergencia, e falha de instalacao."""
    declarada, lock = _declarada(), _no_lock()

    assert lock == declarada, (
        f"pyproject declara {'.'.join(map(str, declarada))} e o lock tem "
        f"{'.'.join(map(str, lock))}. Rode `uv lock` depois de subir a versao."
    )
