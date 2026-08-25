"""O que o sistema produz nunca pode virar commit, principalmente dado derivado do parceiro.

O bloco de conteudo restrito do `.gitignore` cobre video e peso de modelo por extensao. O que ele
nao cobria, medido, era artefato de trabalho: `features.csv` na raiz, `artifacts/train.parquet`,
`mlruns/`, safetensors e SavedModel do TensorFlow. O primeiro dono de metrica que gravasse um
dataset derivado comitaria marcacao da Footstats sem perceber.
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import pytest

REPOSITORIO = Path(__file__).resolve().parents[2]

GERADO = [
    "var/cache/perception/a.json",
    "var/runs/2026-08-23/events.jsonl",
    "mlruns/0/meta.yaml",
    "artifacts/train.parquet",
    "outputs/relatorio.json",
    "features.csv",
    "embeddings.npy",
    "embeddings.npz",
    "artifacts/x.feather",
    "artifacts/saida.txt",
    "dados/treino.tfrecord",
    "model.safetensors",
    "modelo/pytorch_model.bin",
    "modelo/model.gguf",
    "modelo/model.ckpt",
    "modelo/variables/variables.index",
    "infra/terraform.tfstate",
    "infra/terraform.tfstate.backup",
    "infra/terraform.tfvars",
    "infra/.terraform/providers/x",
    "models/cache/a/saved_model.pb",
    "derivado.tsv",
    "var/runs/x/events.jsonl",
    "predicoes.jsonl",
]

VERSIONAVEL = [
    "src/tests/fixtures/amostra.csv",
    "src/tests/dataset/exemplo.jsonl",
]


@pytest.mark.parametrize("caminho", GERADO)
def test_generated_artefacts_never_become_a_commit(caminho: str) -> None:
    resultado = subprocess.run(
        ["git", "check-ignore", "-q", caminho],
        cwd=REPOSITORIO,
        capture_output=True,
        check=False,
    )

    assert resultado.returncode == 0, f"{caminho} e versionavel hoje"


@pytest.mark.parametrize("caminho", VERSIONAVEL)
def test_a_test_fixture_is_not_swallowed_by_the_artefact_rules(caminho: str) -> None:
    """Fixture de esquema nao e dado do parceiro: e o que prova que o carregador o entende."""
    resultado = subprocess.run(
        ["git", "check-ignore", "-q", caminho],
        cwd=REPOSITORIO,
        capture_output=True,
        check=False,
    )

    assert resultado.returncode != 0, f"{caminho} ficou invisivel para o git"


def test_the_restricted_block_is_still_there() -> None:
    """O bloco diz para nunca ser removido, e um teste vale mais que a frase."""
    texto = (REPOSITORIO / ".gitignore").read_text(encoding="utf-8")

    assert "CONTEÚDO RESTRITO DA FOOTSTATS" in texto
    assert "NÃO REMOVA ESTE BLOCO" in texto


def test_no_source_file_builds_a_path_inside_the_partner_dump() -> None:
    """Sete formas de escrever dentro do dump passavam pelo grep antigo, que via uma so.

    A que mais preocupa e `data_root() / "derivado.parquet"`: `paths.py` publica `data_root` e ele
    e o caminho mais curto para gravar exatamente onde nao se pode.
    """

    padroes = re.compile(
        r"""(?x)
        Path\(\s*["']\.?/?data[/"']        # Path("data/...") e Path("data")
        | open\(\s*["']\.?/?data/          # open("data/...")
        | f["']\.?/?data/                   # f"data/{...}"
        | data_root\([^)]*\)\s*/            # data_root() / "..."
        """
    )
    fonte = Path(__file__).resolve().parents[1]
    define_a_raiz = fonte / "edge" / "paths.py"
    ofensores = sorted(
        str(arquivo.relative_to(fonte))
        for area in ("api", "cli", "contracts", "edge", "mlops")
        for arquivo in (fonte / area).rglob("*.py")
        if arquivo != define_a_raiz and padroes.search(arquivo.read_text(encoding="utf-8"))
    )

    assert ofensores == [], f"{ofensores} montam caminho dentro do dump do parceiro"


def test_the_source_tree_has_no_package_the_gitignore_would_swallow() -> None:
    """`runs/` no ignore faria um pacote `src/runs/` nascer invisivel, e ninguem notaria."""
    pacotes = sorted(p.parent for p in (REPOSITORIO / "src").rglob("__init__.py"))
    engolidos = [
        p.relative_to(REPOSITORIO)
        for p in pacotes
        if subprocess.run(
            ["git", "check-ignore", "-q", str(p.relative_to(REPOSITORIO))],
            cwd=REPOSITORIO,
            capture_output=True,
            check=False,
        ).returncode
        == 0
    ]

    assert engolidos == [], f"{engolidos} sao invisiveis para o git"


CONTEXTO_PROIBIDO = [
    "data/footstats/db/x.csv",
    "data/footstats/cortes/900/900_004_Mexico_gol_10.mp4",
    "var/runs/x/events.jsonl",
    ".git/config",
    ".venv/lib/x.py",
]


@pytest.mark.parametrize("caminho", CONTEXTO_PROIBIDO)
def test_the_build_context_never_carries_partner_material(caminho: str) -> None:
    """Na raiz ha 3,5 GB de clipe, e nao existe CI que construa imagem para pegar regressao aqui.

    Sem o `.dockerignore`, `docker build .` envia tudo ao daemon e um COPY largo demais assa
    material restrito dentro de uma camada. Medido: o contexto cai de 5,3 GB para 16 kB.
    """
    regras = [
        linha.strip()
        for linha in (REPOSITORIO / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if linha.strip() and not linha.startswith("#")
    ]
    primeiro = caminho.split("/")[0]

    assert f"{primeiro}/" in regras or primeiro in regras, f"{caminho} iria para o daemon"


def test_the_compose_mounts_the_partner_data_read_only() -> None:
    """Garantia tecnica, e nao promessa em documento: conferido rodando `touch` no conteiner."""
    compose = (REPOSITORIO / "compose.yaml").read_text(encoding="utf-8")

    assert "./data/footstats:/data:ro" in compose


def test_the_base_image_is_pinned_by_digest() -> None:
    """Tag se move. Sem digest, duas pessoas constroem ambientes diferentes do mesmo arquivo."""

    dockerfile = (REPOSITORIO / "Dockerfile").read_text(encoding="utf-8")

    assert re.search(r"^FROM \S+@sha256:[0-9a-f]{64}$", dockerfile, re.MULTILINE)


def test_no_module_walks_out_of_the_package_to_find_a_file() -> None:
    """`Path(__file__).parents[N]` que sai do pacote quebra quando instalado como wheel.

    Medido: `mlops` fazia isso para achar `config/`, e instalado a raiz resolvia para dentro de
    `site-packages`, onde `config/` nao existe. `xvision run --all` morria com `FileNotFoundError`
    cru apontando para o venv, e o `except` do CLI nao pegava esse tipo. A regra ja estava escrita
    em `edge/paths.py` e cumprida em `edge/`, e quebrada em `mlops/`.

    Teste continua livre para fazer isso: ele sempre roda do repositorio.
    """
    ofensores: list[str] = []
    for arquivo in sorted((REPOSITORIO / "src").rglob("*.py")):
        if "tests" in arquivo.parts:
            continue
        arvore = ast.parse(arquivo.read_text(encoding="utf-8"))
        for no in ast.walk(arvore):
            if not isinstance(no, ast.Subscript):
                continue
            alvo = no.value
            if not (isinstance(alvo, ast.Attribute) and alvo.attr == "parents"):
                continue
            if "__file__" in ast.dump(alvo):
                ofensores.append(f"{arquivo.relative_to(REPOSITORIO)}:{no.lineno}")

    assert not ofensores, (
        f"caminho derivado de __file__ para fora do pacote: {ofensores}. "
        "Use `edge.paths` (relativo ao diretorio de trabalho) ou `importlib.resources` "
        "quando o arquivo for dado empacotado"
    )


# Arquivo nao versionado que um teste pode ler, com o motivo e a obrigacao: quem le tem de
# pular explicitamente quando ele faltar, e nao presumir que existe.
OPCIONAIS = {
    "pipeline.config.json": "config local da pipeline de tarefas, fora do git de proposito",
}


def test_no_test_depends_on_a_file_the_ci_does_not_have() -> None:
    """Arquivo nao versionado existe na maquina de quem escreveu e nao no runner.

    Aconteceu duas vezes nesta branch. A segunda foi um teste que lia `pipeline.config.json`, que
    nao e versionado: passava aqui, quebrava la, e a mensagem de erro nao dizia nada sobre o
    arquivo faltar. E a mesma forma do `uv sync` sem os extras, e a mesma cura, que e cobrar o
    ambiente e nao a intencao.

    A varredura pega caminho escrito a partir da raiz do repositorio, que e como um teste alcanca
    arquivo de configuracao. Caminho construido em tempo de execucao fica fora, e por isso o
    `skip` explicito continua sendo a saida para quem precisa de arquivo opcional.
    """
    listados = subprocess.run(
        ["git", "ls-files"], cwd=REPOSITORIO, capture_output=True, text=True, check=True
    ).stdout.split()
    # `git ls-files` lista arquivo e nao diretorio, entao as pastas entram pelos pais.
    rastreados = set(listados)
    for caminho in listados:
        for pai in Path(caminho).parents:
            rastreados.add(str(pai))

    ausentes: list[str] = []
    for arquivo in sorted((REPOSITORIO / "src" / "tests").rglob("test_*.py")):
        texto = arquivo.read_text(encoding="utf-8")
        for raiz, alvo in re.findall(r'\b(REPOSITORY|REPOSITORIO)\s*/\s*"([^"]+)"', texto):
            del raiz
            if alvo in rastreados or alvo in OPCIONAIS:
                continue
            ausentes.append(f"{arquivo.relative_to(REPOSITORIO)} le {alvo!r}")

    assert not ausentes, (
        f"teste depende de arquivo que o git nao tem: {ausentes}. "
        "Versione o arquivo, ou use pytest.skip quando ele faltar"
    )
