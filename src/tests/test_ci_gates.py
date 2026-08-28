"""O CI e os portoes locais precisam cobrar a mesma coisa.

Sem isto os dois envelhecem separados: alguem esvazia o script de um job e a MR passa verde num
portao que ninguem esta mais rodando, ou o piso de cobertura fica com dois numeros diferentes em
dois arquivos. O piso mora em `[tool.coverage.report] fail_under` do `pyproject.toml`, entao o
portao local cobra sozinho e este teste compara dois arquivos de verdade, e nao uma constante
escrita aqui.

Tres estagios do CI sairam, e o motivo de cada corte esta aqui:
`boundaries` porque `test_import_boundary.py` ja roda na suite em menos de um segundo, `contract`
porque ele existia para pegar divergencia entre dois pacotes e hoje existe um pacote so, e o
teste do piso de cobertura porque ele testaria o comportamento de uma flag do `pytest-cov`.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, cast

import pytest
import yaml  # pyright: ignore[reportMissingModuleSource]

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - a borda suporta 3.10, onde tomllib ainda nao existe
    import tomli as tomllib

REPOSITORY = Path(__file__).resolve().parents[2]
PROJECT = REPOSITORY / "src"
PYTHON_CI = REPOSITORY / ".gitlab" / "ci" / "python.yml"
COMANDOS = {
    "python:lint": ("uv run ruff check .", "uv run ruff format --check ."),
    "python:types": ("uv run pyright",),
    "python:test": ("pytest",),
    "python:arm64": ("uv export",),
}
MERGE_REQUEST = '$CI_PIPELINE_SOURCE == "merge_request_event"'


INTEGRACAO_BRANCH = "develop"
INTEGRACAO = f'$CI_COMMIT_BRANCH == "{INTEGRACAO_BRANCH}"'

CONFIG_LOCAL = REPOSITORY / "pipeline.config.json"


def pyproject() -> dict[str, Any]:
    return tomllib.loads((PROJECT / "pyproject.toml").read_text(encoding="utf-8"))


def ci() -> dict[str, Any]:
    documento: dict[str, Any] = {}
    for arquivo in (
        PYTHON_CI,
        REPOSITORY / ".gitlab" / "ci" / "terraform.yml",
        REPOSITORY / ".gitlab" / "ci" / "frontend.yml",
        REPOSITORY / ".gitlab" / "ci" / "actions-rules.yml",
    ):
        documento.update(
            cast("dict[str, Any]", yaml.safe_load(arquivo.read_text(encoding="utf-8")))
        )
    return documento


def resolved(job: str) -> dict[str, Any]:
    definicoes = ci()
    herdados: list[str] = []
    atual = job
    while atual in definicoes:
        herdados.append(atual)
        herdado = definicoes[atual].get("extends")
        herdado = [herdado] if isinstance(herdado, str) else list(herdado or [])
        atual = next((nome for nome in herdado if nome in definicoes), "")
    efetivo: dict[str, Any] = {}
    for nome in reversed(herdados):
        efetivo.update(definicoes[nome])
    efetivo["heranca"] = herdados
    return efetivo


def scripts(job: str) -> list[str]:
    return [str(linha) for linha in resolved(job)["script"]]


def before(job: str) -> list[str]:
    return [str(linha) for linha in resolved(job).get("before_script", [])]


def test_the_pipeline_includes_the_python_jobs() -> None:
    raiz = (REPOSITORY / ".gitlab-ci.yml").read_text(encoding="utf-8")

    assert ".gitlab/ci/python.yml" in raiz


@pytest.mark.parametrize("job", sorted(COMANDOS))
def test_every_declared_job_exists_and_runs_on_merge_request(job: str) -> None:
    definicao = resolved(job)
    condicoes = [regra.get("if") for regra in ci()[".actions-rules:python"]["rules"]]

    assert definicao["stage"] == "test"
    assert ".python" in definicao["heranca"]
    assert MERGE_REQUEST in condicoes
    assert INTEGRACAO in condicoes, "o commit de merge da branch de integracao ficaria sem portao"


@pytest.mark.parametrize("job", sorted(COMANDOS))
def test_every_job_still_runs_the_command_it_exists_for(job: str) -> None:
    linhas = " ".join(scripts(job))

    for comando in COMANDOS[job]:
        assert comando in linhas


def test_the_python_rule_does_not_filter_by_changed_paths() -> None:
    assert all("changes" not in regra for regra in ci()[".actions-rules:python"]["rules"])


def test_the_coverage_floor_is_declared_once_and_the_job_only_turns_measurement_on() -> None:
    linha = next(item for item in scripts("python:test") if "pytest" in item)

    assert pyproject()["tool"]["coverage"]["report"]["fail_under"] == 80
    assert "--cov " in f"{linha} "
    assert "--cov-fail-under" not in linha


def test_the_suite_runs_on_the_floor_the_project_declares() -> None:
    piso = re.findall(r">=\s*(\d+\.\d+)", pyproject()["project"]["requires-python"])[0]

    assert resolved("python:lint")["variables"]["UV_PYTHON"] == piso


def _comando(job: str, prefixo: str) -> str:
    """Achar o comando pelo prefixo, e nao colar tudo, e o que liga a flag ao comando certo.

    Colando os passos num texto so, mover `--frozen` do `uv export` para a resolucao passa no
    teste e quebra o CI, porque `--frozen` nao existe no outro comando.
    """
    partes = prefixo.split()
    return next(item for item in scripts(job) if item.split()[: len(partes)] == partes)


def test_the_arm64_job_exports_the_runtime_dependencies_frozen() -> None:
    linha = _comando("python:arm64", "uv export")

    assert "--frozen" in linha
    assert "--no-dev" in linha, "ruff e pyright nunca vao para o dispositivo"
    assert "--no-emit-project" in linha, "o proprio pacote nao e dependencia de terceiro"


def test_the_arm64_job_resolves_the_floor_the_pyproject_declares() -> None:
    piso = re.findall(r">=\s*(\d+\.\d+)", pyproject()["project"]["requires-python"])[0]
    linha = _comando("python:arm64", "uv pip compile")

    assert "aarch64-unknown-linux-gnu" in linha
    assert f"--python-version {piso}" in linha


def test_the_arm64_job_demands_a_binary_wheel_and_not_just_a_resolution() -> None:
    """Sem `--only-binary`, o portao aceita sdist e nao prova nada sobre a placa.

    Medido: `nvidia-pyindex`, que nao publica roda nenhuma, passa com "Would install 1 package"
    no comando antigo, e e recusado com "Wheels are required" no novo.
    """
    assert "--only-binary :all:" in _comando("python:arm64", "uv pip compile")


def test_the_export_feeds_exactly_the_file_the_resolution_reads() -> None:
    """Trocar o alvo de um dos dois deixa o portao resolvendo um arquivo velho, ou nenhum."""
    escrito = _comando("python:arm64", "uv export").split("-o ")[1].split()[0]

    assert escrito in _comando("python:arm64", "uv pip compile")


def test_the_arm64_job_never_installs_anything() -> None:
    """`uv pip compile` resolve sem ambiente, que e por que o job nao tem venv e nem precisa."""
    passos = scripts("python:arm64")

    assert not any(item.startswith("uv pip install") for item in passos)
    assert not any(item.startswith("uv sync") for item in passos)


def test_the_local_interpreter_is_pinned_to_the_same_floor_the_ci_runs() -> None:
    arquivo = PROJECT / ".python-version"

    assert arquivo.exists(), "sem `.python-version` o uv escolhe o interpretador da maquina"
    assert (
        arquivo.read_text(encoding="utf-8").strip()
        == resolved("python:types")["variables"]["UV_PYTHON"]
    )


def test_the_pinned_floor_is_the_floor_the_package_declares() -> None:
    fixado = (PROJECT / ".python-version").read_text(encoding="utf-8").strip()
    declarado = cast("str", pyproject()["project"]["requires-python"])

    assert declarado == f">={fixado}"


TERRAFORM_CI = REPOSITORY / ".gitlab" / "ci" / "terraform.yml"


def test_the_pipeline_includes_the_terraform_jobs() -> None:
    raiz = (REPOSITORY / ".gitlab-ci.yml").read_text(encoding="utf-8")

    assert ".gitlab/ci/terraform.yml" in raiz


@pytest.mark.parametrize("job", ["terraform:fmt", "terraform:validate", "terraform:plan"])
def test_the_terraform_gate_runs_on_merge_request(job: str) -> None:
    definicao = resolved(job)

    assert definicao["stage"] == "test"
    assert MERGE_REQUEST in [regra.get("if") for regra in ci()[".actions-rules:python"]["rules"]]


def test_the_terraform_plan_runs_without_a_credential() -> None:
    """A topologia nao amarra provedor, entao `plan` roda no CI de verdade e nao so `validate`.

    Amarrar a um provedor agora custaria credencial no CI e escolha que o time ainda nao fez.
    """
    assert "terraform:plan" in ci()


def test_the_terraform_gate_never_applies() -> None:
    """A decisao foi produzir infraestrutura declarada sem implantar.

    Sem `apply`, o risco levantado contra Terraform, o de destravar ou apagar
    configuracao de que oito pessoas dependem, nao existe. O que sobra e HCL versionado com
    formato e validacao cobrados, que e entregavel sem risco operacional.
    """
    passos = " ".join(
        scripts("terraform:fmt") + scripts("terraform:validate") + scripts("terraform:plan")
    )

    assert "apply" not in passos
    assert "destroy" not in passos
    assert "-backend=false" in " ".join(resolved("terraform:validate")["before_script"])


def test_the_ci_installs_every_extra_the_package_declares() -> None:
    """Sincronizar sem os extras deixa verde na maquina e vermelho no CI, sem ninguem ver.

    Medido no clone limpo: `uv sync --frozen` sem extras derruba a coleta inteira do pytest com
    `No module named 'fastapi'` e leva o pyright a 81 erros. Aqui passava porque o `.venv` ja
    carregava o extra de um `uv run --extra` anterior, que o CI nao tem. Enquanto os portoes
    lerem o pacote todo, o ambiente deles precisa conter o pacote todo.
    """
    extras = pyproject()["project"].get("optional-dependencies", {})

    assert extras, "sem extra declarado este teste vira decoracao e deve ser revisto"
    for job in ("python:lint", "python:types", "python:test"):
        instalacao = next(item for item in before(job) if item.startswith("uv sync"))
        assert "--all-extras" in instalacao, (
            f"{job} sincroniza menos do que o pyproject declara, e os portoes leem "
            f"codigo que importa {', '.join(sorted(extras))}"
        )


def test_the_integration_branch_here_is_the_one_the_pipeline_really_targets() -> None:
    """A constante acima e conferida contra o `pipeline.config.json` quando ele existe.

    Ele nao e versionado, entao o CI nao o tem: derivar dele fazia o teste passar na maquina de
    quem tem o arquivo e quebrar no runner, que e a mesma classe de erro que o `--all-extras`
    corrigiu. Aqui a constante e a fonte no CI, e o arquivo local vira conferencia extra para
    quem o tiver: se o alvo mudar, quem roda em casa descobre antes de abrir a MR.
    """
    if not CONFIG_LOCAL.is_file():
        pytest.skip("pipeline.config.json nao e versionado e nao existe neste ambiente")

    alvo = json.loads(CONFIG_LOCAL.read_text(encoding="utf-8"))["targetBranch"]

    assert alvo == INTEGRACAO_BRANCH, (
        f"as MRs apontam para {alvo!r} e o portao do CI defende {INTEGRACAO_BRANCH!r}"
    )


FRONTEND_CI = REPOSITORY / ".gitlab" / "ci" / "frontend.yml"
INTERFACE = REPOSITORY / "src" / "interface"

FRONTEND = {
    "frontend:lint": "npm run lint",
    "frontend:format": "npm run format:check",
    "frontend:types": "npm run types:check",
    "frontend:test": "npm run test",
    "frontend:build": "npm run build",
}


def package_json() -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads((INTERFACE / "package.json").read_text("utf-8")))


def test_the_pipeline_includes_the_frontend_jobs() -> None:
    raiz = (REPOSITORY / ".gitlab-ci.yml").read_text(encoding="utf-8")

    assert ".gitlab/ci/frontend.yml" in raiz


@pytest.mark.parametrize("job", sorted(FRONTEND))
def test_every_frontend_job_runs_the_script_it_exists_for(job: str) -> None:
    """Esvaziar o script de um job deixa a MR verde num portao que ninguem esta mais rodando."""
    assert FRONTEND[job] in " ".join(scripts(job))


@pytest.mark.parametrize("job", sorted(FRONTEND))
def test_every_frontend_job_only_runs_when_the_interface_changes(job: str) -> None:
    """O filtro de caminho e a diferenca entre este portao e o de Python, e ele tem de existir.

    Sem `changes`, toda MR de documentacao paga `npm ci` sem provar nada; com `changes` apontando
    para a pasta errada, o portao some sem ninguem ver.
    """
    definicao = resolved(job)
    regras = ci()[".actions-rules:frontend"]["rules"]
    condicoes = [regra.get("if") for regra in regras]

    assert definicao["stage"] == "test"
    assert ".frontend" in definicao["heranca"]
    assert MERGE_REQUEST in condicoes
    assert INTEGRACAO in condicoes, "o commit de merge da branch de integracao ficaria sem portao"
    for regra in regras:
        assert regra.get("changes") == ["src/interface/**/*"], (
            f"{job} nao filtra pela pasta da interface: {regra.get('changes')}"
        )


@pytest.mark.parametrize("job", sorted(FRONTEND))
def test_every_frontend_command_exists_in_the_package(job: str) -> None:
    """Renomear um script no `package.json` sem tocar o CI derruba o portao, e nao a MR."""
    script = FRONTEND[job].removeprefix("npm run ")

    assert script in package_json()["scripts"], f"{job} chama `npm run {script}`, que nao existe"


def test_the_frontend_build_receives_the_api_address_the_bundle_bakes_in() -> None:
    """Sem a variavel, o Vite assa `undefined` no lugar do endereco e nenhum portao repara."""
    assert resolved("frontend:build")["variables"]["VITE_API_URL"]


def test_the_frontend_coverage_floor_lives_in_the_config_and_the_job_only_measures() -> None:
    """O piso mora num lugar so, como o `fail_under` do lado Python."""
    configuracao = (INTERFACE / "vite.config.ts").read_text(encoding="utf-8")
    definicao = resolved("frontend:test")

    assert "thresholds" in configuracao, "sem piso declarado a cobertura pode cair sem avisar"
    assert "--coverage" in package_json()["scripts"]["test"]
    assert definicao["coverage"], "sem isto o GitLab nao desenha a porcentagem na MR"
