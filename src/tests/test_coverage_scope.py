"""Modulo sem teste precisa entrar no relatorio com 0%, nao sumir dele.

Area e cada pasta declarada em `coverage.run.source`. Scripts que exigem modelos, GPU ou dados
locais podem ficar fora dessa lista sem criar uma falha artificial na cobertura.

Quando `source` nao aponta para diretorio, o coverage lista apenas o que foi importado, e
codigo sem teste nenhum desaparece em vez de puxar o total para baixo. O piso de cobertura
vira decoracao e ninguem percebe.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
from coverage import Coverage

SRC = Path(__file__).resolve().parents[1]
ROOT = SRC.parent


def _fontes_declaradas() -> list[str]:
    configurada = Coverage(config_file=str(ROOT / "pyproject.toml")).config.source
    return list(configurada or [])


def _executa_line_tracking(monkeypatch: pytest.MonkeyPatch) -> None:
    modulo_falso = types.ModuleType("ultralytics")

    class YOLO:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def train(self, *args: object, **kwargs: object) -> None:
            return None

        def predict(self, *args: object, **kwargs: object) -> list[object]:
            return []

    modulo_falso.__dict__["YOLO"] = YOLO
    monkeypatch.setitem(sys.modules, "ultralytics", modulo_falso)

    for arquivo in (
        ROOT / "src" / "models" / "line_tracking" / "dataset.py",
        ROOT / "src" / "models" / "line_tracking" / "train.py",
        ROOT / "src" / "models" / "line_tracking" / "validate.py",
    ):
        nome = f"coverage_probe_{arquivo.stem}"
        sys.modules.pop(nome, None)
        spec = importlib.util.spec_from_file_location(nome, arquivo)
        assert spec is not None and spec.loader is not None
        modulo = importlib.util.module_from_spec(spec)
        sys.modules[nome] = modulo
        spec.loader.exec_module(modulo)


AREAS = tuple(sorted((ROOT / fonte).resolve() for fonte in _fontes_declaradas()))


@pytest.fixture
def measured_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> set[Path]:
    monkeypatch.chdir(ROOT)
    coverage = Coverage(
        config_file=str(ROOT / "pyproject.toml"),
        data_file=str(tmp_path / ".coverage"),
    )
    coverage.config.disable_warnings = ["no-data-collected"]
    coverage.start()
    _executa_line_tracking(monkeypatch)
    coverage.stop()
    coverage.save()
    return {Path(name).resolve() for name in coverage.get_data().measured_files()}


def test_o_repositorio_tem_area_para_conferir() -> None:
    assert len(AREAS) > 1, f"nenhuma area em {SRC} alem de tests"


@pytest.mark.parametrize("area", AREAS, ids=lambda path: path.name)
def test_todo_modulo_da_area_entra_no_relatorio(area: Path, measured_files: set[Path]) -> None:
    modulos = set(area.rglob("*.py"))

    assert modulos, f"{area} nao tem modulo nenhum, o teste perderia o sentido"
    assert modulos <= measured_files, f"a cobertura nao enxerga {sorted(modulos - measured_files)}"


def test_a_fonte_da_cobertura_nao_alcanca_os_testes() -> None:
    testes = Path(__file__).resolve().parent
    fontes = [(ROOT / declarada).resolve() for declarada in _fontes_declaradas()]

    assert fontes, "a cobertura nao declara fonte nenhuma"
    for fonte in fontes:
        assert fonte != testes and fonte not in testes.parents, (
            f"a fonte {fonte} alcanca {testes}, entao a suite entra no relatorio"
        )
