"""Duas raizes com significado unico: `data/` e o que chegou, `var/` e o que o sistema produz."""

from __future__ import annotations

from pathlib import Path

import pytest

from edge.paths import DataUnavailable, data_root, pipeline_config, var_dir


def test_var_defaults_to_var_beside_the_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XVISION_VAR_DIR", raising=False)

    assert var_dir().name == "var"


def test_var_follows_the_environment_when_it_is_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XVISION_VAR_DIR", str(tmp_path / "escrita"))

    assert var_dir() == tmp_path / "escrita"


def test_data_root_points_at_the_partner_dump(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XVISION_DATA_DIR", raising=False)

    assert data_root(required=False).parts[-2:] == ("data", "footstats")


def test_data_root_follows_the_environment_when_it_is_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XVISION_DATA_DIR", str(tmp_path))

    assert data_root(required=False) == tmp_path


def test_a_missing_dump_says_where_to_put_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Quem clona o repositorio nao tem o dump, entao a mensagem precisa ensinar o caminho."""
    ausente = tmp_path / "sem-dump"
    monkeypatch.setenv("XVISION_DATA_DIR", str(ausente))

    with pytest.raises(DataUnavailable) as erro:
        data_root()

    assert str(ausente) in str(erro.value)
    assert "XVISION_DATA_DIR" in str(erro.value)


def test_the_two_roots_never_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Escrever dentro do dump restrito e o modo de falha que esta separacao existe para impedir."""
    monkeypatch.delenv("XVISION_VAR_DIR", raising=False)
    monkeypatch.delenv("XVISION_DATA_DIR", raising=False)

    escrita, entrada = var_dir().resolve(), data_root(required=False).resolve()

    assert escrita != entrada
    assert entrada not in escrita.parents
    assert escrita not in entrada.parents


def test_the_perception_cache_never_writes_inside_the_partner_dump() -> None:
    from edge.cache.perception import PerceptionCache

    entrada = data_root(required=False).resolve()
    raiz = PerceptionCache().root.resolve()

    assert entrada != raiz
    assert entrada not in raiz.parents


def test_a_variavel_escolhe_qual_pipeline_o_servico_usa(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Sem isto o servico so roda o pipeline de stub, que responde sem medir nada."""
    monkeypatch.setenv("XVISION_CONFIG_DIR", str(tmp_path))
    (tmp_path / "pipeline.toml").write_text("", encoding="utf-8")
    (tmp_path / "pipeline.models.toml").write_text("", encoding="utf-8")

    monkeypatch.delenv("XVISION_PIPELINE", raising=False)
    assert pipeline_config().name == "pipeline.toml"

    monkeypatch.setenv("XVISION_PIPELINE", "pipeline.models.toml")
    assert pipeline_config().name == "pipeline.models.toml"
