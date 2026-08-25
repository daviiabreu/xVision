"""O que o frontend le de cada etapa, afirmado por valor e nao por chave.

Tres mutacoes passavam na suite inteira: cravar o status da etapa em verdadeiro, apagar a unidade
de toda metrica, e repetir as metricas em todas as etapas. As tres mudam o que a tela mostra, e
nenhuma derrubava teste, porque `config/pipeline.toml` nao declara feature e as listas saiam
vazias: o laco do teste nao rodava, e laco que nao roda nao afirma nada.

Aqui a configuracao declara feature de verdade, com unidade, e uma etapa que falha.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api import PREFIXO, create_app, execution
from edge.testing.example_metric import UNIDADE, VALOR
from edge.testing.make_clip import make_synthetic_clip

from .apoio import cliente_autenticado

MP4 = "video/mp4"


@pytest.fixture
def config(tmp_path: Path) -> Path:
    """A configuracao de producao mais duas features e um modulo que quebra ao carregar."""
    base = Path("config/pipeline.toml").read_text(encoding="utf-8")
    destino = tmp_path / "com-features.toml"
    destino.write_text(
        base
        + "\n[features.distancia_falsa]\n"
        + 'impl = "edge.testing.example_metric:DistanciaFalsa"\n'
        + "\n[features.sem_leitura]\n"
        + 'impl = "edge.testing.example_metric:SemLeitura"\n',
        encoding="utf-8",
    )
    return destino


@pytest.fixture
def cliente(config: Path) -> Iterator[TestClient]:
    with cliente_autenticado(create_app(config)) as aberto:
        yield aberto


def _resultado(cliente: TestClient, tmp_path: Path) -> dict[str, Any]:
    video = make_synthetic_clip(tmp_path / "s.mp4").read_bytes()
    reserva = cliente.post(
        f"{PREFIXO}/videos/upload-url", json={"fileName": "c.mp4", "contentType": MP4}
    ).json()
    cliente.put(reserva["uploadUrl"], content=video)
    execution.aguardar()
    resposta = cliente.get(f"{PREFIXO}/videos/{reserva['id']}")
    assert resposta.status_code == 200, resposta.text
    return dict(resposta.json())


def test_the_metrics_are_not_empty_in_this_scenario(cliente: TestClient, tmp_path: Path) -> None:
    """A guarda dos outros: com lista vazia, todo teste abaixo passaria sem afirmar nada."""
    corpo = _resultado(cliente, tmp_path)

    metricas = [m for r in corpo["resultados"] for m in r["metricas"]]
    assert len(metricas) == 2, f"o cenario nao produziu metrica: {corpo['resultados']}"


def test_a_metric_carries_the_unit_its_feature_declares(
    cliente: TestClient, tmp_path: Path
) -> None:
    """A unidade vem da `FeatureSpec`, e apagada ela deixa o numero sem significado.

    O teste anterior conferia as chaves do dicionario, entao cravar a unidade em vazio passava.
    """
    corpo = _resultado(cliente, tmp_path)

    por_nome = {m["nome"]: m for r in corpo["resultados"] for m in r["metricas"]}
    assert por_nome["distancia_falsa"]["unidade_de_medida"] == UNIDADE
    assert por_nome["distancia_falsa"]["valor"] == VALOR
    assert por_nome["distancia_falsa"]["tipo"] == "float"


def test_a_feature_that_could_not_measure_does_not_look_like_one_that_did(
    cliente: TestClient, tmp_path: Path
) -> None:
    """Numero medido e numero indisponivel nao podem chegar com a mesma cara.

    E a promessa que `edge.features.base` faz por escrito, e ela so vale se atravessar a fronteira
    HTTP: aqui o valor vem nulo e o status diz por que.
    """
    corpo = _resultado(cliente, tmp_path)

    por_nome = {m["nome"]: m for r in corpo["resultados"] for m in r["metricas"]}
    assert por_nome["sem_leitura"]["status"] == "unavailable"
    assert por_nome["sem_leitura"]["valor"] is None
    assert por_nome["distancia_falsa"]["status"] == "ok"


def test_the_metrics_travel_once_and_not_on_every_stage(
    cliente: TestClient, tmp_path: Path
) -> None:
    """Repetidas em todas as etapas, o frontend desenha a mesma medida sete vezes."""
    corpo = _resultado(cliente, tmp_path)

    com_metrica = [r["nome_processo"] for r in corpo["resultados"] if r["metricas"]]
    assert com_metrica == ["shot"], f"as metricas apareceram em {com_metrica}"


def test_a_stage_that_failed_is_not_reported_to_the_frontend_as_successful(tmp_path: Path) -> None:
    """`status: true` numa etapa que quebrou e a tela dizendo que deu certo quando nao deu."""
    base = Path("config/pipeline.toml").read_text(encoding="utf-8")
    quebrado = tmp_path / "com-modulo-quebrado.toml"
    trocado = re.sub(
        r'(\[modules\.players\]\s*\n\s*impl\s*=\s*)"[^"]+"',
        r'\1"edge.testing.example_module:QuebraNoLoad"',
        base,
    )
    assert trocado != base, "a configuracao mudou de forma e o teste nao trocou o modulo"
    quebrado.write_text(trocado, encoding="utf-8")

    with cliente_autenticado(create_app(quebrado)) as cliente:
        corpo = _resultado(cliente, tmp_path)

    por_etapa = {r["nome_processo"]: r["status"] for r in corpo["resultados"]}
    assert por_etapa["players"] is False, "a etapa que falhou chegou ao frontend como sucesso"
    assert por_etapa["ball"] is True, "as que rodaram continuam verdadeiras"
    assert sum(1 for v in por_etapa.values() if v is False) == 1, (
        "so a etapa quebrada devia falhar: nenhum modulo declara `needs` nesta configuracao"
    )
