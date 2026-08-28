"""O gabarito manual do instante: como ele e guardado, medido, e o que ele nao promete.

O defeito que mais importa aqui nao levanta nada: um gabarito sem barra de erro propria faz todo
erro medido contra ele parecer erro do modelo. Por isso a concordancia entre anotadores e primeira
classe, e nao um extra.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mlops.instant import (
    TOLERANCIA_ALVO,
    TOLERANCIA_LIMITE,
    Anotacao,
    InstantError,
    agreement,
    consensus,
    latest,
    measure,
    read,
    sample,
    wilson,
    write,
)


def _anotar(root: Path, clip: str, quadro: int, quem: str = "davi") -> None:
    write(Anotacao(clip_id=clip, frame_index=quadro, annotator=quem), root=root)


def test_a_anotacao_vai_e_volta_do_disco(tmp_path: Path) -> None:
    _anotar(tmp_path, "c1", 120)

    (lida,) = read(root=tmp_path)

    assert lida == Anotacao("c1", 120, "davi")


def test_sem_arquivo_a_leitura_devolve_vazio_em_vez_de_estourar(tmp_path: Path) -> None:
    assert read(root=tmp_path) == ()


def test_linha_corrompida_e_nomeada_e_nao_engolida(tmp_path: Path) -> None:
    """Anotacao perdida em silencio custaria o trabalho de quem anotou sem ninguem notar."""
    destino = tmp_path / "labels"
    destino.mkdir()
    (destino / "instants.jsonl").write_text('{"clip_id": "c1"}\n', encoding="utf-8")

    with pytest.raises(InstantError):
        read(root=tmp_path)


def test_o_consenso_e_a_mediana_e_nao_a_ultima_anotacao(tmp_path: Path) -> None:
    """Mediana resiste a um anotador destoante sem descartar o trabalho dele."""
    for quadro, quem in ((100, "a"), (104, "b"), (300, "c")):
        _anotar(tmp_path, "c1", quadro, quem)

    assert consensus(read(root=tmp_path)) == {"c1": 104}


def test_sem_clipe_anotado_duas_vezes_nao_ha_concordancia_para_reportar(tmp_path: Path) -> None:
    _anotar(tmp_path, "c1", 100)

    assert agreement(read(root=tmp_path)) is None


def test_a_concordancia_mede_o_espalhamento_entre_anotadores(tmp_path: Path) -> None:
    _anotar(tmp_path, "c1", 100, "a")
    _anotar(tmp_path, "c1", 106, "b")
    _anotar(tmp_path, "c2", 200, "a")

    concordancia = agreement(read(root=tmp_path))

    assert concordancia is not None
    assert concordancia["n_clipes"] == 1
    assert concordancia["max_quadros"] == 6.0


def test_a_medida_conta_so_os_clipes_que_tem_os_dois_lados() -> None:
    relatorio = measure({"c1": 100, "c2": 200, "c3": 5}, {"c1": 110, "c2": 400})

    assert relatorio.n == 2


def test_as_duas_tolerancias_saem_da_configuracao_e_nao_de_gosto() -> None:
    """30 quadros preserva dois tercos do contexto anterior; 90 zera ele."""
    previsto = {"a": 100, "b": 100, "c": 100}
    gabarito = {"a": 110, "b": 160, "c": 400}

    relatorio = measure(previsto, gabarito)

    assert relatorio.dentro_da_tolerancia == pytest.approx(1 / 3)
    assert relatorio.dentro_do_limite == pytest.approx(2 / 3)
    assert relatorio.to_dict()["tolerancia_quadros"] == TOLERANCIA_ALVO
    assert relatorio.to_dict()["limite_quadros"] == TOLERANCIA_LIMITE


def test_sem_clipe_em_comum_o_relatorio_sai_zerado_em_vez_de_estourar() -> None:
    assert measure({"a": 1}, {"b": 2}).n == 0


def test_o_relatorio_separa_medido_de_estimado() -> None:
    """Sem isso, o ponto medio entra na conta como se fosse medida da trajetoria."""
    relatorio = measure(
        {"a": 100, "b": 100},
        {"a": 105, "b": 105},
        fontes={"a": "ball-break", "b": "midpoint"},
    )

    assert relatorio.fontes == {"ball-break": 1, "midpoint": 1}


def test_o_relatorio_nao_carrega_identificador_de_finalizacao() -> None:
    """A mesma regra que ja vale para o `evaluate`: o relatorio e agregado por construcao."""
    relatorio = measure({"4995949": 100}, {"4995949": 105})

    assert "4995949" not in str(relatorio.to_dict())


def test_o_intervalo_de_wilson_encolhe_quando_a_amostra_cresce() -> None:
    """Uma fracao sobre 40 clipes nao e um numero exato, e o relatorio precisa dizer isso."""
    curto = wilson(34, 40)
    longo = wilson(340, 400)

    assert curto[1] - curto[0] > longo[1] - longo[0]


def test_wilson_com_amostra_vazia_devolve_o_intervalo_inteiro() -> None:
    assert wilson(0, 0) == (0.0, 1.0)


def test_a_amostra_e_estratificada_por_partida() -> None:
    """Sem estratificar, o sorteio concentra na partida com mais cortes, que tem tres vezes mais."""
    clipes = [(f"m1-{i}", "m1") for i in range(30)] + [(f"m2-{i}", "m2") for i in range(10)]

    escolhidos = sample(clipes, 6, seed=1)

    assert len(escolhidos) == 6
    assert sum(1 for c in escolhidos if c.startswith("m1")) == 3


def test_a_amostra_e_reproduzivel_pela_semente() -> None:
    clipes = [(f"m{p}-{i}", f"m{p}") for p in range(4) for i in range(10)]

    assert sample(clipes, 8, seed=7) == sample(clipes, 8, seed=7)
    assert sample(clipes, 8, seed=7) != sample(clipes, 8, seed=8)


def test_pedir_mais_do_que_existe_devolve_o_que_existe() -> None:
    assert len(sample([("c1", "m1"), ("c2", "m1")], 10, seed=1)) == 2


def test_correcao_da_mesma_pessoa_nao_vira_discordancia_entre_anotadores() -> None:
    """Quem volta num clipe e marca de novo nao e um segundo anotador.

    Sem isto, `agreement` mede a pessoa contra ela mesma e o gabarito ganha uma barra de erro que
    nao existe: a correcao de 40 para 41 apareceria como um clipe com dois anotadores discordando.
    """
    anotacoes = (
        Anotacao("c1", 40, "davi"),
        Anotacao("c1", 41, "davi"),
    )

    assert consensus(anotacoes) == {"c1": 41}
    assert agreement(anotacoes) is None


def test_duas_pessoas_no_mesmo_clipe_continuam_contando() -> None:
    anotacoes = (
        Anotacao("c1", 40, "davi"),
        Anotacao("c1", 45, "claude"),
        Anotacao("c1", 41, "davi"),
    )

    medido = agreement(anotacoes)

    assert medido is not None
    assert medido["n_clipes"] == 1
    assert medido["max_quadros"] == 4.0


def test_latest_guarda_a_ultima_de_cada_pessoa() -> None:
    anotacoes = (
        Anotacao("c1", 10, "davi"),
        Anotacao("c2", 20, "davi"),
        Anotacao("c1", 11, "davi"),
        Anotacao("c1", 99, "claude"),
    )

    assert sorted((a.clip_id, a.frame_index, a.annotator) for a in latest(anotacoes)) == [
        ("c1", 11, "davi"),
        ("c1", 99, "claude"),
        ("c2", 20, "davi"),
    ]
