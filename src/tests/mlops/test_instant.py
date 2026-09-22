"""O gabarito manual do instante: como ele e guardado, medido, e o que ele nao promete.

O defeito que mais importa aqui nao levanta nada: um gabarito sem barra de erro propria faz todo
erro medido contra ele parecer erro do modelo. Por isso a concordancia entre anotadores e primeira
classe, e nao um extra.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mlops.instant import (
    ACERTO,
    DISCORDANTES_PARA_SINAL,
    ESCOLHA,
    ESTIMATIVA,
    NAO_DECLARADO,
    TOLERANCIA_ALVO,
    TOLERANCIA_LIMITE,
    TRAJETORIA,
    Anotacao,
    BaselineReport,
    InstantError,
    Resolucao,
    agreement,
    baselines,
    consensus,
    diagnose,
    latest,
    measure,
    read,
    read_spots,
    resolution,
    sample,
    taxonomy,
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


def _candidato(quadro: int) -> dict[str, object]:
    return {"frame_index": quadro, "sample_index": quadro, "score": 0.5}


def test_erro_dentro_da_tolerancia_e_acerto_e_nao_tem_causa() -> None:
    (achado,) = diagnose({"c": 100}, {"c": 100 + TOLERANCIA_ALVO})

    assert achado.tipo == ACERTO
    assert achado.erro == TOLERANCIA_ALVO


def test_quebra_certa_em_segundo_lugar_e_erro_de_escolha() -> None:
    """O unico tipo em que reordenar candidatos resolve, e por isso ele existe separado."""
    (achado,) = diagnose(
        {"c": 400},
        {"c": 100},
        fontes={"c": "ball-break"},
        candidatos={"c": [_candidato(400), _candidato(105)]},
    )

    assert achado.tipo == ESCOLHA
    assert achado.melhor_rank == 2
    assert achado.melhor_erro == 5


def test_nenhuma_quebra_perto_do_gabarito_e_erro_de_trajetoria() -> None:
    """Aqui reordenar nao compra nada: a quebra do contato nunca chegou a existir."""
    (achado,) = diagnose(
        {"c": 400},
        {"c": 100},
        fontes={"c": "ball-break"},
        candidatos={"c": [_candidato(400), _candidato(380)]},
    )

    assert achado.tipo == TRAJETORIA
    assert achado.melhor_rank is None
    assert achado.melhor_erro == 280


def test_ponto_medio_e_estimativa_mesmo_com_candidato_bom_na_lista() -> None:
    """Candidato bom abaixo do limiar e problema de limiar, nao de ordenacao."""
    (achado,) = diagnose(
        {"c": 400},
        {"c": 100},
        fontes={"c": "midpoint"},
        candidatos={"c": [_candidato(102)]},
    )

    assert achado.tipo == ESTIMATIVA


def test_execucao_sem_candidato_registrado_nao_vira_erro_de_trajetoria() -> None:
    """O defeito replantado: ausencia de registro lida como ausencia de quebra.

    Execucao anterior ao campo nao sabe nada sobre candidato, e chamar isso de `trajetoria`
    produziria uma taxonomia inteira apontando para a deteccao da bola sem nenhuma evidencia.
    """
    (achado,) = diagnose({"c": 400}, {"c": 100}, fontes={"c": "ball-break"})

    assert achado.tipo == NAO_DECLARADO
    assert achado.melhor_erro is None


def test_taxonomy_mede_a_cobertura_das_causas_sobre_os_erros() -> None:
    achados = diagnose(
        {"ok": 100, "escolheu": 400, "sem_registro": 400},
        {"ok": 100, "escolheu": 100, "sem_registro": 100},
        fontes={"ok": "ball-break", "escolheu": "ball-break", "sem_registro": "ball-break"},
        candidatos={"escolheu": [_candidato(400), _candidato(100)]},
    )

    relatorio = taxonomy(achados)

    assert relatorio.n == 3
    assert relatorio.erros == 2
    assert relatorio.cobertura == 0.5
    assert relatorio.tipos[ACERTO] == 1


def test_recall_em_k_mede_o_teto_de_reordenar() -> None:
    """A distancia entre `top1` e `top5` e o que um criterio de escolha melhor teria para ganhar."""
    achados = diagnose(
        {"a": 400, "b": 400},
        {"a": 100, "b": 100},
        fontes={"a": "ball-break", "b": "ball-break"},
        candidatos={
            "a": [_candidato(100)],
            "b": [_candidato(400), _candidato(390), _candidato(100)],
        },
    )

    relatorio = taxonomy(achados)

    assert relatorio.recall["top1"] == 0.5
    assert relatorio.recall["top2"] == 0.5
    assert relatorio.recall["top3"] == 1.0


def test_taxonomy_sem_erro_declara_cobertura_cheia() -> None:
    relatorio = taxonomy(diagnose({"c": 100}, {"c": 100}))

    assert relatorio.erros == 0
    assert relatorio.cobertura == 1.0


def test_o_acaso_sai_ao_lado_do_teto_de_reordenar() -> None:
    """O defeito replantado nao levanta nada: um teto sem linha de base parece maior do que e.

    Com 50 quebras no clipe e uma so dentro da tolerancia, sortear uma acerta 1 em 50. O
    `recall_em_k` sozinho nao diz isso, e foi assim que o primeiro relatorio desta sprint
    apresentou 0,833 como se fossem 54 pontos percentuais de ganho disponivel.
    """
    quadros = [1000 + 100 * i for i in range(49)] + [100]
    achados = diagnose(
        {"c": 5000},
        {"c": 100},
        fontes={"c": "ball-break"},
        candidatos={"c": [_candidato(5000), _candidato(100)]},
    )

    relatorio = taxonomy(achados, gabarito={"c": 100}, pools={"c": quadros})

    assert relatorio.n_com_conjunto == 1
    assert relatorio.recall["top2"] == 1.0
    assert relatorio.acaso["top1"] == pytest.approx(1 / 50)
    assert relatorio.acaso["top2"] == pytest.approx(1 - (49 * 48) / (50 * 49))


def test_sem_conjunto_de_quebras_nao_ha_linha_de_base() -> None:
    """Execucao que nao declarou o conjunto nao pode produzir acaso inventado."""
    relatorio = taxonomy(diagnose({"c": 100}, {"c": 400}), gabarito={"c": 400})

    assert relatorio.acaso == {}
    assert relatorio.n_com_conjunto == 0


def test_conjunto_menor_que_k_acerta_sempre_quando_ha_quebra_boa() -> None:
    """Sortear 5 de 3 e pegar as 3, e ai o acaso e 1 se alguma serve e 0 se nenhuma serve."""
    boa = taxonomy(
        diagnose({"c": 400}, {"c": 100}),
        gabarito={"c": 100},
        pools={"c": [100, 900, 1800]},
    )
    ruim = taxonomy(
        diagnose({"c": 400}, {"c": 100}),
        gabarito={"c": 100},
        pools={"c": [900, 1500, 1800]},
    )

    assert boa.acaso["top5"] == 1.0
    assert ruim.acaso["top5"] == 0.0


def test_read_spots_recusa_o_mesmo_clipe_duas_vezes(tmp_path: Path) -> None:
    """Duplicata silenciosa faz um detector com duas saidas passar por um com uma."""
    caminho = tmp_path / "spots.jsonl"
    caminho.write_text(
        '{"clip_id": "a", "frame_index": 10}\n{"clip_id": "a", "frame_index": 90}\n',
        encoding="utf-8",
    )

    with pytest.raises(InstantError, match="duas vezes"):
        read_spots(caminho)


def test_read_spots_recusa_booleano_como_quadro(tmp_path: Path) -> None:
    """`bool` e `int` em Python, entao `true` entraria como quadro 1 sem a guarda."""
    caminho = tmp_path / "spots.jsonl"
    caminho.write_text('{"clip_id": "a", "frame_index": true}\n', encoding="utf-8")

    with pytest.raises(InstantError, match="frame_index"):
        read_spots(caminho)


def test_read_spots_recusa_arquivo_sem_spot(tmp_path: Path) -> None:
    caminho = tmp_path / "spots.jsonl"
    caminho.write_text("\n\n", encoding="utf-8")

    with pytest.raises(InstantError, match="nenhum spot"):
        read_spots(caminho)


def test_read_spots_le_um_instante_por_clipe(tmp_path: Path) -> None:
    caminho = tmp_path / "spots.jsonl"
    caminho.write_text(
        '{"clip_id": "a", "frame_index": 10, "score": 0.9}\n{"clip_id": "b", "frame_index": 20}\n',
        encoding="utf-8",
    )

    assert read_spots(caminho) == {"a": 10, "b": 20}


def test_o_acaso_e_a_fatia_do_clipe_coberta_pela_tolerancia() -> None:
    """Exato e nao simulado: 61 quadros de janela num clipe de 400 sao 0,1525."""
    relatorio = baselines({"c": 200}, {"c": 400}, tolerancia=30)

    assert relatorio.acaso == pytest.approx(61 / 400)
    assert relatorio.n == 1


def test_o_acaso_conta_so_a_parte_da_janela_dentro_do_clipe() -> None:
    """Gabarito na borda tem meia janela, e contar a janela inteira inflaria o nulo."""
    relatorio = baselines({"c": 5}, {"c": 400}, tolerancia=30)

    assert relatorio.acaso == pytest.approx(36 / 400)


def test_o_ponto_medio_fixo_e_a_linha_de_base_de_maioria() -> None:
    """Os cortes do parceiro contem a finalizacao, entao chutar o meio precisa ser batido."""
    relatorio = baselines({"perto": 200, "longe": 10}, {"perto": 400, "longe": 400}, tolerancia=30)

    assert relatorio.ponto_medio == 0.5


def test_baselines_sem_clipe_em_comum_nao_inventa_numero() -> None:
    relatorio = baselines({"a": 10}, {"b": 400})

    assert relatorio == BaselineReport(0, 0.0, 0.0)


def test_resolucao_diz_quantos_clipes_a_amostra_enxerga() -> None:
    """A taxa de 0,292 em 24 clipes tem meia largura de 4,1 clipes.

    O defeito que isto replanta nao levanta nada e aconteceu varias vezes nesta sprint: uma
    mudanca move um clipe, a taxa sobe, e o relatorio nao diz que um clipe esta dentro do ruido
    por construcao.
    """
    medido = resolution(7, 24)

    assert medido.n == 24
    assert medido.clipes == pytest.approx(4.1, abs=0.1)
    assert medido.fracao == pytest.approx(0.171, abs=0.001)


def test_amostra_maior_enxerga_efeito_menor_em_fracao() -> None:
    pequena = resolution(30, 100)
    grande = resolution(90, 300)

    assert grande.fracao < pequena.fracao
    assert grande.clipes > pequena.clipes


def test_resolucao_de_amostra_vazia_nao_inventa_numero() -> None:
    assert resolution(0, 0) == Resolucao(0, 0.0, 0.0, DISCORDANTES_PARA_SINAL)


def test_cinco_discordantes_e_o_primeiro_que_sustenta_conclusao() -> None:
    """`0,5 ** 4` da 0,0625 e `0,5 ** 5` da 0,0312, entao quatro nao serve e cinco serve."""
    assert 0.5**DISCORDANTES_PARA_SINAL < 0.05
    assert 0.5 ** (DISCORDANTES_PARA_SINAL - 1) >= 0.05
