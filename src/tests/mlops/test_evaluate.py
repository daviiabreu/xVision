"""O unico lugar do projeto onde aparece um numero de acuracia."""

from __future__ import annotations

from typing import Any, cast

import pytest

from mlops.evaluate import EvaluationError, MetricReport, evaluate
from mlops.labels import Label, LabelError, Labels, Metric

MARKED = Label(
    name="marked",
    table="t",
    column="marked",
    type="bool",
    domain=("true", "false"),
    missing=("",),
    required=True,
)
DISTANCIA = Label(
    name="shot_distance_m",
    table="t",
    column="meters",
    type="numeric",
    missing=("",),
    grain=5.0,
)
TABELA = Labels(
    labels={"marked": MARKED, "shot_distance_m": DISTANCIA},
    metrics=(Metric("marked", "marked"), Metric("distancia", "shot_distance_m")),
)


def _verdade(**kw: str) -> dict[str, str]:
    return {"marked": "true", "meters": "10", **kw}


def test_a_bool_metric_reports_accuracy_and_the_majority_baseline_beside_it() -> None:
    """Tres das sete metricas parecem quase resolvidas por chute constante, entao acuracia
    sozinha e mentira que este conjunto convida."""
    gabarito = {str(i): _verdade(marked="true" if i < 9 else "false") for i in range(10)}
    predicoes = {str(i): {"marked": (True, None)} for i in range(10)}

    (metrica,) = [m for m in evaluate(predicoes, gabarito, TABELA).metrics if m.feature == "marked"]

    assert metrica.n_labeled == 10
    assert metrica.scores["accuracy"] == pytest.approx(0.9)
    assert metrica.baseline_majority is not None
    assert metrica.baseline_majority["accuracy"] == pytest.approx(0.9)


def test_unavailable_is_its_own_category_and_never_a_silent_miss() -> None:
    """`unavailable` e a forma em numero de declarar entrega parcial, que a rubrica premia."""
    gabarito = {str(i): _verdade() for i in range(10)}
    predicoes = {
        str(i): {"marked": (True, None) if i < 6 else (None, "sem homografia")} for i in range(10)
    }

    (metrica,) = [m for m in evaluate(predicoes, gabarito, TABELA).metrics if m.feature == "marked"]

    assert (metrica.n_predicted, metrica.n_unavailable) == (6, 4)
    assert metrica.coverage == pytest.approx(0.6)
    assert metrica.scores["accuracy"] == pytest.approx(1.0)


def test_a_numeric_metric_carries_the_grain_of_the_label_beside_the_error() -> None:
    """MAE abaixo de metade do grao mede o rotulo, e nao o modelo."""
    gabarito = {str(i): _verdade() for i in range(4)}
    predicoes = {str(i): {"distancia": (12.0, None)} for i in range(4)}

    (metrica,) = [
        m for m in evaluate(predicoes, gabarito, TABELA).metrics if m.feature == "distancia"
    ]

    assert metrica.scores["mae"] == pytest.approx(2.0)
    assert metrica.label_grain == {"unit": "m", "step": 5.0}
    assert metrica.baseline_majority is None


def test_a_numeric_prediction_for_a_bool_label_reports_the_declared_threshold_and_the_best() -> (
    None
):
    """O limiar continua sendo do dono, e quem ainda nao tem curva nao e obrigado a chutar um."""
    gabarito = {str(i): _verdade(marked="true" if i < 5 else "false") for i in range(10)}
    predicoes = {str(i): {"marked": (0.9 if i < 5 else 0.1, None)} for i in range(10)}

    (metrica,) = [m for m in evaluate(predicoes, gabarito, TABELA).metrics if m.feature == "marked"]

    assert metrica.threshold is not None
    assert metrica.threshold["best_f1"] == pytest.approx(1.0)


def test_a_feature_with_no_label_in_the_table_is_refused_instead_of_scored() -> None:
    with pytest.raises(EvaluationError, match="orfa"):
        evaluate({"1": {"orfa": (True, None)}}, {"1": _verdade()}, TABELA)


def test_the_report_carries_no_shot_id_because_it_is_aggregate_by_construction() -> None:
    """Predicao individual sobre clipe do parceiro nunca pode virar arquivo descritivel."""
    import json

    gabarito = {str(i): _verdade() for i in range(3)}
    relatorio = evaluate({str(i): {"marked": (True, None)} for i in range(3)}, gabarito, TABELA)

    texto = json.dumps(relatorio.to_dict())
    assert "shot_id" not in texto
    for identificador in gabarito:
        assert f'"{identificador}"' not in texto


def test_a_declared_metric_with_no_prediction_shows_up_with_zero_coverage() -> None:
    """Sumir da tabela daria impressao de que ninguem prometeu, quando sete pessoas prometeram."""
    gabarito = {str(i): _verdade() for i in range(5)}

    relatorio = evaluate({}, gabarito, TABELA)

    marcado = next(m for m in relatorio.metrics if m.feature == "marked")
    assert (marcado.n_labeled, marcado.n_predicted, marcado.coverage) == (5, 0, 0.0)
    assert marcado.baseline_majority is not None


def test_the_macro_f1_covers_every_class_the_model_predicted_too() -> None:
    """Iterar so as classes do gabarito dobrava o numero: 0,667 no lugar de 0,333.

    Dez tiros todos `open play`, o modelo acerta cinco e chuta `set piece` nos outros cinco. A
    classe `set piece` nao esta no gabarito destes pares, entao ela sumia do macro.
    """
    jogada = Label(
        name="jogada",
        table="t",
        column="jogada",
        type="enum",
        domain=("open play", "set piece", "individual"),
        missing=("",),
    )
    tabela = Labels(labels={"jogada": jogada}, metrics=(Metric("jogada", "jogada"),))
    gabarito = {str(i): {"jogada": "open play"} for i in range(10)}
    predicoes = {
        str(i): {"jogada": ("open play" if i < 5 else "set piece", None)} for i in range(10)
    }

    (metrica,) = evaluate(predicoes, gabarito, tabela).metrics

    assert metrica.scores["macro_f1"] == pytest.approx(1 / 3, abs=0.01)


def test_an_integer_prediction_does_not_make_the_three_columns_contradict() -> None:
    """`int` 0/1 passava cru: accuracy 1,0, macro_f1 0,0 e confusao errada, na mesma linha."""
    gabarito = {str(i): _verdade(marked="true" if i < 5 else "false") for i in range(10)}
    predicoes = {str(i): {"marked": (1 if i < 5 else 0, None)} for i in range(10)}

    (metrica,) = [m for m in evaluate(predicoes, gabarito, TABELA).metrics if m.feature == "marked"]

    assert metrica.scores["accuracy"] == pytest.approx(1.0)
    assert metrica.scores["macro_f1"] == pytest.approx(1.0)
    assert metrica.confusion == {"tp": 5, "fp": 0, "fn": 0, "tn": 5}


def test_the_baseline_is_measured_on_the_same_population_the_scores_are() -> None:
    """Modelo que so responde nos faceis imprimia acuracia 1,000 ao lado de maioria 0,894.

    Comparar um numero sobre o subconjunto coberto com outro sobre a populacao inteira e a
    mentira que este relatorio existe para impedir.
    """
    gabarito = {str(i): _verdade(marked="true" if i < 9 else "false") for i in range(10)}
    predicoes = {
        str(i): {"marked": (True, None) if i < 3 else (None, "sem cobertura")} for i in range(10)
    }

    (metrica,) = [m for m in evaluate(predicoes, gabarito, TABELA).metrics if m.feature == "marked"]

    assert metrica.n_predicted == 3
    assert metrica.scores["accuracy"] == pytest.approx(1.0)
    assert metrica.baseline_majority is not None
    assert metrica.baseline_majority["accuracy"] == pytest.approx(1.0)
    assert metrica.baseline_majority["n"] == 3


def test_a_metric_with_nothing_measured_has_no_score_instead_of_zero() -> None:
    """ "Mediu zero" e "nao mediu nada" nao podem ocupar a mesma celula do relatorio."""
    gabarito = {str(i): _verdade() for i in range(5)}

    (metrica,) = [m for m in evaluate({}, gabarito, TABELA).metrics if m.feature == "marked"]

    assert metrica.scores == {}
    assert metrica.confusion is None


def test_the_declared_threshold_includes_the_boundary() -> None:
    """`>= 0.5` e `> 0.5` decidem diferente exatamente em 0,5, que e o valor que todo mundo usa."""
    gabarito = {"1": _verdade(marked="true")}
    predicoes = {"1": {"marked": (0.5, None)}}

    (metrica,) = [m for m in evaluate(predicoes, gabarito, TABELA).metrics if m.feature == "marked"]

    assert metrica.scores["accuracy"] == pytest.approx(1.0)


def test_the_majority_baseline_is_reported_even_when_nothing_was_predicted() -> None:
    """E o piso honesto do projeto: com cobertura 0%, ele e o unico numero que existe, e ele
    e propriedade do rotulo e nao da predicao."""
    gabarito = {str(i): _verdade(marked="true" if i < 9 else "false") for i in range(10)}

    (metrica,) = [m for m in evaluate({}, gabarito, TABELA).metrics if m.feature == "marked"]

    assert metrica.n_predicted == 0
    assert metrica.baseline_majority is not None
    assert metrica.baseline_majority["accuracy"] == pytest.approx(0.9)
    assert metrica.baseline_majority["n"] == 10


def test_a_melhor_constante_do_coberto_sai_ao_lado_da_da_populacao() -> None:
    """Com cobertura parcial, a maioria da populacao nao e a barra do subconjunto avaliado."""
    verdade = {f"s{i}": {"marked": "true" if i < 2 else "false"} for i in range(10)}
    previsto = {f"s{i}": {"marked": (True, None)} for i in range(2)}

    (linha, _) = evaluate(previsto, verdade, TABELA).metrics

    assert linha.baseline_majority is not None
    assert linha.baseline_majority_covered is not None
    assert linha.baseline_majority["accuracy"] == 0.0
    assert linha.baseline_majority_covered["accuracy"] == 1.0
    assert linha.baseline_majority_covered["n"] == 2


def test_o_baseline_do_metadado_atravessa_a_tabela_ate_o_relatorio() -> None:
    """Tres rotulos ja estao resolvidos por coluna tabular, e a maioria sozinha nao e a barra."""
    tabela = Labels(
        labels={"marked": TABELA.label("marked")},
        metrics=(Metric("marked", "marked", baseline_metadata=0.7947),),
    )
    verdade = {"s0": {"marked": "true"}}

    (linha,) = evaluate({"s0": {"marked": (True, None)}}, verdade, tabela).metrics

    assert linha.baseline_metadata == 0.7947
    assert linha.to_dict()["baseline_metadata"] == 0.7947


def test_baseline_de_metadado_fora_da_faixa_e_recusado() -> None:
    with pytest.raises(LabelError):
        Metric("marked", "marked", baseline_metadata=1.4)


def test_mcnemar_conta_so_onde_o_modelo_e_o_constante_discordam() -> None:
    verdade = {f"s{i}": {"marked": "true" if i < 3 else "false"} for i in range(10)}
    previsto = {f"s{i}": {"marked": (i < 3, None)} for i in range(10)}

    (linha, _) = evaluate(previsto, verdade, TABELA).metrics

    assert linha.mcnemar is not None
    assert linha.mcnemar["modelo_acertou_constante_errou"] == 3
    assert linha.mcnemar["modelo_errou_constante_acertou"] == 0


def test_o_p_valor_do_mcnemar_separa_ruido_de_sinal() -> None:
    """Cinco acertos sem falso positivo nao passam de 0,05, e seis passam: e o limiar do exato."""
    from mlops.evaluate import mcnemar

    cinco = [(True, True)] * 5 + [(False, False)] * 20
    seis = [(True, True)] * 6 + [(False, False)] * 20

    assert mcnemar(cinco)["p_value"] >= 0.05
    assert mcnemar(seis)["p_value"] < 0.05


def test_um_modelo_identico_ao_constante_nao_tem_p_valor_significativo() -> None:
    from mlops.evaluate import mcnemar

    pares = [(False, True)] * 3 + [(False, False)] * 20

    assert mcnemar(pares)["p_value"] == 1.0


JOGADA = Label(
    name="jogada",
    table="t",
    column="jogada",
    type="enum",
    domain=("open play", "set piece", "individual"),
    missing=("",),
)
TABELA_JOGADA = Labels(labels={"jogada": JOGADA}, metrics=(Metric("jogada", "jogada"),))


def _jogada() -> MetricReport:
    """Seis abertas, duas bolas paradas e dois individuais; o modelo nunca preve bola parada."""
    rotulos = ["open play"] * 6 + ["set piece"] * 2 + ["individual"] * 2
    previstos = ["open play"] * 5 + ["individual"] + ["open play"] * 2 + ["individual"] * 2
    gabarito = {str(i): {"jogada": r} for i, r in enumerate(rotulos)}
    predicoes = {str(i): {"jogada": (p, None)} for i, p in enumerate(previstos)}
    (metrica,) = evaluate(predicoes, gabarito, TABELA_JOGADA).metrics
    return metrica


def test_a_matriz_do_enum_mostra_a_coluna_da_classe_que_o_modelo_nunca_preve() -> None:
    metrica = _jogada()

    assert metrica.confusion_matrix == {
        "individual": {"individual": 2, "open play": 0, "set piece": 0},
        "open play": {"individual": 1, "open play": 5, "set piece": 0},
        "set piece": {"individual": 0, "open play": 2, "set piece": 0},
    }
    assert metrica.confusion is None


def test_por_classe_diz_qual_classe_o_modelo_errou_e_nao_inventa_zero() -> None:
    """Precisao de uma classe nunca prevista nao e zero: o modelo nao tentou."""
    por_classe = _jogada().per_class

    assert por_classe is not None
    assert por_classe["set piece"] == {
        "precision": None,
        "recall": 0.0,
        "f1": 0.0,
        "support": 2,
        "predicted": 0,
    }
    assert por_classe["open play"]["recall"] == pytest.approx(5 / 6)
    assert por_classe["open play"]["precision"] == pytest.approx(5 / 7)
    assert por_classe["individual"]["precision"] == pytest.approx(2 / 3)


def test_o_macro_f1_e_exatamente_a_media_do_f1_por_classe() -> None:
    metrica = _jogada()

    assert metrica.per_class is not None
    f1s = [cast("float", valores["f1"]) for valores in metrica.per_class.values()]
    assert metrica.scores["macro_f1"] == pytest.approx(sum(f1s) / len(f1s))


def test_booleano_ganha_por_classe_e_continua_sem_matriz() -> None:
    gabarito = {str(i): _verdade(marked="true" if i < 7 else "false") for i in range(10)}
    predicoes = {str(i): {"marked": (i < 5, None)} for i in range(10)}

    (metrica,) = [m for m in evaluate(predicoes, gabarito, TABELA).metrics if m.feature == "marked"]

    assert metrica.confusion_matrix is None
    assert metrica.per_class is not None
    assert set(metrica.per_class) == {"True", "False"}
    assert metrica.per_class["True"]["recall"] == pytest.approx(5 / 7)


def test_numerico_e_metrica_sem_predicao_nao_tem_nada_por_classe() -> None:
    gabarito = {str(i): _verdade() for i in range(4)}
    relatorio = evaluate(
        {str(i): {"distancia": (12.0, None)} for i in range(4)}, gabarito, TABELA
    ).to_dict()

    por_nome = {m["feature"]: m for m in relatorio["metrics"]}
    assert por_nome["distancia"]["per_class"] is None
    assert por_nome["marked"]["per_class"] is None
    assert por_nome["marked"]["confusion_matrix"] is None


def _por_partida(
    acertos: dict[str, int], total: int = 5
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    """Cada partida com `total` chutes; `acertos` deles o modelo acerta."""
    gabarito: dict[str, Any] = {}
    predicoes: dict[str, Any] = {}
    grupos: dict[str, str] = {}
    for partida, certos in acertos.items():
        for i in range(total):
            chute = f"{partida}-{i}"
            gabarito[chute] = _verdade(marked="true")
            predicoes[chute] = {"marked": (i < certos, None)}
            grupos[chute] = partida
    return predicoes, gabarito, grupos


def _marcado(relatorio: Any) -> MetricReport:
    return next(m for m in relatorio.metrics if m.feature == "marked")


def test_o_intervalo_reamostra_partidas_e_contem_o_ponto() -> None:
    predicoes, gabarito, grupos = _por_partida({"a": 5, "b": 5, "c": 0, "d": 1})

    metrica = _marcado(evaluate(predicoes, gabarito, TABELA, groups=grupos, resamples=300))

    assert metrica.bootstrap is not None
    faixa = cast("list[float]", metrica.bootstrap["accuracy"])
    assert metrica.bootstrap["groups"] == 4
    assert faixa[0] <= metrica.scores["accuracy"] <= faixa[1]
    assert faixa[1] - faixa[0] > 0.3


def test_o_intervalo_e_reproduzivel_pela_semente() -> None:
    predicoes, gabarito, grupos = _por_partida({"a": 5, "b": 2, "c": 0})

    uma = _marcado(evaluate(predicoes, gabarito, TABELA, groups=grupos, resamples=200, seed=7))
    outra = _marcado(evaluate(predicoes, gabarito, TABELA, groups=grupos, resamples=200, seed=7))

    assert uma.bootstrap == outra.bootstrap


def test_sem_reamostra_nao_ha_intervalo_e_chute_sem_partida_vira_grupo_proprio() -> None:
    predicoes, gabarito, grupos = _por_partida({"a": 5, "b": 0})

    sem = _marcado(evaluate(predicoes, gabarito, TABELA, groups=grupos))
    orfaos = _marcado(evaluate(predicoes, gabarito, TABELA, groups={}, resamples=50))

    assert sem.bootstrap is None
    assert orfaos.bootstrap is not None and orfaos.bootstrap["groups"] == 10
