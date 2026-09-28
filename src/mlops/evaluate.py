"""O unico lugar do projeto onde aparece um numero de acuracia.

Tres campos sao obrigatorios, e cada um existe para impedir uma mentira que este conjunto
convida. `baseline_majority`, porque tres das sete metricas parecem quase resolvidas sem modelo
nenhum: chute constante acerta 0,894 em `looked_before`, 0,876 em `goalkeaper_position` e 0,844
em `marked`, medido sobre os 604. `coverage` e `n_unavailable`, porque indisponivel e categoria
propria e nunca erro nem acerto calado, e e a forma em numero de declarar entrega parcial. E
`label_grain` para numerico, porque `meters` tem grao de 5 m e MAE abaixo de metade disso mede o
rotulo, e nao o modelo.

As metricas saem das entradas `[[metric]]` da tabela, e nao das predicoes. E de proposito: uma
metrica declarada e sem nenhuma predicao aparece com cobertura 0%, que e o estado real do projeto
hoje, em vez de sumir da tabela e dar impressao de que ninguem prometeu nada.

O limiar continua sendo decisao do dono. Se o extractor emite booleano, conta direto; se emite
numero, o relatorio traz o F1 no limiar declarado e tambem o melhor da varredura com o limiar que
o atinge, para quem ainda nao tem curva nao ser obrigado a chutar um.

Nao decide se um numero e bom, nao tem nota de corte, nao plota e nao compara execucoes de
populacao diferente. Nao usa sklearn, pandas nem numpy: e aritmetica sobre listas, para nao
acrescentar nada ao resolvedor da borda. E nao guarda predicao individual nem identificador de
finalizacao, porque o relatorio e agregado por construcao, que e o que a regra de conteudo
restrito permite descrever.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from mlops.labels import Labels

SCHEMA_VERSION = "1.0"
Predicted = tuple[object, str | None]


class EvaluationError(ValueError): ...


@dataclass(frozen=True)
class MetricReport:
    feature: str
    label: str
    type: str
    n_labeled: int
    n_predicted: int
    n_unavailable: int
    coverage: float
    scores: Mapping[str, float]
    confusion: Mapping[str, int] | None = None
    baseline_majority: Mapping[str, float] | None = None
    baseline_majority_covered: Mapping[str, float] | None = None
    baseline_metadata: float | None = None
    mcnemar: Mapping[str, float] | None = None
    threshold: Mapping[str, float] | None = None
    label_grain: Mapping[str, object] | None = None
    per_class: Mapping[str, Mapping[str, float | int | None]] | None = None
    confusion_matrix: Mapping[str, Mapping[str, int]] | None = None
    bootstrap: Mapping[str, object] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "label": self.label,
            "type": self.type,
            "n_labeled": self.n_labeled,
            "n_predicted": self.n_predicted,
            "n_unavailable": self.n_unavailable,
            "coverage": self.coverage,
            "scores": dict(self.scores),
            "confusion": None if self.confusion is None else dict(self.confusion),
            "baseline_majority": (
                None if self.baseline_majority is None else dict(self.baseline_majority)
            ),
            "baseline_majority_covered": (
                None
                if self.baseline_majority_covered is None
                else dict(self.baseline_majority_covered)
            ),
            "baseline_metadata": self.baseline_metadata,
            "mcnemar": None if self.mcnemar is None else dict(self.mcnemar),
            "threshold": None if self.threshold is None else dict(self.threshold),
            "label_grain": None if self.label_grain is None else dict(self.label_grain),
            "per_class": (
                None
                if self.per_class is None
                else {classe: dict(valores) for classe, valores in self.per_class.items()}
            ),
            "confusion_matrix": (
                None
                if self.confusion_matrix is None
                else {classe: dict(linha) for classe, linha in self.confusion_matrix.items()}
            ),
            "bootstrap": None if self.bootstrap is None else dict(self.bootstrap),
        }


@dataclass(frozen=True)
class Report:
    population: str
    metrics: Sequence[MetricReport]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "population": self.population,
            "metrics": [m.to_dict() for m in self.metrics],
        }


def evaluate(
    predictions: Mapping[str, Mapping[str, Predicted]],
    truth: Mapping[str, Mapping[str, str]],
    table: Labels,
    *,
    population: str = "",
    groups: Mapping[str, str] | None = None,
    resamples: int = 0,
    seed: int = 0,
) -> Report:
    declaradas = [m.feature for m in table.metrics]
    extras = sorted(
        {nome for por_tiro in predictions.values() for nome in por_tiro} - set(declaradas)
    )
    features = declaradas + extras
    relatorios = [
        _metric(nome, predictions, truth, table, groups or {}, resamples, seed) for nome in features
    ]
    return Report(population=population or f"n={len(truth)}", metrics=tuple(relatorios))


def _metric(
    feature: str,
    predictions: Mapping[str, Mapping[str, Predicted]],
    truth: Mapping[str, Mapping[str, str]],
    table: Labels,
    groups: Mapping[str, str],
    resamples: int,
    seed: int,
) -> MetricReport:
    metrica = table.metric_for(feature)
    rotulo = table.label_for(feature)
    if rotulo is None:
        raise EvaluationError(
            f"a feature {feature!r} e orfa: nenhuma entrada [[metric]] em config/labels.toml "
            "diz qual coluna do parceiro ela mede"
        )
    pares: list[tuple[object, object]] = []
    por_grupo: dict[str, list[tuple[object, object]]] = {}
    esperados: list[object] = []
    indisponiveis = 0
    for shot_id, colunas in truth.items():
        alvo = rotulo.parse(colunas.get(rotulo.column, ""))
        if alvo is None:
            continue
        esperados.append(alvo)
        previsto = predictions.get(shot_id, {}).get(feature)
        if previsto is None or previsto[0] is None:
            indisponiveis += 1
            continue
        pares.append((previsto[0], alvo))
        por_grupo.setdefault(groups.get(shot_id, shot_id), []).append((previsto[0], alvo))
    n = len(esperados)
    return MetricReport(
        feature=feature,
        label=rotulo.name,
        type=rotulo.type,
        n_labeled=n,
        n_predicted=len(pares),
        n_unavailable=indisponiveis,
        coverage=0.0 if n == 0 else len(pares) / n,
        scores=scores(rotulo.type, pares),
        confusion=confusion(pares) if pares and rotulo.type == "bool" else None,
        baseline_majority=(
            None if rotulo.type == "numeric" else majority([a for _, a in pares], esperados)
        ),
        baseline_majority_covered=(
            None if rotulo.type == "numeric" or not pares else majority_covered(pares)
        ),
        baseline_metadata=metrica.baseline_metadata if metrica is not None else None,
        mcnemar=None if rotulo.type == "numeric" or not pares else mcnemar(pares),
        threshold=_threshold(pares) if rotulo.type == "bool" else None,
        label_grain=(None if rotulo.grain is None else {"unit": "m", "step": rotulo.grain}),
        per_class=per_class(pares) if pares and rotulo.type != "numeric" else None,
        confusion_matrix=confusion_matrix(pares) if pares and rotulo.type == "enum" else None,
        bootstrap=(
            bootstrap_by_group(por_grupo, resamples, seed)
            if resamples > 0 and pares and rotulo.type != "numeric"
            else None
        ),
    )


def scores(tipo: str, pares: Sequence[tuple[object, object]]) -> dict[str, float]:
    if not pares:
        return {}
    if tipo == "numeric":
        erros = [abs(float(p) - float(a)) for p, a in pares]  # type: ignore[arg-type]
        quadrados = [e * e for e in erros]
        return {
            "mae": sum(erros) / len(erros),
            "rmse": (sum(quadrados) / len(quadrados)) ** 0.5,
        }
    acertos = sum(1 for p, a in pares if _as_bool(p) == a)
    return {"accuracy": acertos / len(pares), **_f1(pares)}


def _as_bool(valor: object) -> object:
    if isinstance(valor, bool):
        return valor
    if isinstance(valor, int):
        return bool(valor)
    return valor >= 0.5 if isinstance(valor, float) else valor


def _f1(pares: Sequence[tuple[object, object]]) -> dict[str, float]:
    por_classe = [_f1_da_classe(*_contagens(pares, classe)) for classe in _classes(pares)]
    return {"macro_f1": sum(por_classe) / len(por_classe) if por_classe else 0.0}


def per_class(pares: Sequence[tuple[object, object]]) -> dict[str, dict[str, float | int | None]]:
    """Metricas por classe com a mesma contagem do macro F1; divisao por zero sai `None`."""
    resultado: dict[str, dict[str, float | int | None]] = {}
    for classe in _classes(pares):
        tp, fp, fn = _contagens(pares, classe)
        resultado[classe] = {
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": _f1_da_classe(tp, fp, fn),
            "support": tp + fn,
            "predicted": tp + fp,
        }
    return resultado


def confusion_matrix(pares: Sequence[tuple[object, object]]) -> dict[str, dict[str, int]]:
    """Linha e o rotulo, coluna e a predicao, com toda classe vista nos dois eixos."""
    classes = _classes(pares)
    matriz = {verdade: dict.fromkeys(classes, 0) for verdade in classes}
    for p, a in pares:
        matriz[str(a)][str(_as_bool(p))] += 1
    return matriz


def bootstrap_by_group(
    por_grupo: Mapping[str, Sequence[tuple[object, object]]], resamples: int, seed: int
) -> dict[str, object]:
    """Intervalo de 95% de acuracia e macro F1 reamostrando partidas inteiras, e nao clipes."""
    grupos = sorted(por_grupo)
    sorteio = random.Random(seed)
    acuracias: list[float] = []
    macros: list[float] = []
    for _ in range(resamples):
        amostra = [par for g in sorteio.choices(grupos, k=len(grupos)) for par in por_grupo[g]]
        acuracias.append(sum(1 for p, a in amostra if _as_bool(p) == a) / len(amostra))
        macros.append(_f1(amostra)["macro_f1"])
    return {
        "accuracy": _faixa(acuracias),
        "macro_f1": _faixa(macros),
        "groups": len(grupos),
        "resamples": resamples,
        "level": 0.95,
    }


def _faixa(valores: list[float]) -> list[float]:
    ordenados = sorted(valores)
    return [
        ordenados[int(0.025 * (len(ordenados) - 1))],
        ordenados[int(0.975 * (len(ordenados) - 1))],
    ]


def _classes(pares: Sequence[tuple[object, object]]) -> list[str]:
    return sorted({str(a) for _, a in pares} | {str(_as_bool(p)) for p, _ in pares})


def _contagens(pares: Sequence[tuple[object, object]], classe: str) -> tuple[int, int, int]:
    tp = sum(1 for p, a in pares if str(_as_bool(p)) == classe and str(a) == classe)
    fp = sum(1 for p, a in pares if str(_as_bool(p)) == classe and str(a) != classe)
    fn = sum(1 for p, a in pares if str(_as_bool(p)) != classe and str(a) == classe)
    return tp, fp, fn


def _f1_da_classe(tp: int, fp: int, fn: int) -> float:
    return 0.0 if not tp else tp / (tp + 0.5 * (fp + fn))


def confusion(pares: Sequence[tuple[object, object]]) -> dict[str, int]:
    return {
        "tp": sum(1 for p, a in pares if _as_bool(p) is True and a is True),
        "fp": sum(1 for p, a in pares if _as_bool(p) is True and a is False),
        "fn": sum(1 for p, a in pares if _as_bool(p) is not True and a is True),
        "tn": sum(1 for p, a in pares if _as_bool(p) is not True and a is False),
    }


def majority(cobertos: Sequence[object], todos: Sequence[object]) -> dict[str, float]:
    """A maioria sai da populacao inteira, e o acerto e medido no subconjunto que o modelo cobriu.

    Medir a maioria so no coberto premiaria quem responde exatamente onde a classe e obvia, e
    medir o acerto na populacao inteira puniria quem declara indisponivel. Os dois numeros da
    linha precisam falar do mesmo `n`.
    """
    if not todos:
        return {"accuracy": 0.0, "macro_f1": 0.0, "n": 0}
    maioria, _ = Counter(str(a) for a in todos).most_common(1)[0]
    alvo = cobertos or todos
    pares = [(maioria, str(a)) for a in alvo]
    acertos = sum(1 for _, a in pares if str(a) == maioria)
    return {"accuracy": acertos / len(alvo), **_f1(pares), "n": len(alvo)}


def majority_covered(pares: Sequence[tuple[object, object]]) -> dict[str, float]:
    """A melhor constante calculada DENTRO do subconjunto coberto, e nao na populacao inteira.

    `majority` mede a maioria na populacao e a acuracia no coberto, o que e o par certo para
    julgar cobertura. Mas ele nao responde a outra pergunta, que e a que decide se a feature
    aprendeu alguma coisa: naquele subconjunto especifico, um constante teria ido melhor? Com
    cobertura de 60% num rotulo de maioria 0,52, a melhor constante do coberto pode chegar a 0,86,
    e um numero absoluto de 0,60 passaria por bom estando muito abaixo do trivial.
    """
    alvos = [a for _, a in pares]
    maioria, acertos = Counter(str(a) for a in alvos).most_common(1)[0]
    constante = [(maioria, str(a)) for a in alvos]
    return {"accuracy": acertos / len(alvos), **_f1(constante), "n": len(alvos)}


def mcnemar(pares: Sequence[tuple[object, object]]) -> dict[str, float]:
    """O teste pareado contra o constante, que e o que separa sinal de ruido neste conjunto.

    Diferenca de acuracia nao decide quando a maioria e 0,894 e ha 64 positivos em 604: um modelo
    util de `looked_before` tem acuracia 0,8825, ABAIXO do constante, e ainda assim vai de macro F1
    0,472 para 0,743. O pareado olha so os casos em que os dois discordam, que e onde a informacao
    esta.

    E o exato binomial, e nao a aproximacao qui-quadrado, porque com `b + c` na casa das dezenas a
    aproximacao erra justamente na faixa que interessa.
    """
    alvos = [a for _, a in pares]
    maioria = Counter(str(a) for a in alvos).most_common(1)[0][0]
    b = sum(1 for p, a in pares if str(_as_bool(p)) == str(a) and maioria != str(a))
    c = sum(1 for p, a in pares if str(_as_bool(p)) != str(a) and maioria == str(a))
    return {
        "modelo_acertou_constante_errou": b,
        "modelo_errou_constante_acertou": c,
        "p_value": _binomial_bilateral(b, c),
    }


def _binomial_bilateral(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    menor = min(b, c)
    cauda = sum(_combinacoes(n, i) for i in range(menor + 1)) / (2**n)
    return min(1.0, 2.0 * cauda)


def _combinacoes(n: int, k: int) -> int:
    resultado = 1
    for i in range(k):
        resultado = resultado * (n - i) // (i + 1)
    return resultado


def _threshold(pares: Sequence[tuple[object, object]]) -> dict[str, float] | None:
    numeros = [(p, a) for p, a in pares if isinstance(p, float)]
    if not numeros:
        return None
    melhor, onde = 0.0, 0.0
    for candidato in sorted({round(p, 3) for p, _ in numeros}):
        f1 = _f1([(1.0 if p >= candidato else 0.0, a) for p, a in numeros])["macro_f1"]
        if f1 > melhor:
            melhor, onde = f1, candidato
    return {"best_f1": melhor, "best_threshold": onde}
