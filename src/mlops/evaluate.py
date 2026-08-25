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
    threshold: Mapping[str, float] | None = None
    label_grain: Mapping[str, object] | None = None

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
            "threshold": None if self.threshold is None else dict(self.threshold),
            "label_grain": None if self.label_grain is None else dict(self.label_grain),
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
) -> Report:
    declaradas = [nome for nome, _ in table.metrics]
    extras = sorted(
        {nome for por_tiro in predictions.values() for nome in por_tiro} - set(declaradas)
    )
    features = declaradas + extras
    relatorios = [_metric(nome, predictions, truth, table) for nome in features]
    return Report(population=population or f"n={len(truth)}", metrics=tuple(relatorios))


def _metric(
    feature: str,
    predictions: Mapping[str, Mapping[str, Predicted]],
    truth: Mapping[str, Mapping[str, str]],
    table: Labels,
) -> MetricReport:
    rotulo = table.label_for(feature)
    if rotulo is None:
        raise EvaluationError(
            f"a feature {feature!r} e orfa: nenhuma entrada [[metric]] em config/labels.toml "
            "diz qual coluna do parceiro ela mede"
        )
    pares: list[tuple[object, object]] = []
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
        threshold=_threshold(pares) if rotulo.type == "bool" else None,
        label_grain=(None if rotulo.grain is None else {"unit": "m", "step": rotulo.grain}),
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
    classes = sorted({str(a) for _, a in pares} | {str(_as_bool(p)) for p, _ in pares})
    por_classe: list[float] = []
    for classe in classes:
        tp = sum(1 for p, a in pares if str(_as_bool(p)) == classe and str(a) == classe)
        fp = sum(1 for p, a in pares if str(_as_bool(p)) == classe and str(a) != classe)
        fn = sum(1 for p, a in pares if str(_as_bool(p)) != classe and str(a) == classe)
        por_classe.append(0.0 if not tp else tp / (tp + 0.5 * (fp + fn)))
    return {"macro_f1": sum(por_classe) / len(por_classe) if por_classe else 0.0}


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
