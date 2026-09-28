"""Treino da softmax sobre a evidencia de uma feature, validado deixando uma dobra de fora."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from edge.features.evidence_model import VALOR, Amostra, Column, EvidenceModel, softmax
from mlops.evaluate import per_class, scores

L2_CANDIDATOS = (0.01, 0.1, 1.0, 10.0)
ITERACOES = 3000
MAX_CATEGORIAS = 12
"""Texto com mais valores distintos que isso e texto livre, e nao categoria."""

EXCLUIDAS = frozenset(
    {
        "frame_index",
        "frames",
        "frame_do_chute",
        "frame_utilizado",
        "quadro_de_referencia",
        "shooter_track_id",
        "goalkeeper_track_id",
        "marked_keys",
        "pairs",
        "leitura",
    }
)
"""Indice de quadro, id de trajetoria e texto livre: nada disso generaliza entre clipes."""


@dataclass(frozen=True)
class Exemplo:
    shot_id: str
    fold: int
    valor: object
    evidencia: Mapping[str, object]
    rotulo: str


def discover_columns(amostras: Sequence[Amostra]) -> tuple[Column, ...]:
    """Numero vira coluna numerica, texto com poucos valores vira uma coluna por valor."""
    vistos: dict[tuple[str, ...], list[object]] = {}
    for valor, evidencia in amostras:
        vistos.setdefault((VALOR,), []).append(valor)
        for caminho, bruto in _achata(evidencia):
            vistos.setdefault(caminho, []).append(bruto)
    colunas: list[Column] = []
    for caminho in sorted(vistos):
        presentes = [v for v in vistos[caminho] if v is not None]
        falta = len(presentes) < len(amostras)
        if presentes and all(isinstance(v, bool | int | float) for v in presentes):
            colunas.append(Column(caminho, "number"))
        elif presentes and all(isinstance(v, str) for v in presentes):
            distintos = sorted({str(v) for v in presentes})
            if len(distintos) > MAX_CATEGORIAS:
                continue
            colunas.extend(Column(caminho, "equals", v) for v in distintos)
        else:
            continue
        if falta:
            colunas.append(Column(caminho, "missing"))
    return tuple(colunas)


def fit(
    exemplos: Sequence[Exemplo], classes: Sequence[str], l2: float, feature: str = ""
) -> EvidenceModel:
    """Softmax com L2 e peso de classe balanceado, por gradiente com passo 1/L."""
    amostras = [(e.valor, e.evidencia) for e in exemplos]
    colunas = discover_columns(amostras)
    esboco = EvidenceModel(feature, tuple(classes), colunas, (), (), (), ())
    cru = esboco.matrix(amostras)
    media = cru.mean(axis=0)
    escala = cru.std(axis=0)
    escala[escala < 1e-9] = 1.0
    x = (cru - media) / escala
    y = np.array([classes.index(e.rotulo) for e in exemplos])
    contagem = np.bincount(y, minlength=len(classes)).astype(np.float64)
    peso = (len(y) / (len(classes) * np.maximum(contagem, 1.0)))[y]
    peso = peso / peso.sum()
    w, b = _ajusta(x, y, len(classes), l2, peso)
    return EvidenceModel(
        feature=feature,
        classes=tuple(classes),
        columns=colunas,
        mean=tuple(cast("list[float]", media.tolist())),
        scale=tuple(cast("list[float]", escala.tolist())),
        coef=tuple(tuple(linha) for linha in cast("list[list[float]]", w.tolist())),
        intercept=tuple(cast("list[float]", b.tolist())),
    )


def train(
    exemplos: Sequence[Exemplo], feature: str, l2s: Sequence[float] = L2_CANDIDATOS
) -> tuple[EvidenceModel, dict[str, object]]:
    """Escolhe o L2 pelo macro F1 deixando uma dobra de fora, e treina com todas no fim."""
    classes = sorted({e.rotulo for e in exemplos})
    dobras = sorted({e.fold for e in exemplos})
    if len(classes) < 2 or len(dobras) < 2:
        raise ValueError("o treino precisa de duas classes e duas dobras")
    por_l2: dict[float, list[tuple[object, object]]] = {}
    for l2 in l2s:
        previstos: dict[str, str] = {}
        for dobra in dobras:
            treino = [e for e in exemplos if e.fold != dobra]
            teste = [e for e in exemplos if e.fold == dobra]
            modelo = fit(treino, classes, l2, feature)
            previstos.update(
                zip(
                    (e.shot_id for e in teste),
                    modelo.predict([(e.valor, e.evidencia) for e in teste]),
                    strict=True,
                )
            )
        por_l2[l2] = [(previstos[e.shot_id], e.rotulo) for e in exemplos]
    melhor = max(l2s, key=lambda l2: (scores("enum", por_l2[l2])["macro_f1"], l2))
    relatorio = {
        "feature": feature,
        "n": len(exemplos),
        "folds": dobras,
        "classes": classes,
        "cv": {str(l2): scores("enum", por_l2[l2]) for l2 in l2s},
        "escolhido": {"l2": melhor, **_medidas(por_l2[melhor])},
        **_referencias(exemplos),
    }
    return fit(exemplos, classes, melhor, feature), relatorio


def evaluate_model(modelo: EvidenceModel, exemplos: Sequence[Exemplo]) -> dict[str, object]:
    """Uma passada do modelo congelado, ao lado da regra e da maioria nos mesmos clipes."""
    previstos = modelo.predict([(e.valor, e.evidencia) for e in exemplos])
    pares = [(p, e.rotulo) for p, e in zip(previstos, exemplos, strict=True)]
    return {"n": len(exemplos), "modelo": _medidas(pares), **_referencias(exemplos)}


def _referencias(exemplos: Sequence[Exemplo]) -> dict[str, object]:
    maioria = Counter(e.rotulo for e in exemplos).most_common(1)[0][0]
    return {
        "regra": _medidas([(str(e.valor), e.rotulo) for e in exemplos]),
        "maioria": {"classe": maioria, **scores("enum", [(maioria, e.rotulo) for e in exemplos])},
    }


def _medidas(pares: Sequence[tuple[object, object]]) -> dict[str, object]:
    return {**scores("enum", pares), "per_class": per_class(pares)}


def _ajusta(
    x: np.ndarray[Any, np.dtype[np.float64]],
    y: np.ndarray[Any, Any],
    k: int,
    l2: float,
    peso: np.ndarray[Any, np.dtype[np.float64]],
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]]:
    n, d = x.shape
    com_vies = np.hstack([x, np.ones((n, 1))])
    lipschitz = 0.5 * float(peso.max()) * float(np.linalg.norm(com_vies, 2)) ** 2 + l2 / n
    passo = 1.0 / lipschitz
    alvo = np.eye(k)[y]
    w = np.zeros((k, d))
    b = np.zeros(k)
    for _ in range(ITERACOES):
        erro = (softmax(x @ w.T + b) - alvo) * peso[:, None]
        w -= passo * (erro.T @ x + (l2 / n) * w)
        b -= passo * erro.sum(axis=0)
    return w, b


def _achata(
    evidencia: Mapping[str, object], prefixo: tuple[str, ...] = ()
) -> Iterator[tuple[tuple[str, ...], object]]:
    for chave, valor in evidencia.items():
        if chave in EXCLUIDAS:
            continue
        caminho = (*prefixo, chave)
        if isinstance(valor, Mapping):
            yield from _achata(cast("Mapping[str, object]", valor), caminho)
        elif not isinstance(valor, list | tuple):
            yield caminho, valor
