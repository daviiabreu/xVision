"""Os numeros que o `eval` imprime sao os unicos numeros de acuracia do projeto.

Errados, eles nao param nada: o comando roda, imprime uma tabela bonita, e a decisao de aceitar
ou recusar um modelo sai de um valor que ninguem conferiu. Tres mutacoes passaram na suite sem
derrubar teste nenhum, e as tres mudam o numero sem mudar a forma da saida.
"""

from __future__ import annotations

import math

from mlops.evaluate import confusion, majority, scores


def test_the_rmse_is_a_root_and_not_the_mean_square() -> None:
    """Sem a raiz, o valor reportado como RMSE e o erro quadratico medio, sempre maior.

    Quem definir criterio de aceite em cima dele estara pedindo um modelo melhor do que pretende,
    e nada na saida diz que a unidade mudou.
    """
    pares: list[tuple[object, object]] = [(0.0, 3.0), (0.0, 4.0)]

    numeros = scores("numeric", pares)

    assert numeros["mae"] == 3.5
    assert math.isclose(numeros["rmse"], math.sqrt((9 + 16) / 2))
    assert numeros["rmse"] < (9 + 16) / 2, "o rmse veio sem a raiz"


def test_a_false_positive_is_not_reported_as_a_false_negative() -> None:
    """Trocadas, as duas celulas invertem o diagnostico: um modelo que inventa deteccao passa a
    parecer um modelo que deixa de detectar, e a correcao pedida e a oposta da necessaria.

    Nenhum teste usava caso assimetrico, entao `tp` e `tn` cobriam e as duas erradas passavam.
    """
    so_falso_positivo: list[tuple[object, object]] = [(True, False), (True, False), (True, True)]

    matriz = confusion(so_falso_positivo)

    assert matriz == {"tp": 1, "fp": 2, "fn": 0, "tn": 0}


def test_the_baseline_takes_the_majority_from_the_whole_population() -> None:
    """A maioria sai da populacao inteira e o acerto e medido no subconjunto coberto.

    Tirando a maioria so do coberto, um modelo que responde apenas onde a classe e obvia infla a
    propria linha de base e depois se compara favoravelmente contra ela. E exatamente o vies que
    o campo obrigatorio existe para expor.
    """
    todos = [True] * 3 + [False] * 7
    cobertos = [True, True]

    linha = majority(cobertos, todos)

    assert linha["n"] == 2, "o acerto tem de ser medido no coberto"
    assert linha["accuracy"] == 0.0, (
        "a maioria da populacao inteira e False, entao ela erra os dois cobertos. "
        "Acerto 1.0 aqui significa que a maioria saiu do proprio coberto"
    )
