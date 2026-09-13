"""A concordância entre anotadores do instante, a barra de erro do próprio gabarito."""

from __future__ import annotations

import pytest

from mlops.concordancia import ConcordanciaError, Par, discordancias, medir, pares
from mlops.instant import Anotacao


def anot(clip: str, frame: int, annotator: str) -> Anotacao:
    return Anotacao(clip_id=clip, frame_index=frame, annotator=annotator)


def test_uma_anotacao_nao_gera_par() -> None:
    assert pares([anot("c1", 10, "davi")]) == ()


def test_um_par_por_clipe_com_dois_anotadores() -> None:
    resultado = pares([anot("c1", 10, "davi"), anot("c1", 13, "gabriel")])
    assert len(resultado) == 1
    assert resultado[0].delta == 3


def test_correcao_da_mesma_pessoa_nao_vira_par() -> None:
    resultado = pares([anot("c1", 10, "davi"), anot("c1", 40, "davi")])
    assert resultado == ()


def test_tres_anotadores_geram_tres_pares() -> None:
    resultado = pares([anot("c1", 10, "x"), anot("c1", 12, "y"), anot("c1", 20, "z")])
    assert len(resultado) == 3
    assert sorted(par.delta for par in resultado) == [2, 8, 10]


def test_par_ordenado_pelo_nome_do_anotador() -> None:
    (par,) = pares([anot("c1", 10, "zeca"), anot("c1", 13, "ana")])
    assert isinstance(par, Par)
    assert par.a == "ana"
    assert par.b == "zeca"
    assert par.frame_a == 13
    assert par.frame_b == 10
    assert par.delta == 3


def test_sem_clipe_repetido_nao_ha_concordancia() -> None:
    assert medir([anot("c1", 10, "davi"), anot("c2", 20, "gabriel")]) is None


def test_medir_agrega_mediana_tolerancia_e_limite() -> None:
    resultado = medir(
        [
            anot("c1", 100, "davi"),
            anot("c1", 105, "gabriel"),
            anot("c2", 200, "davi"),
            anot("c2", 250, "gabriel"),
            anot("c3", 300, "davi"),
            anot("c3", 500, "gabriel"),
        ]
    )
    assert resultado is not None
    assert resultado.n_clipes == 3
    assert resultado.n_pares == 3
    assert resultado.mediana_delta == 50.0
    assert resultado.max_delta == 200.0
    assert resultado.dentro_da_tolerancia == 1 / 3
    assert resultado.dentro_do_limite == 2 / 3


def test_media_dos_pares_num_clipe_com_tres_anotadores() -> None:
    resultado = medir([anot("c1", 10, "x"), anot("c1", 20, "y"), anot("c1", 34, "z")])
    assert resultado is not None
    assert resultado.n_clipes == 1
    assert resultado.n_pares == 3
    assert resultado.media_delta == (10 + 24 + 14) / 3


def test_conversao_para_segundos_usa_fps() -> None:
    resultado = medir([anot("c1", 100, "davi"), anot("c1", 160, "gabriel")], fps=60.0)
    assert resultado is not None
    assert resultado.mediana_delta == 60.0
    assert resultado.mediana_delta_s == 1.0


def test_sem_fps_nao_ha_segundos() -> None:
    resultado = medir([anot("c1", 100, "davi"), anot("c1", 130, "gabriel")])
    assert resultado is not None
    assert resultado.mediana_delta_s is None


def test_fps_nao_positivo_recusa() -> None:
    with pytest.raises(ConcordanciaError):
        medir([anot("c1", 1, "x"), anot("c1", 2, "y")], fps=0.0)


def test_discordancias_ordena_do_pior_ao_melhor() -> None:
    entrada = [
        anot("c1", 100, "davi"),
        anot("c1", 105, "gabriel"),
        anot("c2", 200, "davi"),
        anot("c2", 260, "gabriel"),
        anot("c3", 300, "davi"),
        anot("c3", 340, "gabriel"),
    ]
    fora = discordancias(entrada)
    assert [par.clip_id for par in fora] == ["c2", "c3"]
    assert fora[0].delta == 60


def test_discordancias_respeita_tolerancia_customizada() -> None:
    entrada = [anot("c1", 100, "davi"), anot("c1", 105, "gabriel")]
    assert discordancias(entrada, tolerancia=3)[0].delta == 5
    assert discordancias(entrada, tolerancia=10) == ()


def test_discordancias_tolerancia_negativa_recusa() -> None:
    with pytest.raises(ConcordanciaError):
        discordancias([anot("c1", 1, "x"), anot("c1", 2, "y")], tolerancia=-1)


def test_to_dict_carrega_as_tolerancias() -> None:
    resultado = medir([anot("c1", 100, "davi"), anot("c1", 130, "gabriel")])
    assert resultado is not None
    documento = resultado.to_dict()
    assert documento["tolerancia_quadros"] == 30
    assert documento["limite_quadros"] == 90
    assert documento["n_pares"] == 1
    assert documento["mediana_delta_segundos"] is None
