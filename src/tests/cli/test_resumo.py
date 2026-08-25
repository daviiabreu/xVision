"""O resumo de uma execucao em lote e o que diz se ela prestou, e ele so serve se agregar.

O `--all` roda centenas de clipes. O resumo e a unica coisa que alguem vai ler, e um teste de
mutacao mostrou que dava para crava-lo em "tudo certo" sem derrubar nada: o teste que carregava o
nome da promessa rodava um cenario onde nada falhava, entao ele nao conseguia distinguir "agrega
as falhas" de "sempre reporta zero".
"""

from __future__ import annotations

from cli import nome_da_execucao, resumo_dos_modulos


def test_a_stage_that_failed_in_some_clips_is_not_reported_as_ok() -> None:
    """Guardar so o ultimo faz uma execucao em que metade dos clipes falhou aparecer como `ok`.

    E o que o docstring de `resumo_dos_modulos` diz, e o que nenhum teste exercia.
    """
    resumo = resumo_dos_modulos(
        tempos={"players": [10.0, 20.0, 30.0]},
        identidade={"players": ("stub-players", "0.1.0")},
        falhas={"players": (2, "RuntimeError: o peso nao carregou")},
    )

    assert resumo["players"]["status"] == "failed"
    assert resumo["players"]["failed_clips"] == 2
    assert resumo["players"]["reason"] == "RuntimeError: o peso nao carregou"


def test_the_time_of_a_batch_keeps_its_spread_and_not_only_the_average() -> None:
    """O numero de manchete do projeto e ms por clipe, e a media sozinha esconde o pior caso.

    O teste que existia afirmava as chaves do dicionario, e nao os valores, entao trocar minimo e
    maximo pela media passava.
    """
    resumo = resumo_dos_modulos(
        tempos={"ball": [5.0, 100.0]},
        identidade={"ball": ("stub-ball", "0.1.0")},
        falhas={},
    )

    duracao = resumo["ball"]["duration_ms"]
    assert isinstance(duracao, dict)
    assert duracao == {"n": 2, "min": 5.0, "max": 100.0, "mean": 52.5}


def test_two_different_clips_do_not_write_over_each_other() -> None:
    """O carimbo separa uma execucao da outra. Fixo, o segundo clipe apaga o evento do primeiro."""
    from pathlib import Path

    assert nome_da_execucao(Path("a/primeiro.mp4")) != nome_da_execucao(Path("b/segundo.mp4"))
