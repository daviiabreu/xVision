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
    assert duracao == {
        "n": 2,
        "min": 5.0,
        "max": 100.0,
        "mean": 52.5,
        "median": 100.0,
        "p95": 5.0,
    }


def test_two_different_clips_do_not_write_over_each_other() -> None:
    """O carimbo separa uma execucao da outra. Fixo, o segundo clipe apaga o evento do primeiro."""
    from pathlib import Path

    assert nome_da_execucao(Path("a/primeiro.mp4")) != nome_da_execucao(Path("b/segundo.mp4"))


def test_o_percentil_de_duas_amostras_e_degenerado_e_isso_e_declarado() -> None:
    """Com n=2, `int(0,95 * 1)` e zero, entao o p95 e a amostra de baixo.

    Nao e defeito, e a mesma indexacao por posto que `mlops.instant.measure` usa, e manter as duas
    iguais vale mais do que fazer o caso degenerado parecer bonito: duas definicoes de percentil no
    mesmo repositorio dariam numeros diferentes para a mesma amostra.
    """
    resumo = resumo_dos_modulos(
        {"ball": [5.0, 100.0]},
        {"ball": ("yolo-ball", "0.1.0")},
        {},
    )

    duracao = resumo["ball"]["duration_ms"]
    assert isinstance(duracao, dict)
    assert duracao["p95"] == 5.0
    assert duracao["median"] == 100.0


def test_o_ajuste_da_especie_acompanha_o_tempo_dela() -> None:
    """Tempo sem o ajuste ao lado nao e reproduzivel: fp16 e fp32 chegam com a mesma cara."""
    resumo = resumo_dos_modulos(
        {"ball": [10.0]},
        {"ball": ("yolo-ball", "0.1.0")},
        {},
        {"ball": "quantize='fp16';imgsz=1600"},
    )

    assert resumo["ball"]["tuning"] == "quantize='fp16';imgsz=1600"


def test_especie_sem_ajuste_declarado_nao_inventa_um() -> None:
    resumo = resumo_dos_modulos({"shot": [1.0]}, {"shot": ("ball-break", "0.1.0")}, {})

    assert resumo["shot"]["tuning"] is None
