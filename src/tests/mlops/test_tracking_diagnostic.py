"""O diagnostico de fragmentacao do tracking, sobre runs pequenos escritos no teste.

O que se cobra e que as contas do diagnostico do handoff continuem as mesmas depois de sair do
script, e que a troca e a costura sejam medidas pela caixa contra o run de referencia, porque o run
costurado tem IDs diferentes dos que o agente rotulou.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from cli import app
from mlops.tracking_diagnostic import diagnosticar, ler_rotulos, ler_run, retomadas


def _caixa(x: float) -> dict[str, float]:
    return {"x1": x, "y1": 100.0, "x2": x + 30.0, "y2": 180.0}


def _run(pasta: Path, quadros: list[list[tuple[float, int | None]]]) -> Path:
    clipe = pasta / "clipe_4995949"
    clipe.mkdir(parents=True)
    documento = {
        "clip": {"id": "4995949"},
        "frames": [
            {
                "frame_index": 4 * indice,
                "players": [
                    {"box": _caixa(x), "track_id": track_id, "role": "player"}
                    for x, track_id in jogadores
                ],
            }
            for indice, jogadores in enumerate(quadros)
        ],
    }
    (clipe / "detections.json").write_text(json.dumps(documento), encoding="utf-8")
    return pasta


def _rotulos(caminho: Path, rotulo_do_par: str) -> Path:
    documento = {
        "tracks": [
            {"clip": "4995949", "track_id": 1, "crops": [{"label": "V"}]},
            {"clip": "4995949", "track_id": 2, "crops": [{"label": "V"}]},
            {"clip": "4995949", "track_id": 3, "crops": [{"label": "A"}]},
        ],
        "pairs": [{"clip": "4995949", "a": 1, "b": 2, "label": rotulo_do_par}],
    }
    caminho.write_text(json.dumps(documento), encoding="utf-8")
    return caminho


def test_trajetoria_que_nasce_perto_do_fim_de_outra_conta_como_retomada(tmp_path: Path) -> None:
    run = _run(tmp_path, [[(100, 1)], [(102, 1)], [(104, 2)], [(106, 2)]])

    (clipe,) = ler_run(run)

    assert retomadas(clipe.trajetorias) == ((1, 2),)


def test_ids_curtas_e_sem_id_saem_por_clipe(tmp_path: Path) -> None:
    run = _run(tmp_path, [[(100, 1), (500, None)], [(102, 1), (500, 3)], [(104, 1), (900, 4)]])

    (clipe,) = ler_run(run)
    linha = diagnosticar(clipe)

    assert (linha["ids"], linha["pico"], linha["curtas"], linha["sem_id"]) == (3, 2, 2, 1)


def test_costura_e_julgada_pela_caixa_contra_a_referencia(tmp_path: Path) -> None:
    """O run medido chama tudo de 7, e a caixa diz que ele uniu as originais 1 e 2."""
    referencia = _run(tmp_path / "ref", [[(100, 1)], [(102, 1)], [(104, 2)], [(106, 2)]])
    costurado = _run(tmp_path / "novo", [[(100, 7)], [(102, 7)], [(104, 7)], [(106, 7)]])
    rotulos = ler_rotulos(_rotulos(tmp_path / "rotulos.json", "S"), referencia)

    (clipe,) = ler_run(costurado)
    linha = diagnosticar(clipe, rotulos)

    assert (linha["costuras_S"], linha["costuras_D"], linha["trocas"]) == (1, 0, 0)


def test_costura_que_junta_cores_diferentes_vira_troca(tmp_path: Path) -> None:
    referencia = _run(tmp_path / "ref", [[(100, 1)], [(102, 3)]])
    costurado = _run(tmp_path / "novo", [[(100, 7)], [(102, 7)]])
    rotulos = ler_rotulos(_rotulos(tmp_path / "rotulos.json", "S"), referencia)

    (clipe,) = ler_run(costurado)
    linha = diagnosticar(clipe, rotulos)

    assert (linha["trocas"], linha["costuras_sem"]) == (1, 1)


def test_o_comando_funciona_sem_rotulos(tmp_path: Path) -> None:
    run = _run(tmp_path / "run", [[(100, 1)], [(102, 1)]])

    resultado = CliRunner().invoke(
        app, ["players", "diagnose", str(run), "--rotulos", str(tmp_path / "nao-existe.json")]
    )

    assert resultado.exit_code == 0, resultado.output
    assert "IDs por pessoa visivel: 1.00" in resultado.output
    assert "trocas" not in resultado.output


def test_o_comando_imprime_a_precisao_da_costura_com_rotulos(tmp_path: Path) -> None:
    referencia = _run(tmp_path / "ref", [[(100, 1)], [(102, 1)], [(104, 2)]])
    costurado = _run(tmp_path / "novo", [[(100, 7)], [(102, 7)], [(104, 7)]])
    rotulos = _rotulos(tmp_path / "rotulos.json", "D")

    resultado = CliRunner().invoke(
        app,
        [
            "players",
            "diagnose",
            str(costurado),
            "--rotulos",
            str(rotulos),
            "--referencia",
            str(referencia),
        ],
    )

    assert resultado.exit_code == 0, resultado.output
    assert "precisao da costura, rotulo de agente: 0.00" in resultado.output
