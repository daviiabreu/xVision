"""A casca `xvision`: ajuda e subcomandos, sem escrever parser."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from cli import app
from edge.cache.perception import PerceptionCache
from edge.modules.base import ModuleOutput

runner = CliRunner()


def test_the_shell_lists_the_commands_it_carries() -> None:
    resultado = runner.invoke(app, ["--help"])

    assert resultado.exit_code == 0
    assert "clip" in resultado.output
    assert "cache" in resultado.output
    assert "eval-field" in resultado.output
    assert "eval-ball-trajectory" in resultado.output


def test_eval_ball_closes_modules_already_loaded_when_the_next_load_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    events: list[str] = []

    class Module:
        def __init__(self, name: str, *, fails: bool = False) -> None:
            self.name = name
            self.fails = fails

        def load(self) -> None:
            events.append(f"load:{self.name}")
            if self.fails:
                raise RuntimeError("load failed")

        def close(self) -> None:
            events.append(f"close:{self.name}")

    config = tmp_path / "pipeline.toml"
    config.write_text("[frames.dense]\nstrategy = 'uniform'\nfps = 15\n", encoding="utf-8")
    players = Module("players")
    ball = Module("ball", fails=True)

    def annotations() -> tuple[object, ...]:
        return (object(),)

    def agreed(_: Any) -> dict[str, int]:
        return {"clip": 1}

    def parsed_config(_: Path) -> object:
        return object()

    def modules(_: object) -> dict[str, Module]:
        return {"players": players, "ball": ball}

    def framesets(_: Path) -> dict[str, object]:
        return {}

    monkeypatch.setattr("cli.read", annotations)
    monkeypatch.setattr("cli.consensus", agreed)
    monkeypatch.setattr(
        "cli.load_catalog",
        lambda: (SimpleNamespace(shot_id="clip", big_enough=True, fold=0),),
    )
    monkeypatch.setattr("cli.read_config", parsed_config)
    monkeypatch.setattr("cli.load_modules", modules)
    monkeypatch.setattr("cli.read_framesets", framesets)

    result = runner.invoke(app, ["eval-ball", "--config", str(config)])

    assert result.exit_code != 0
    assert events == ["load:players", "load:ball", "close:players"]


def test_clip_writes_the_canonical_fixture_where_it_was_told(tmp_path: Path) -> None:
    destino = tmp_path / "fundo" / "clipe.mp4"

    resultado = runner.invoke(app, ["clip", "--out", str(destino)])

    assert resultado.exit_code == 0
    assert destino.is_file()
    assert str(destino) in resultado.output


def test_cache_clear_reports_what_it_removed(tmp_path: Path) -> None:
    PerceptionCache(tmp_path).put("abc", ModuleOutput(kind="players", frames=(0,), records=((),)))

    resultado = runner.invoke(app, ["cache", "clear", "--root", str(tmp_path)])

    assert resultado.exit_code == 0
    assert "1" in resultado.output


def test_cache_clear_shows_a_readable_default_instead_of_the_absolute_path() -> None:
    """O caminho absoluto e truncado com reticencia na largura do help, entao nao ensina nada."""
    resultado = runner.invoke(app, ["cache", "clear", "--help"])

    assert resultado.exit_code == 0
    assert "var/cache/perception" in resultado.output


@pytest.mark.parametrize("comando", [["clip", "--help"], ["cache", "--help"]])
def test_every_subcommand_answers_help(comando: list[str]) -> None:
    assert runner.invoke(app, comando).exit_code == 0


def test_the_clip_default_shown_in_help_is_the_one_the_command_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """O `show_default` e texto solto e nao acompanha `out_dir`, entao um teste amarra os dois.

    Ele anuncia o padrao, e a variavel de ambiente e override declarado, por isso a comparacao
    limpa o ambiente antes: o invariante e que o help nao minta sobre o padrao.
    """
    from edge.paths import VAR_ENV, out_dir

    monkeypatch.delenv(VAR_ENV, raising=False)
    ajuda = runner.invoke(app, ["clip", "--help"])
    anunciado = next(linha for linha in ajuda.output.splitlines() if "synthetic.mp4" in linha)

    assert str(out_dir() / "synthetic.mp4") in anunciado.replace("│", "").strip()


def test_run_takes_a_clip_and_writes_an_event_that_validates(tmp_path: Path) -> None:
    """O criterio da fase: um clipe entra por linha de comando e um evento valido sai em disco."""
    import json

    from contracts.validation import validate_shot_event
    from edge.testing.make_clip import make_synthetic_clip

    clipe = make_synthetic_clip(tmp_path / "s.mp4")
    saida = tmp_path / "execucao"

    resultado = runner.invoke(app, ["run", "--clip", str(clipe), "--out", str(saida)])

    assert resultado.exit_code == 0, resultado.output
    (eventos,) = list(saida.rglob("events.jsonl"))
    for linha in eventos.read_text(encoding="utf-8").splitlines():
        validate_shot_event(json.loads(linha))


def test_run_prints_one_line_per_stage_so_a_failure_is_visible(tmp_path: Path) -> None:
    from edge.testing.make_clip import make_synthetic_clip

    clipe = make_synthetic_clip(tmp_path / "s.mp4")

    resultado = runner.invoke(app, ["run", "--clip", str(clipe), "--out", str(tmp_path / "e")])

    for especie in ("players", "ball", "field", "tracking", "shot"):
        assert especie in resultado.output


def test_run_on_a_clip_that_does_not_open_says_so_and_exits_nonzero(tmp_path: Path) -> None:
    quebrado = tmp_path / "truncado.mp4"
    quebrado.write_bytes(b"nao e video")

    resultado = runner.invoke(app, ["run", "--clip", str(quebrado), "--out", str(tmp_path / "e")])

    assert resultado.exit_code != 0
    assert "truncado.mp4" in resultado.output


def test_run_with_a_config_that_is_not_there_says_what_to_pass(tmp_path: Path) -> None:
    """Instalado como wheel, o padrao relativo nao existe fora do checkout, e o traceback nu
    nao ensina nada a quem esta na Jetson."""
    from edge.testing.make_clip import make_synthetic_clip

    clipe = make_synthetic_clip(tmp_path / "s.mp4")

    resultado = runner.invoke(
        app, ["run", "--clip", str(clipe), "--config", str(tmp_path / "sumiu.toml")]
    )

    assert resultado.exit_code != 0
    assert "sumiu.toml" in resultado.output
    assert "--config" in resultado.output


def test_run_twice_on_the_same_clip_does_not_leave_a_half_written_file(tmp_path: Path) -> None:
    import json

    from edge.testing.make_clip import make_synthetic_clip

    clipe = make_synthetic_clip(tmp_path / "s.mp4")
    saida = tmp_path / "e"
    runner.invoke(app, ["run", "--clip", str(clipe), "--out", str(saida)])
    runner.invoke(app, ["run", "--clip", str(clipe), "--out", str(saida)])

    (eventos,) = list(saida.rglob("events.jsonl"))
    linhas = eventos.read_text(encoding="utf-8").splitlines()
    assert len(linhas) == 1
    json.loads(linhas[0])


def test_run_leaves_no_partial_file_behind(tmp_path: Path) -> None:
    """A escrita e por arquivo temporario mais rename, entao nada parcial pode sobrar em disco."""
    from edge.testing.make_clip import make_synthetic_clip

    clipe = make_synthetic_clip(tmp_path / "s.mp4")
    saida = tmp_path / "e"

    runner.invoke(app, ["run", "--clip", str(clipe), "--out", str(saida)])

    assert list(saida.rglob("*.parcial")) == []
    assert [p.name for p in saida.rglob("*.jsonl")] == ["events.jsonl"]


def test_data_check_without_the_dump_says_where_to_put_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Quem clona nao tem os 3,5 GB e o dump e restrito: a mensagem tem que ensinar."""
    from edge.paths import DATA_ENV

    monkeypatch.setenv(DATA_ENV, str(tmp_path / "ausente"))

    resultado = runner.invoke(app, ["data", "check"])

    assert resultado.exit_code != 0
    assert DATA_ENV in resultado.output


def test_data_check_counts_what_the_dataset_has(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import csv

    from edge.paths import DATA_ENV

    db = tmp_path / "db"
    db.mkdir()
    with (db / "0005_match_shots_1390.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "id_match", "body_part", "meters", "is_goal"])
        w.writeheader()
        w.writerow(
            {"id": "10", "id_match": "900", "body_part": "CABECA", "meters": "5", "is_goal": "true"}
        )
    with (db / "0006_match_shots_xg_1390.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "id_match_shot",
                "marked",
                "looked_before",
                "occupied_area",
                "goalkeaper_position",
                "assist",
                "used",
                "type_of_play",
                "expected_goals",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "id": "1",
                "id_match_shot": "10",
                "marked": "true",
                "looked_before": "false",
                "occupied_area": "true",
                "goalkeaper_position": "false",
                "assist": "true",
                "used": "head",
                "type_of_play": "open play",
                "expected_goals": "0.4",
            }
        )
    corte = tmp_path / "cortes" / "900" / "900_004_Mexico_gol_10.mp4"
    corte.parent.mkdir(parents=True)
    corte.write_bytes(b"\0" * 5000)
    monkeypatch.setenv(DATA_ENV, str(tmp_path))

    resultado = runner.invoke(app, ["data", "check"])

    assert resultado.exit_code == 0, resultado.output
    assert "1" in resultado.output
    assert "fold" in resultado.output.lower()


def test_eval_on_a_run_prints_one_line_per_metric_with_the_majority_beside_it(
    tmp_path: Path,
) -> None:
    """Acuracia sozinha e mentira neste conjunto: tres metricas parecem quase resolvidas."""
    import json

    execucao = tmp_path / "run"
    execucao.mkdir()
    (execucao / "events.jsonl").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "shot_id": "10:5",
                "clip_id": "10",
                "match_id": "900",
                "shot_timestamp_s": 1.0,
                "features": [
                    {
                        "name": "marcado",
                        "version": "0.1.0",
                        "status": "ok",
                        "value": True,
                        "confidence": None,
                        "reason": None,
                        "evidence": {},
                    }
                ],
                "pipeline": {
                    "versions": {},
                    "timings_ms": {},
                    "host": "h",
                    "started_at": "2026-08-24T00:00:00Z",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    labels = tmp_path / "labels.toml"
    labels.write_text(
        '[[label]]\nname = "marked"\ntable = "t"\ncolumn = "marked"\ntype = "bool"\n'
        'domain = ["true", "false"]\nmissing = [""]\n'
        '[[metric]]\nfeature = "marcado"\nlabel = "marked"\n',
        encoding="utf-8",
    )
    truth = tmp_path / "truth.json"
    truth.write_text(json.dumps({"10": {"marked": "true"}}), encoding="utf-8")

    resultado = runner.invoke(
        app, ["eval", str(execucao), "--labels", str(labels), "--truth", str(truth)]
    )

    assert resultado.exit_code == 0, resultado.output
    assert "marcado" in resultado.output
    assert "maioria" in resultado.output.lower()
    assert (execucao / "report.json").is_file()


def test_eval_on_a_directory_without_events_says_so(tmp_path: Path) -> None:
    resultado = runner.invoke(app, ["eval", str(tmp_path)])

    assert resultado.exit_code != 0
    assert "events.jsonl" in resultado.output


@pytest.fixture
def conjunto_falso(tmp_path: Path) -> Path:
    """Conjunto sintetico com as colunas e o formato de nome do parceiro, e ids inventados."""
    import csv

    from edge.testing.make_clip import make_synthetic_clip

    db = tmp_path / "db"
    db.mkdir()
    tiros = [
        {
            "id": str(700 + i),
            "id_match": "900",
            "body_part": "CABECA",
            "meters": "5",
            "is_goal": "false",
        }
        for i in range(3)
    ]
    with (db / "0005_match_shots_1390.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(tiros[0]))
        w.writeheader()
        w.writerows(tiros)
    xg = [
        {
            "id": str(i),
            "id_match_shot": str(700 + i),
            "marked": "true",
            "looked_before": "false",
            "occupied_area": "true",
            "goalkeaper_position": "false",
            "assist": "true",
            "crowded": "3",
            "used": "head",
            "type_of_play": "open play",
            "expected_goals": "0.4",
        }
        for i in range(3)
    ]
    with (db / "0006_match_shots_xg_1390.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(xg[0]))
        w.writeheader()
        w.writerows(xg)
    cortes = tmp_path / "cortes" / "900"
    cortes.mkdir(parents=True)
    for i in range(3):
        make_synthetic_clip(cortes / f"900_00{i}_Mexico_off_{700 + i}.mp4")
    return tmp_path


def test_run_all_writes_one_event_per_clip_and_a_run_record(
    conjunto_falso: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from edge.paths import DATA_ENV

    monkeypatch.setenv(DATA_ENV, str(conjunto_falso))
    saida = tmp_path / "execucoes"

    resultado = runner.invoke(app, ["run", "--all", "--out", str(saida)])

    assert resultado.exit_code == 0, resultado.output
    (registro,) = list(saida.rglob("run.json"))
    dados = json.loads(registro.read_text(encoding="utf-8"))
    assert dados["dataset"]["clips"] == 3
    assert dados["config_sha256"]
    assert isinstance(dados["git_dirty"], bool)
    assert dados["host"] and dados["python"] and dados["platform"]
    eventos = (registro.parent / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(eventos) == 3
    assert {json.loads(linha)["clip_id"] for linha in eventos} == {"700", "701", "702"}


def test_run_all_summarises_every_stage_instead_of_keeping_the_last_clip(
    conjunto_falso: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guardar o ultimo faz etapa que falhou na metade dos clipes aparecer como `ok`."""
    import json

    from edge.paths import DATA_ENV

    monkeypatch.setenv(DATA_ENV, str(conjunto_falso))
    saida = tmp_path / "execucoes"

    runner.invoke(app, ["run", "--all", "--out", str(saida)])

    (registro,) = list(saida.rglob("run.json"))
    modulos = json.loads(registro.read_text(encoding="utf-8"))["modules"]
    assert modulos["players"]["duration_ms"]["n"] == 3
    assert set(modulos["players"]["duration_ms"]) == {
        "n",
        "min",
        "max",
        "mean",
        "median",
        "p95",
    }
    assert modulos["players"]["failed_clips"] == 0


def test_run_all_limits_how_many_clips_it_processes(
    conjunto_falso: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from edge.paths import DATA_ENV

    monkeypatch.setenv(DATA_ENV, str(conjunto_falso))
    saida = tmp_path / "execucoes"

    runner.invoke(app, ["run", "--all", "--limit", "2", "--out", str(saida)])

    (eventos,) = list(saida.rglob("events.jsonl"))
    assert len(eventos.read_text(encoding="utf-8").splitlines()) == 2


def test_run_all_and_eval_agree_on_the_population(
    conjunto_falso: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Populacoes diferentes entre o que rodou e o que e avaliado fazem clipe ser decodificado,
    inferido, gravado e descartado no join sem ninguem contar."""
    import json

    from edge.paths import DATA_ENV

    monkeypatch.setenv(DATA_ENV, str(conjunto_falso))
    saida = tmp_path / "execucoes"
    rodagem = runner.invoke(app, ["run", "--all", "--out", str(saida)])
    assert rodagem.exit_code == 0, rodagem.output
    (registro,) = list(saida.rglob("run.json"))

    resultado = runner.invoke(app, ["eval", str(registro.parent)])

    assert resultado.exit_code == 0, resultado.output
    relatorio = json.loads((registro.parent / "report.json").read_text(encoding="utf-8"))
    rodados = json.loads(registro.read_text(encoding="utf-8"))["dataset"]["population"]
    assert relatorio["population"] == rodados


def test_eval_counts_an_unavailable_feature_as_uncovered_and_not_as_a_miss(
    tmp_path: Path,
) -> None:
    """Sem o filtro, `unavailable` viraria predicao com valor None e contaria como erro."""
    import json

    execucao = tmp_path / "run"
    execucao.mkdir()
    (execucao / "events.jsonl").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "shot_id": "10:5",
                "clip_id": "10",
                "match_id": None,
                "shot_timestamp_s": 1.0,
                "features": [
                    {
                        "name": "marcado",
                        "version": "0.1.0",
                        "status": "unavailable",
                        "value": True,
                        "confidence": None,
                        "reason": "sem homografia",
                        "evidence": {},
                    }
                ],
                "pipeline": {
                    "versions": {},
                    "timings_ms": {},
                    "host": "h",
                    "started_at": "2026-08-24T00:00:00Z",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    labels = tmp_path / "labels.toml"
    labels.write_text(
        '[[label]]\nname = "marked"\ntable = "t"\ncolumn = "marked"\ntype = "bool"\n'
        'domain = ["true", "false"]\nmissing = [""]\n'
        '[[metric]]\nfeature = "marcado"\nlabel = "marked"\n',
        encoding="utf-8",
    )
    truth = tmp_path / "truth.json"
    truth.write_text(json.dumps({"10": {"marked": "true"}}), encoding="utf-8")

    runner.invoke(app, ["eval", str(execucao), "--labels", str(labels), "--truth", str(truth)])

    relatorio = json.loads((execucao / "report.json").read_text(encoding="utf-8"))
    (metrica,) = relatorio["metrics"]
    assert (metrica["n_predicted"], metrica["n_unavailable"]) == (0, 1)


def test_run_all_records_whether_the_tree_was_dirty(
    conjunto_falso: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sem isto, `git_dirty` cravado em False passava verde e a procedencia mentia."""
    import json

    from edge.paths import DATA_ENV
    from mlops.run import git_state

    monkeypatch.setenv(DATA_ENV, str(conjunto_falso))
    saida = tmp_path / "execucoes"

    runner.invoke(app, ["run", "--all", "--limit", "1", "--out", str(saida)])

    (registro,) = list(saida.rglob("run.json"))
    dados = json.loads(registro.read_text(encoding="utf-8"))
    sha, sujo = git_state()
    assert (dados["git_sha"], dados["git_dirty"]) == (sha, sujo)


def test_artifact_add_writes_a_manifest_that_describes_what_is_there(tmp_path: Path) -> None:
    peso = tmp_path / "pesos"
    peso.mkdir()
    (peso / "model.onnx").write_bytes(b"peso")
    (peso / "model.onnx.data").write_bytes(b"externo")
    manifestos = tmp_path / "models"

    resultado = runner.invoke(
        app,
        [
            "artifact",
            "add",
            "--kind",
            "players",
            "--name",
            "demo",
            "--version",
            "0.0.1",
            str(peso),
            "--entrypoint",
            "model.onnx",
            "--out",
            str(manifestos),
        ],
    )

    assert resultado.exit_code == 0, resultado.output
    from edge.artifacts import read_manifest

    lido = read_manifest(manifestos / "players" / "demo-0.0.1.toml")
    assert [f.path for f in lido.files] == ["model.onnx", "model.onnx.data"]
    assert lido.digest is not None


def test_artifact_add_defaults_to_the_versioned_manifest_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    peso = tmp_path / "peso"
    peso.mkdir()
    (peso / "model.onnx").write_bytes(b"peso")
    monkeypatch.chdir(tmp_path)

    resultado = runner.invoke(
        app,
        [
            "artifact",
            "add",
            "--kind",
            "players",
            "--name",
            "demo",
            "--version",
            "0.0.1",
            str(peso),
            "--entrypoint",
            "model.onnx",
        ],
    )

    esperado = tmp_path / "models" / "manifests" / "players" / "demo-0.0.1.toml"
    assert resultado.exit_code == 0, resultado.output
    assert esperado.is_file()


def test_artifact_get_refuses_a_weight_whose_hash_diverges(tmp_path: Path) -> None:
    """Peso adulterado com o mesmo nome e o caso que o digest existe para pegar."""
    peso = tmp_path / "pesos"
    peso.mkdir()
    (peso / "model.onnx").write_bytes(b"peso")
    manifestos = tmp_path / "models"
    runner.invoke(
        app,
        [
            "artifact",
            "add",
            "--kind",
            "players",
            "--name",
            "demo",
            "--version",
            "0.0.1",
            str(peso),
            "--entrypoint",
            "model.onnx",
            "--out",
            str(manifestos),
        ],
    )
    caminho = manifestos / "players" / "demo-0.0.1.toml"
    from edge.artifacts import read_manifest

    digest = read_manifest(caminho).digest
    assert digest is not None
    guardado = tmp_path / "var-models" / digest
    guardado.mkdir(parents=True)
    (guardado / "model.onnx").write_bytes(b"outro peso")

    resultado = runner.invoke(
        app, ["artifact", "get", str(caminho), "--root", str(tmp_path / "var-models")]
    )

    assert resultado.exit_code != 0
    assert "sha256" in resultado.output


def test_artifact_get_prints_the_entrypoint_when_the_hashes_match(tmp_path: Path) -> None:
    peso = tmp_path / "pesos"
    peso.mkdir()
    (peso / "model.onnx").write_bytes(b"peso")
    manifestos = tmp_path / "models"
    runner.invoke(
        app,
        [
            "artifact",
            "add",
            "--kind",
            "players",
            "--name",
            "demo",
            "--version",
            "0.0.1",
            str(peso),
            "--entrypoint",
            "model.onnx",
            "--out",
            str(manifestos),
        ],
    )
    caminho = manifestos / "players" / "demo-0.0.1.toml"
    from edge.artifacts import read_manifest

    raiz = tmp_path / "var-models"
    digest = read_manifest(caminho).digest
    assert digest is not None
    guardado = raiz / digest
    guardado.mkdir(parents=True)
    (guardado / "model.onnx").write_bytes(b"peso")

    resultado = runner.invoke(app, ["artifact", "get", str(caminho), "--root", str(raiz)])

    assert resultado.exit_code == 0, resultado.output
    assert "model.onnx" in resultado.output


def test_artifact_add_writes_a_parseable_manifest_for_hostile_file_names(tmp_path: Path) -> None:
    """f-string sem escape gerava TOML invalido, ou pior: `\\b` virava backspace calado."""
    from edge.artifacts import read_manifest

    peso = tmp_path / "pesos"
    peso.mkdir()
    (peso / 'tem"aspas.bin').write_bytes(b"a")
    (peso / "tem\\barra.bin").write_bytes(b"b")
    (peso / "nome com espaco.bin").write_bytes(b"c")
    manifestos = tmp_path / "models"

    runner.invoke(
        app,
        [
            "artifact",
            "add",
            "--kind",
            "players",
            "--name",
            "d",
            "--version",
            "1",
            str(peso),
            "--entrypoint",
            "nome com espaco.bin",
            "--out",
            str(manifestos),
        ],
    )

    lido = read_manifest(manifestos / "players" / "d-1.toml")
    assert set(f.path for f in lido.files) == {
        'tem"aspas.bin',
        "tem\\barra.bin",
        "nome com espaco.bin",
    }


def test_artifact_add_ignores_the_file_manager_droppings(tmp_path: Path) -> None:
    """Num time de seis pessoas em Mac, `.DS_Store` mudava o digest do mesmo peso."""
    from edge.artifacts import read_manifest

    peso = tmp_path / "pesos"
    peso.mkdir()
    (peso / "model.onnx").write_bytes(b"peso")
    manifestos = tmp_path / "models"
    runner.invoke(
        app,
        [
            "artifact",
            "add",
            "--kind",
            "players",
            "--name",
            "d",
            "--version",
            "1",
            str(peso),
            "--entrypoint",
            "model.onnx",
            "--out",
            str(manifestos),
        ],
    )
    limpo = read_manifest(manifestos / "players" / "d-1.toml").digest

    (peso / ".DS_Store").write_bytes(b"lixo do finder")
    (peso / "__pycache__").mkdir()
    (peso / "__pycache__" / "x.pyc").write_bytes(b"lixo do python")
    runner.invoke(
        app,
        [
            "artifact",
            "add",
            "--kind",
            "players",
            "--name",
            "d",
            "--version",
            "1",
            str(peso),
            "--entrypoint",
            "model.onnx",
            "--out",
            str(manifestos),
        ],
    )

    assert read_manifest(manifestos / "players" / "d-1.toml").digest == limpo


def test_artifact_add_refuses_an_entrypoint_that_is_not_there(tmp_path: Path) -> None:
    peso = tmp_path / "pesos"
    peso.mkdir()
    (peso / "model.onnx").write_bytes(b"peso")

    resultado = runner.invoke(
        app,
        [
            "artifact",
            "add",
            "--kind",
            "players",
            "--name",
            "d",
            "--version",
            "1",
            str(peso),
            "--entrypoint",
            "outro.onnx",
            "--out",
            str(tmp_path / "models"),
        ],
    )

    assert resultado.exit_code != 0
    assert "outro.onnx" in resultado.output
