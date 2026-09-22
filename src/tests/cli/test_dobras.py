"""O filtro de dobra dos comandos que abrem clipe anotado.

Ele existe porque `run --gabarito` abre os 38 clipes anotados e 14 deles sao o pacote reservado.
Sem o filtro, medir uma melhoria durante o desenvolvimento consome o holdout sem ninguem decidir
isso, e o consumo nao aparece em lugar nenhum: a execucao termina com sucesso.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

import pytest
from typer import Exit
from typer.testing import CliRunner

import cli
from cli import DOBRAS_DA_BOLA, app, dobras_de

runner = CliRunner()


def _consenso(valor: dict[str, int]) -> Any:
    def consensus(_: Any) -> dict[str, int]:
        return valor

    return consensus


def _nada(_: Any) -> None:
    return None


def _trinta(_: Any) -> float:
    return 30.0


def test_o_vocabulario_de_dobra_e_um_so() -> None:
    assert dobras_de("ajuste") == frozenset({0, 1, 2})
    assert dobras_de("reservado") == frozenset({3, 4})
    assert dobras_de("tudo") == frozenset({0, 1, 2, 3, 4})


def test_so_um_lugar_traduz_o_nome_da_dobra() -> None:
    """Meta-teste: a traducao mora em `dobras_de` e em nenhum outro lugar.

    Ela ja esteve copiada em quatro comandos. Copia nao e so duplicacao: `dobras !=
    DOBRAS_DA_BOLA["tudo"]` compara conjuntos e nao nomes, entao um nome novo cujo conjunto e o
    completo viraria filtro que nao filtra, calado, em tres dos quatro lugares.
    """
    fonte = Path(cli.__file__).read_text(encoding="utf-8")

    assert fonte.count("DOBRAS_DA_BOLA.get(") == 1, (
        "a traducao do nome da dobra voltou a ser copiada; ela pertence a `dobras_de`"
    )


def test_ajuste_e_reservado_nao_se_encostam() -> None:
    """Os dois conjuntos particionam o total, entao nenhum clipe cai nos dois nem fora dos dois.

    O agrupamento por partida, que e o que garante que um clipe de ajuste e um reservado nao vem do
    mesmo gramado, e propriedade de `mlops.split` e tem teste proprio la.
    """
    assert not dobras_de("ajuste") & dobras_de("reservado")
    assert dobras_de("ajuste") | dobras_de("reservado") == dobras_de("tudo")


def test_dobra_desconhecida_lista_o_que_existe() -> None:
    resultado = runner.invoke(app, ["run", "--all", "--dobra", "desenvolvimento"])

    assert resultado.exit_code == 1
    assert "nao existe" in resultado.output
    for nome in DOBRAS_DA_BOLA:
        assert nome in resultado.output


def test_eval_instant_recusa_dobra_desconhecida_antes_de_abrir_o_gabarito(
    tmp_path: Path,
) -> None:
    """A recusa vem antes de ler a anotacao: erro de digitacao nao pode virar erro de dado
    faltando, que manda a pessoa procurar defeito no lugar errado."""
    execucao = tmp_path / "run"
    execucao.mkdir()
    (execucao / "events.jsonl").write_text("", encoding="utf-8")

    resultado = runner.invoke(app, ["eval-instant", str(execucao), "--dobra", "dev"])

    assert resultado.exit_code == 1
    assert "dobra 'dev' nao existe" in resultado.output


def test_eval_instant_filtra_o_gabarito_pela_dobra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Um `run` que contenha clipe reservado ainda pode ser lido so na parte de desenvolvimento."""
    import cli

    execucao = tmp_path / "run"
    execucao.mkdir()
    linhas = [
        {"clip_id": "dev", "shot_timestamp_s": 1.0},
        {"clip_id": "res", "shot_timestamp_s": 1.0},
    ]
    (execucao / "events.jsonl").write_text(
        "".join(json.dumps(linha) + "\n" for linha in linhas), encoding="utf-8"
    )

    class Registro:
        def __init__(self, shot_id: str, fold: int) -> None:
            self.shot_id = shot_id
            self.fold = fold
            self.big_enough = True

    monkeypatch.setattr(cli, "read", lambda: ())
    monkeypatch.setattr(cli, "consensus", _consenso({"dev": 30, "res": 30}))
    monkeypatch.setattr(cli, "agreement", _nada)
    monkeypatch.setattr(cli, "load_catalog", lambda: [Registro("dev", 0), Registro("res", 4)])
    monkeypatch.setattr(cli, "_fps_do_clipe", _trinta)

    vistos: list[dict[str, int]] = []

    def espia(previsto: dict[str, int], gabarito: dict[str, int], **_: Any) -> Any:
        vistos.append(dict(gabarito))
        raise SystemExit(0)

    monkeypatch.setattr(cli, "measure", espia)
    runner.invoke(app, ["eval-instant", str(execucao), "--dobra", "ajuste"])

    assert vistos == [{"dev": 30}]


def test_eval_instant_avisa_quando_a_dobra_fica_vazia(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cli

    execucao = tmp_path / "run"
    execucao.mkdir()
    (execucao / "events.jsonl").write_text("", encoding="utf-8")

    class Registro:
        def __init__(self) -> None:
            self.shot_id = "res"
            self.fold = 4
            self.big_enough = True

    monkeypatch.setattr(cli, "read", lambda: ())
    monkeypatch.setattr(cli, "consensus", _consenso({"res": 30}))
    monkeypatch.setattr(cli, "load_catalog", lambda: [Registro()])

    resultado = runner.invoke(app, ["eval-instant", str(execucao), "--dobra", "ajuste"])

    assert resultado.exit_code == 1
    assert "nenhum clipe anotado na dobra ajuste" in resultado.output


def _primeiro_json(saida: str) -> dict[str, Any]:
    """O primeiro objeto da saida, porque `eval-instant` imprime dois: medida e taxonomia.

    Cortar no ultimo `}` juntava os dois num texto que nao e JSON, e o teste morria em
    `JSONDecodeError` em vez de comparar o que queria comparar.
    """
    objeto, _ = json.JSONDecoder().raw_decode(saida[saida.index("{") :])
    return cast("dict[str, Any]", objeto)


def test_eval_instant_reporta_a_procedencia_quando_a_execucao_a_gravou(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """O defeito replantado: as 38 medidas saiam como `desconhecida` enquanto o `run.json` da
    mesma execucao sabia que 37 vinham de `ball-break` e uma de `midpoint`."""
    import cli
    from mlops.instant import OBSERVATIONS_FILE

    execucao = tmp_path / "run"
    execucao.mkdir()
    (execucao / "events.jsonl").write_text(
        json.dumps({"clip_id": "a", "shot_timestamp_s": 1.0})
        + "\n"
        + json.dumps({"clip_id": "b", "shot_timestamp_s": 1.0})
        + "\n",
        encoding="utf-8",
    )
    (execucao / OBSERVATIONS_FILE).write_text(
        json.dumps({"clip_id": "a", "shot_source": "ball-break"})
        + "\n"
        + json.dumps({"clip_id": "b", "shot_source": "midpoint"})
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "read", lambda: ())
    monkeypatch.setattr(cli, "consensus", _consenso({"a": 30, "b": 30}))
    monkeypatch.setattr(cli, "agreement", _nada)
    monkeypatch.setattr(cli, "_fps_do_clipe", _trinta)
    monkeypatch.setattr(cli, "_comprimentos", _sem_comprimentos)

    resultado = runner.invoke(app, ["eval-instant", str(execucao)])

    assert resultado.exit_code == 0
    assert _primeiro_json(resultado.output)["fontes"] == {"ball-break": 1, "midpoint": 1}


def test_eval_instant_de_execucao_antiga_continua_dizendo_desconhecida(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Execucao sem o arquivo nao pode virar `ball-break` por otimismo: nao saber e um estado."""
    import cli

    execucao = tmp_path / "run"
    execucao.mkdir()
    (execucao / "events.jsonl").write_text(
        json.dumps({"clip_id": "a", "shot_timestamp_s": 1.0}) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(cli, "read", lambda: ())
    monkeypatch.setattr(cli, "consensus", _consenso({"a": 30}))
    monkeypatch.setattr(cli, "agreement", _nada)
    monkeypatch.setattr(cli, "_fps_do_clipe", _trinta)
    monkeypatch.setattr(cli, "_comprimentos", _sem_comprimentos)

    resultado = runner.invoke(app, ["eval-instant", str(execucao)])

    assert _primeiro_json(resultado.output)["fontes"] == {"desconhecida": 1}


def _todos_os_json(saida: str) -> list[dict[str, Any]]:
    """Os objetos que `eval-instant` imprime, em ordem: medida e depois taxonomia."""
    decoder = json.JSONDecoder()
    objetos: list[dict[str, Any]] = []
    resto = saida[saida.index("{") :]
    while resto.lstrip().startswith("{"):
        objeto, fim = decoder.raw_decode(resto.lstrip())
        objetos.append(cast("dict[str, Any]", objeto))
        resto = resto.lstrip()[fim:]
        if "{" not in resto:
            break
        resto = resto[resto.index("{") :]
    return objetos


def test_eval_instant_nomeia_a_causa_do_erro_a_partir_dos_candidatos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sessenta quadros de erro pedem trabalho diferente conforme a quebra certa tenha existido.

    O defeito replantado nao levanta nada: o relatorio para na mediana, e quem le decide o
    refinamento sem saber se ha o que reordenar.
    """
    import cli
    from mlops.instant import OBSERVATIONS_FILE

    execucao = tmp_path / "run"
    execucao.mkdir()
    (execucao / "events.jsonl").write_text(
        json.dumps({"clip_id": "a", "shot_timestamp_s": 10.0})
        + "\n"
        + json.dumps({"clip_id": "b", "shot_timestamp_s": 10.0})
        + "\n",
        encoding="utf-8",
    )
    (execucao / OBSERVATIONS_FILE).write_text(
        json.dumps(
            {
                "clip_id": "a",
                "shot_source": "ball-break",
                "candidates": [{"frame_index": 300}, {"frame_index": 30}],
            }
        )
        + "\n"
        + json.dumps(
            {
                "clip_id": "b",
                "shot_source": "ball-break",
                "candidates": [{"frame_index": 300}, {"frame_index": 290}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "read", lambda: ())
    monkeypatch.setattr(cli, "consensus", _consenso({"a": 30, "b": 30}))
    monkeypatch.setattr(cli, "agreement", _nada)
    monkeypatch.setattr(cli, "_fps_do_clipe", _trinta)
    monkeypatch.setattr(cli, "_comprimentos", _sem_comprimentos)

    resultado = runner.invoke(app, ["eval-instant", str(execucao)])

    assert resultado.exit_code == 0
    _, _, causas = _todos_os_json(resultado.output)
    assert causas["tipos"] == {"escolha": 1, "trajetoria": 1}
    assert causas["cobertura_das_causas"] == 1.0
    assert causas["recall_em_k"]["top1"] == 0.0
    assert causas["recall_em_k"]["top2"] == 0.5


def test_eval_instant_de_execucao_sem_candidatos_declara_que_nao_sabe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Execucao anterior ao campo nao pode produzir taxonomia apontando para a deteccao da bola."""
    import cli
    from mlops.instant import OBSERVATIONS_FILE

    execucao = tmp_path / "run"
    execucao.mkdir()
    (execucao / "events.jsonl").write_text(
        json.dumps({"clip_id": "a", "shot_timestamp_s": 10.0}) + "\n", encoding="utf-8"
    )
    (execucao / OBSERVATIONS_FILE).write_text(
        json.dumps({"clip_id": "a", "shot_source": "ball-break"}) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(cli, "read", lambda: ())
    monkeypatch.setattr(cli, "consensus", _consenso({"a": 30}))
    monkeypatch.setattr(cli, "agreement", _nada)
    monkeypatch.setattr(cli, "_fps_do_clipe", _trinta)
    monkeypatch.setattr(cli, "_comprimentos", _sem_comprimentos)

    resultado = runner.invoke(app, ["eval-instant", str(execucao)])

    _, _, causas = _todos_os_json(resultado.output)
    assert causas["tipos"] == {"nao-declarado": 1}
    assert causas["cobertura_das_causas"] == 0.0


@pytest.mark.parametrize("clips", [["a"], ["a", "b", "c"], ["a", "a", "b"]])
def test_eval_instant_rejects_mismatched_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clips: list[str]
) -> None:
    (tmp_path / "events.jsonl").write_text(
        "".join(json.dumps({"clip_id": c, "shot_timestamp_s": 1.0}) + "\n" for c in ["a", "b"])
    )
    (tmp_path / "instant-observations.jsonl").write_text(
        "".join(json.dumps({"clip_id": c, "shot_source": "ball-break"}) + "\n" for c in clips)
    )
    monkeypatch.setattr(cli, "read", lambda: ())
    monkeypatch.setattr(cli, "consensus", _consenso({"a": 30, "b": 30}))
    monkeypatch.setattr(cli, "agreement", _nada)
    monkeypatch.setattr(cli, "_fps_do_clipe", _trinta)

    resultado = runner.invoke(app, ["eval-instant", str(tmp_path)])

    assert resultado.exit_code == 1
    assert "procedencia" in resultado.output


@pytest.mark.parametrize("record", [[], None, {}, {"clip_id": "a", "shot_source": None}])
def test_provenance_reader_rejects_malformed_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], record: object
) -> None:
    (tmp_path / "instant-observations.jsonl").write_text(json.dumps(record) + "\n")

    with pytest.raises(Exit) as failure:
        cli._procedencia(tmp_path, {"a"})  # pyright: ignore[reportPrivateUsage]

    assert failure.value.exit_code == 1
    assert "registro de procedencia invalido" in capsys.readouterr().err


def test_eval_filtra_o_gabarito_pela_dobra(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Sem o filtro, a unica forma de respeitar as dobras 0 a 2 era recortar um `--truth` a mao."""
    import cli

    execucao = tmp_path / "run"
    execucao.mkdir()
    (execucao / "events.jsonl").write_text("", encoding="utf-8")
    verdade = tmp_path / "truth.json"
    verdade.write_text(json.dumps({"dev": {"c": "x"}, "res": {"c": "x"}}), encoding="utf-8")

    class Registro:
        def __init__(self, shot_id: str, fold: int) -> None:
            self.shot_id = shot_id
            self.fold = fold
            self.big_enough = True

    monkeypatch.setattr(cli, "load_catalog", lambda: [Registro("dev", 1), Registro("res", 3)])
    vistos: list[dict[str, dict[str, str]]] = []

    def espia(_predicoes: Any, gabarito: Any, _tabela: Any) -> Any:
        vistos.append(dict(gabarito))
        raise SystemExit(0)

    monkeypatch.setattr(cli, "evaluate", espia)
    runner.invoke(app, ["eval", str(execucao), "--truth", str(verdade), "--dobra", "ajuste"])

    assert vistos == [{"dev": {"c": "x"}}]


def test_eval_recusa_dobra_que_deixa_o_gabarito_vazio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cli

    execucao = tmp_path / "run"
    execucao.mkdir()
    (execucao / "events.jsonl").write_text("", encoding="utf-8")
    verdade = tmp_path / "truth.json"
    verdade.write_text(json.dumps({"res": {"c": "x"}}), encoding="utf-8")

    class Registro:
        def __init__(self) -> None:
            self.shot_id = "res"
            self.fold = 4
            self.big_enough = True

    monkeypatch.setattr(cli, "load_catalog", lambda: [Registro()])

    resultado = runner.invoke(
        app, ["eval", str(execucao), "--truth", str(verdade), "--dobra", "ajuste"]
    )

    assert resultado.exit_code == 1
    assert "nenhum clipe do gabarito esta na dobra ajuste" in resultado.output


def test_gabarito_sem_dobra_e_recusado(monkeypatch: pytest.MonkeyPatch) -> None:
    """O padrao seguro nao existe aqui, entao a opcao passa a ser obrigatoria na combinacao.

    `--gabarito` abre os 38 clipes anotados e 14 deles sao o pacote reservado. Deixar o padrao em
    `tudo` transformava a protecao em lembrar de digitar, e a execucao terminava com sucesso.
    """
    resultado = runner.invoke(app, ["run", "--all", "--gabarito"])

    assert resultado.exit_code == 1
    assert "passe --dobra explicitamente" in resultado.output
    for nome in DOBRAS_DA_BOLA:
        assert nome in resultado.output


def test_all_sem_gabarito_continua_valendo_o_catalogo_inteiro(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exigir a dobra em `--all` mudaria o significado de processar o catalogo, e nao e disso que a
    protecao trata: sem `--gabarito` nenhum clipe e aberto por causa de anotacao."""
    import cli

    class Registro:
        def __init__(self, shot_id: str, fold: int) -> None:
            self.shot_id = shot_id
            self.fold = fold
            self.big_enough = True
            self.match_id = "900"

    class Evento:
        def __init__(self, clip: str) -> None:
            self.clip = clip

        def to_dict(self) -> dict[str, str]:
            return {"clip_id": self.clip}

    class Resultado:
        def __init__(self, clip: str) -> None:
            self.event = Evento(clip)
            self.stages: tuple[object, ...] = ()
            self.shot_source = "ball-break"
            self.ball_support = "detected"
            self.shot_candidates: tuple[Mapping[str, object], ...] = ()
            self.shot_candidate_pool: tuple[int, ...] = ()
            self.tunings: Mapping[str, str] = {}

    vistos: list[str] = []

    def processa(registro: Any, _config: Path) -> Any:
        vistos.append(registro.shot_id)
        return Resultado(registro.shot_id)

    monkeypatch.setattr(cli, "read_labels", _rotulos)
    monkeypatch.setattr(cli, "load_catalog", _catalogo([Registro("a", 0), Registro("b", 4)]))
    monkeypatch.setattr(cli, "_populacao", _populacao)
    monkeypatch.setattr(cli, "_processa_registro", processa)

    resultado = runner.invoke(
        app,
        ["run", "--all", "--config", "config/pipeline.toml", "--out", str(tmp_path)],
    )

    assert resultado.exit_code == 0, resultado.output
    assert vistos == ["a", "b"], "sem --dobra o padrao continua sendo o catalogo inteiro"


def _rotulos(_: Any) -> object:
    return object()


def _sem_comprimentos(_clips: Iterable[str]) -> dict[str, int]:
    """Sem comprimento nao ha nulo a calcular, e o relatorio diz isso em vez de inventar."""
    return {}


def _comprimentos_de(tabela: dict[str, int]) -> Any:
    def comprimentos(_clips: Iterable[str]) -> dict[str, int]:
        return dict(tabela)

    return comprimentos


def _catalogo(registros: list[Any]) -> Any:
    def load_catalog() -> list[Any]:
        return registros

    return load_catalog


def _populacao(itens: Any, _tabela: Any) -> list[Any]:
    return list(itens)


def test_eval_spots_mede_detector_externo_com_os_mesmos_nulos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Alternativa cujo codigo nao pode entrar em `src/` ainda precisa ser medida aqui.

    O T-DEED e GPL-3.0 contra o CC-BY-4.0 deste projeto. Ele roda fora, escreve `clip_id` e
    `frame_index`, e a comparacao com a regra em vigor usa o mesmo gabarito e os mesmos nulos.
    """
    import cli

    spots = tmp_path / "spots.jsonl"
    spots.write_text(
        json.dumps({"clip_id": "a", "frame_index": 100})
        + "\n"
        + json.dumps({"clip_id": "b", "frame_index": 400})
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "read", lambda: ())
    monkeypatch.setattr(cli, "consensus", _consenso({"a": 110, "b": 100}))
    monkeypatch.setattr(cli, "agreement", _nada)
    monkeypatch.setattr(cli, "_comprimentos", _comprimentos_de({"a": 400, "b": 400}))

    resultado = runner.invoke(app, ["eval-spots", str(spots), "--dobra", "tudo"])

    assert resultado.exit_code == 0, resultado.output
    medida, nulos = _todos_os_json(resultado.output)
    assert medida["n"] == 2
    assert medida["dentro_da_tolerancia"] == 0.5
    assert nulos["linhas_de_base"]["quadro_ao_acaso"] == pytest.approx(61 / 400)


def test_eval_spots_nao_abre_o_reservado_sem_pedirem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """O padrao e `ajuste` e nao `tudo`, porque e aqui que o holdout se perde por descuido."""
    import cli

    class Registro:
        def __init__(self, shot_id: str, fold: int) -> None:
            self.shot_id = shot_id
            self.fold = fold
            self.big_enough = True

    spots = tmp_path / "spots.jsonl"
    spots.write_text(json.dumps({"clip_id": "r", "frame_index": 100}) + "\n", encoding="utf-8")
    monkeypatch.setattr(cli, "read", lambda: ())
    monkeypatch.setattr(cli, "consensus", _consenso({"r": 110}))
    monkeypatch.setattr(cli, "agreement", _nada)
    monkeypatch.setattr(cli, "load_catalog", _catalogo([Registro("r", 4)]))

    resultado = runner.invoke(app, ["eval-spots", str(spots)])

    assert resultado.exit_code == 1
    assert "nenhum clipe anotado na dobra ajuste" in resultado.output
