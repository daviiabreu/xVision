"""A rota INT8: amostra so do ajuste, tensor igual ao do plano e manifesto sem dado do parceiro.

Os clipes sao sinteticos e tem o formato de nome do conjunto real, porque e esse formato que carrega
o desfecho e o identificador que o manifesto nao pode levar. O ONNX dos testes com o ONNX Runtime de
verdade e construido aqui, com uma convolucao e um SiLU, e pula onde o grupo `quantizacao` falta.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from edge.artifacts import sha256_of
from edge.frames.provider import FrameProvider, VideoSource
from edge.modules.ball.detector import DetectorConfig, YoloBallDetector, backend_specs
from edge.modules.field.yolo import OUTPUT, UltralyticsFieldEstimator
from edge.paths import var_dir
from edge.runtime import RuntimeTensor
from edge.testing.make_clip import make_synthetic_clip
from mlops import build_engine, quantize_int8
from mlops.catalog import ShotRecord
from mlops.quantize_int8 import (
    CALIBRACAO,
    CalibracaoError,
    LeitorDeCalibracao,
    especie_de,
    extrai,
    main,
    quantiza,
    sorteia_clipes,
)
from mlops.split import fold_of, read_split, split_config

PROIBIDAS = frozenset({"clip_id", "shot_id", "match_id", "id_match", "clip_path", "path", "clips"})
QDQ_OPS = ("QuantizeLinear", "DequantizeLinear")


def _registro(shot_id: str, fold: int, caminho: Path, *, abrivel: bool = True) -> ShotRecord:
    return ShotRecord(
        shot_id=shot_id,
        match_id=f"9{fold}",
        clip_path=caminho,
        big_enough=abrivel,
        fold=fold,
        labels={},
    )


def _config(tmp_path: Path, especie: str, extra: str = "") -> Path:
    impl = {"ball": "edge.modules.ball:YoloBall", "field": "edge.modules.field:YoloField"}[especie]
    caminho = tmp_path / f"pipeline.{especie}.toml"
    caminho.write_text(
        '[frames.dense]\nstrategy = "uniform"\nfps = 5\n\n'
        f'[modules.{especie}]\nimpl = "{impl}"\nframes = "dense"\n{extra}',
        encoding="utf-8",
    )
    return caminho


BOLA_PEQUENA = "imgsz = 320\nladrilhos = 2\nsobreposicao_px = 100\nladrilhos_por_lote = {lote}\n"


def _chaves(valor: object) -> set[str]:
    if isinstance(valor, dict):
        itens = cast("dict[str, object]", valor)
        return set(itens) | {c for v in itens.values() for c in _chaves(v)}
    if isinstance(valor, list):
        return {c for v in cast("list[object]", valor) for c in _chaves(v)}
    return set()


def test_o_sorteio_so_ve_as_dobras_de_ajuste(tmp_path: Path) -> None:
    registros = [_registro(str(1000 + i), i % 5, tmp_path / f"{i}.mp4") for i in range(20)]

    sorteio = sorteia_clipes(registros, 40, random.Random(3))

    assert {r.fold for r, _ in sorteio} <= {0, 1, 2}
    assert len(sorteio) == 12
    assert sum(cota for _, cota in sorteio) == 40


@pytest.mark.parametrize("dobras", [frozenset({3}), frozenset({0, 4}), frozenset({3, 4})])
def test_pedir_dobra_reservada_e_recusado(tmp_path: Path, dobras: frozenset[int]) -> None:
    registros = [_registro(str(1000 + i), i % 5, tmp_path / f"{i}.mp4") for i in range(10)]

    with pytest.raises(CalibracaoError, match="reservado"):
        sorteia_clipes(registros, 4, random.Random(0), dobras)


def test_so_o_reservado_disponivel_nao_vira_calibracao(tmp_path: Path) -> None:
    registros = [_registro(str(1000 + i), 3 + i % 2, tmp_path / f"{i}.mp4") for i in range(6)]

    with pytest.raises(CalibracaoError, match="nenhum clipe"):
        sorteia_clipes(registros, 4, random.Random(0))


def test_a_mesma_semente_da_a_mesma_amostra_em_qualquer_ordem_do_catalogo(tmp_path: Path) -> None:
    registros = [_registro(str(1000 + i), i % 3, tmp_path / f"{i}.mp4") for i in range(9)]
    embaralhado = random.Random(1).sample(registros, len(registros))

    primeira = sorteia_clipes(registros, 5, random.Random(11))
    segunda = sorteia_clipes(embaralhado, 5, random.Random(11))

    assert [(r.shot_id, c) for r, c in primeira] == [(r.shot_id, c) for r, c in segunda]
    assert [c for _, c in primeira] == [1, 1, 1, 1, 1]


def test_clipe_truncado_fica_fora_e_a_cota_se_espalha(tmp_path: Path) -> None:
    registros = [
        _registro("1", 0, tmp_path / "1.mp4"),
        _registro("2", 1, tmp_path / "2.mp4"),
        _registro("3", 2, tmp_path / "3.mp4", abrivel=False),
    ]

    sorteio = sorteia_clipes(registros, 5, random.Random(0))

    assert sorted(r.shot_id for r, _ in sorteio) == ["1", "2"]
    assert sorted(c for _, c in sorteio) == [2, 3]
    with pytest.raises(CalibracaoError, match="ao menos um"):
        sorteia_clipes(registros, 0, random.Random(0))


class _Captura:
    def __init__(self, saida: np.ndarray[Any, Any]) -> None:
        self.entradas: list[np.ndarray[Any, Any]] = []
        self.info = SimpleNamespace(inference_count=0)
        self._saida = saida

    def infer(self, inputs: dict[str, np.ndarray[Any, Any]]) -> tuple[RuntimeTensor, ...]:
        self.entradas.append(inputs["images"])
        return (RuntimeTensor("predictions", self._saida),)

    def close(self) -> None: ...


@pytest.mark.parametrize("lote", [4, 2])
def test_o_tensor_da_bola_e_byte_a_byte_o_que_o_backend_recebe(tmp_path: Path, lote: int) -> None:
    """Replantar uma montagem propria na calibracao, com outro letterbox, derruba este teste."""
    pytest.importorskip("ultralytics")
    especie = especie_de("ball", _config(tmp_path, "ball", BOLA_PEQUENA.format(lote=lote)))
    config = DetectorConfig(
        weights="x", imgsz=320, ladrilhos=2, sobreposicao_px=100, ladrilhos_por_lote=lote
    )
    _, saida = backend_specs(config)
    backend = _Captura(np.zeros(saida.shape, dtype=np.float32))
    detector = YoloBallDetector(None, config, backend=backend)  # type: ignore[arg-type]
    quadro = np.random.default_rng(5).integers(0, 256, (1088, 1920, 3), dtype=np.uint8)

    detector.detect_raw(quadro)
    calibracao = especie.entradas(quadro)

    assert len(calibracao) == len(backend.entradas) == 4 // lote
    for nosso, do_plano in zip(calibracao, backend.entradas, strict=True):
        assert nosso.dtype == do_plano.dtype == np.float32
        assert nosso.tobytes() == do_plano.tobytes()


def test_o_tensor_do_campo_e_byte_a_byte_o_que_o_backend_recebe(tmp_path: Path) -> None:
    pytest.importorskip("ultralytics")
    especie = especie_de("field", _config(tmp_path, "field"))
    backend = _Captura(np.zeros(OUTPUT.shape, dtype=np.float32))
    estimador = UltralyticsFieldEstimator(
        None,
        imgsz=960,
        conf=0.25,
        device="cpu",
        backend=backend,  # type: ignore[arg-type]
    )
    quadro = np.random.default_rng(6).integers(0, 256, (544, 960, 3), dtype=np.uint8)

    estimador.estimate(quadro)
    (nosso,) = especie.entradas(quadro)

    assert nosso.tobytes() == backend.entradas[0].tobytes()


def test_especie_que_nao_tem_tensor_de_plano_conhecido_e_recusada(tmp_path: Path) -> None:
    config = tmp_path / "pipeline.toml"
    config.write_text('[modules.ball]\nimpl = "edge.modules.stubs:StubBall"\n')

    with pytest.raises(CalibracaoError, match="rota INT8 existe"):
        especie_de("players", config)
    with pytest.raises(CalibracaoError, match="nao declara modules.field"):
        especie_de("field", config)
    with pytest.raises(CalibracaoError, match="StubBall"):
        especie_de("ball", config)


def test_a_extracao_do_campo_grava_o_tensor_do_plano_e_nada_do_parceiro(tmp_path: Path) -> None:
    clipe = make_synthetic_clip(tmp_path / "cortes" / "90001" / "90001_004_Mexico_gol_4996031.mp4")
    registros = [
        _registro("4996031", 1, clipe),
        _registro("4996032", 3, tmp_path / "nao-existe" / "90003_001_Suica_off_4996032.mp4"),
    ]
    config = _config(tmp_path, "field")
    destino = var_dir() / "calibracao" / "field"

    relatorio = extrai("field", config, destino, quantos=3, semente=7, registros=registros)

    tensores = sorted(destino.glob("*.npy"))
    assert [t.name for t in tensores] == ["0000.npy", "0001.npy", "0002.npy"]
    assert relatorio["sha256"] == [sha256_of(t) for t in tensores]
    assert relatorio["shape"] == [1, 3, 544, 960]
    assert relatorio["dobras"] == [0, 1, 2]
    especie = especie_de("field", config)
    provider = FrameProvider(VideoSource(clipe))
    primeiro = provider.frames([0], especie.spec.preprocess).tensor[0]
    provider.close()
    assert np.load(tensores[0]).dtype == np.float32
    assert np.load(tensores[0]).shape == especie.entradas(primeiro)[0].shape
    texto = (destino / CALIBRACAO).read_text(encoding="utf-8")
    for dado_do_parceiro in ("4996031", "90001", "Mexico", ".mp4"):
        assert dado_do_parceiro not in texto
    assert not _chaves(json.loads(texto)) & PROIBIDAS
    with pytest.raises(FileExistsError):
        extrai("field", config, destino, quantos=1, registros=registros)


def test_conjunto_que_depende_do_instante_nao_calibra(tmp_path: Path) -> None:
    config = tmp_path / "pipeline.toml"
    config.write_text(
        '[frames.window]\nstrategy = "around"\nanchor = "shot"\nbefore_s = 1.0\nafter_s = 0.5\n'
        '\n[modules.field]\nimpl = "edge.modules.field:YoloField"\nframes = "window"\n'
    )
    registros = [_registro("1", 0, tmp_path / "1.mp4")]

    with pytest.raises(CalibracaoError, match="uniform"):
        extrai("field", config, tmp_path / "saida", quantos=1, registros=registros)


def _dump_sintetico(raiz: Path) -> dict[str, str]:
    """Um dump com o esquema do parceiro: uma partida no ajuste e outra no reservado."""
    particao = read_split(split_config())
    por_dobra: dict[str, str] = {}
    for candidata in range(90000, 90200):
        dobra = fold_of(str(candidata), particao)
        chave = "ajuste" if dobra <= 2 else "reservado"
        por_dobra.setdefault(chave, str(candidata))
    tiros = [
        {"id": "4996031", "id_match": por_dobra["ajuste"]},
        {"id": "4996032", "id_match": por_dobra["reservado"]},
    ]
    for nome, linhas in (
        ("0005_match_shots_1390.csv", tiros),
        ("0006_match_shots_xg_1390.csv", []),
    ):
        caminho = raiz / "db" / nome
        caminho.parent.mkdir(parents=True, exist_ok=True)
        with caminho.open("w", newline="", encoding="utf-8") as arquivo:
            campos = ["id", "id_match"] if linhas else ["id", "id_match_shot"]
            escritor = csv.DictWriter(arquivo, fieldnames=campos)
            escritor.writeheader()
            escritor.writerows(linhas)
    for tiro, desfecho in zip(tiros, ("gol", "off"), strict=True):
        pasta = raiz / "cortes" / tiro["id_match"]
        nome = f"{tiro['id_match']}_004_Mexico_{desfecho}_{tiro['id']}.mp4"
        make_synthetic_clip(pasta / nome, seed=int(tiro["id"]) % 7)
    return por_dobra


def _fakes(
    monkeypatch: pytest.MonkeyPatch, entrada: tuple[int, ...], qdq: bool = True
) -> dict[str, Any]:
    visto: dict[str, Any] = {}

    def carrega(caminho: str) -> SimpleNamespace:
        dims = [SimpleNamespace(dim_value=d) for d in entrada]
        tipo = SimpleNamespace(tensor_type=SimpleNamespace(shape=SimpleNamespace(dim=dims)))
        nos = [
            SimpleNamespace(op_type="Conv", name="/model.0/conv/Conv"),
            SimpleNamespace(op_type="Conv", name="/model.23/cv2.0/conv/Conv"),
            SimpleNamespace(op_type="Concat", name="/model.23/Concat"),
        ]
        if qdq and Path(caminho).name.endswith("int8.onnx"):
            nos += [SimpleNamespace(op_type="QuantizeLinear", name="q")] * 2
            nos += [SimpleNamespace(op_type="DequantizeLinear", name="dq")] * 3
        return SimpleNamespace(
            graph=SimpleNamespace(
                input=[SimpleNamespace(name="images", type=tipo)], initializer=[], node=nos
            )
        )

    def quantize_static(**argumentos: Any) -> None:
        leitor = argumentos["calibration_data_reader"]
        lidos: list[np.ndarray[Any, Any]] = []
        for inicio in range(len(leitor)):
            leitor.set_range(start_index=inicio, end_index=inicio + 1)
            while (lote := leitor.get_next()) is not None:
                lidos.append(lote["images"])
        visto.update(argumentos, lidos=lidos)
        Path(argumentos["model_output"]).write_bytes(b"onnx com q/dq")

    ort = SimpleNamespace(
        quantize_static=quantize_static,
        QuantFormat=SimpleNamespace(QDQ="qdq"),
        QuantType=SimpleNamespace(QInt8="qint8"),
        CalibrationMethod=SimpleNamespace(MinMax="minmax"),
    )
    modulos = {"onnx": SimpleNamespace(load=carrega), "onnxruntime.quantization": ort}
    monkeypatch.setattr(quantize_int8, "import_module", modulos.__getitem__)

    def versao(nome: str) -> str:
        return {"onnxruntime": "1.23.2"}.get(nome, "0")

    monkeypatch.setattr(quantize_int8, "version", versao)
    return visto


def test_da_extracao_da_bola_ao_manifesto_nada_identifica_o_parceiro(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A cadeia pela linha de comando, com o catalogo lido do dump e o ONNX Runtime de mentira."""
    raiz = tmp_path / "dump"
    por_dobra = _dump_sintetico(raiz)
    monkeypatch.setenv("XVISION_DATA_DIR", str(raiz))
    config = _config(tmp_path, "ball", BOLA_PEQUENA.format(lote=4))

    main(["calibracao", "--especie", "ball", "--config", str(config), "--quadros", "2"])
    capsys.readouterr()
    pasta = var_dir() / "calibracao" / "ball"
    extracao = json.loads((pasta / CALIBRACAO).read_text(encoding="utf-8"))
    assert extracao["clipes"] == 1
    assert extracao["shape"] == [4, 3, 224, 320]
    assert extracao["ajuste"]["ladrilhos"] == 2

    onnx = tmp_path / "ball-b4.onnx"
    onnx.write_bytes(b"onnx exportado")
    visto = _fakes(monkeypatch, (4, 3, 224, 320))
    saida = tmp_path / "engines" / "ball-b4-int8.onnx"
    main(["quantiza", "--onnx", str(onnx), "--calibracao", str(pasta), "--output", str(saida)])

    manifesto = json.loads(saida.with_suffix(".run.json").read_text(encoding="utf-8"))
    assert json.loads(capsys.readouterr().out) == manifesto
    assert manifesto["onnx_sha256"] == sha256_of(onnx)
    assert manifesto["saida_sha256"] == sha256_of(saida)
    assert manifesto["tensores"] == 2
    assert manifesto["tensores_sha256"] == extracao["sha256"]
    assert manifesto["semente"] == 0
    assert manifesto["metodo"] == "MinMax"
    assert manifesto["onnxruntime"] == "1.23.2"
    assert manifesto["nos_qdq"] == {"QuantizeLinear": 2, "DequantizeLinear": 3}
    assert visto["calibrate_method"] == "minmax"
    assert visto["nodes_to_exclude"] == ["/model.23/cv2.0/conv/Conv"]
    assert manifesto["fora_do_int8"] == {"bloco": "/model.23/", "convolucoes": 1}
    assert visto["per_channel"] is True
    assert visto["extra_options"]["ActivationSymmetric"] is True
    assert visto["extra_options"]["QuantizeBias"] is False
    assert [a.tobytes() for a in visto["lidos"]] == [
        np.load(t).tobytes() for t in sorted(pasta.glob("*.npy"))
    ]
    texto = saida.with_suffix(".run.json").read_text(encoding="utf-8")
    for dado_do_parceiro in ("4996031", "4996032", *por_dobra.values(), "Mexico", ".mp4"):
        assert dado_do_parceiro not in texto
    assert not _chaves(manifesto) & PROIBIDAS


def _calibracao_aleatoria(pasta: Path, forma: tuple[int, ...], quantos: int) -> list[Path]:
    pasta.mkdir(parents=True)
    rng = np.random.default_rng(0)
    tensores: list[Path] = []
    for indice in range(quantos):
        caminho = pasta / f"{indice:04d}.npy"
        np.save(caminho, rng.random(forma, dtype=np.float32))
        tensores.append(caminho)
    relatorio: dict[str, Any] = {
        "especie": "field",
        "config_sha256": "a" * 64,
        "frames": "dense",
        "fps": None,
        "preprocess": {
            "width": forma[-1],
            "height": forma[-2],
            "color": "bgr",
            "letterbox": True,
            "normalize": "none",
            "ops": [],
        },
        "ajuste": {},
        "dobras": [0, 1, 2],
        "semente": 13,
        "quadros_pedidos": quantos,
        "quadros": quantos,
        "clipes": 1,
        "shape": list(forma),
        "dtype": "float32",
        "sha256": [sha256_of(t) for t in tensores],
        "versoes": {"numpy": "2.2.6", "opencv-python-headless": "4.12.0.88"},
        "extrator_sha256": "b" * 64,
    }
    (pasta / CALIBRACAO).write_text(json.dumps(relatorio), encoding="utf-8")
    return tensores


def test_calibracao_alterada_ou_sem_procedencia_e_recusada(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fakes(monkeypatch, (1, 3, 8, 8))
    onnx = tmp_path / "m.onnx"
    onnx.write_bytes(b"m")
    pasta = tmp_path / "cal"
    tensores = _calibracao_aleatoria(pasta, (1, 3, 8, 8), 2)
    saida = tmp_path / "m-int8.onnx"

    np.save(tensores[1], np.zeros((1, 3, 8, 8), dtype=np.float32))
    with pytest.raises(CalibracaoError, match="nao sao os que a extracao gravou"):
        quantiza(onnx, pasta, saida)
    (pasta / CALIBRACAO).unlink()
    with pytest.raises(CalibracaoError, match="calibracao.json"):
        quantiza(onnx, pasta, saida)
    with pytest.raises(CalibracaoError, match="nao existe"):
        quantiza(tmp_path / "nada.onnx", pasta, saida)
    assert not saida.exists()


def test_tensor_com_shape_de_outro_plano_e_recusado_antes_do_onnx_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    visto = _fakes(monkeypatch, (4, 3, 8, 8))
    onnx = tmp_path / "m.onnx"
    onnx.write_bytes(b"m")
    pasta = tmp_path / "cal"
    _calibracao_aleatoria(pasta, (1, 3, 8, 8), 1)

    with pytest.raises(CalibracaoError, match=r"o ONNX pede \(4, 3, 8, 8\)"):
        quantiza(onnx, pasta, tmp_path / "m-int8.onnx")
    assert "lidos" not in visto


def test_quantizacao_que_nao_grava_q_dq_e_erro(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fakes(monkeypatch, (1, 3, 8, 8), qdq=False)
    onnx = tmp_path / "m.onnx"
    onnx.write_bytes(b"m")
    pasta = tmp_path / "cal"
    _calibracao_aleatoria(pasta, (1, 3, 8, 8), 1)

    with pytest.raises(CalibracaoError, match="sem no Q/DQ"):
        quantiza(onnx, pasta, tmp_path / "m-int8.onnx")


def test_o_leitor_entrega_um_tensor_por_vez_e_respeita_a_faixa(tmp_path: Path) -> None:
    tensores = _calibracao_aleatoria(tmp_path / "cal", (1, 2), 3)
    leitor = LeitorDeCalibracao("x", tensores)

    assert len(leitor) == 3
    leitor.set_range(start_index=1, end_index=2)
    lote = leitor.get_next()
    assert lote is not None
    assert lote["x"].tobytes() == np.load(tensores[1]).tobytes()
    assert leitor.get_next() is None


def _modelo_pequeno(caminho: Path) -> None:
    """Conv com vies, SiLU e uma cabeca 1x1, com nome e opset como o exportador grava."""
    onnx = pytest.importorskip("onnx")
    helper, numpy_helper = onnx.helper, onnx.numpy_helper
    rng = np.random.default_rng(1)
    pesos = [
        numpy_helper.from_array(rng.normal(size=(4, 3, 3, 3)).astype(np.float32), "w1"),
        numpy_helper.from_array(rng.normal(size=(4,)).astype(np.float32), "b1"),
        numpy_helper.from_array(rng.normal(size=(2, 4, 1, 1)).astype(np.float32), "w2"),
    ]
    nos = [
        helper.make_node(
            "Conv", ["images", "w1", "b1"], ["c1"], name="/model.0/conv/Conv", pads=[1, 1, 1, 1]
        ),
        helper.make_node("Sigmoid", ["c1"], ["s1"], name="/model.0/act/Sigmoid"),
        helper.make_node("Mul", ["c1", "s1"], ["a1"], name="/model.0/act/Mul"),
        helper.make_node("Conv", ["a1", "w2"], ["output0"], name="/model.1/cv2/Conv"),
    ]
    grafo = helper.make_graph(
        nos,
        "pequeno",
        [helper.make_tensor_value_info("images", onnx.TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("output0", onnx.TensorProto.FLOAT, [1, 2, 8, 8])],
        pesos,
    )
    modelo = helper.make_model(grafo, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    onnx.checker.check_model(modelo)
    onnx.save(modelo, str(caminho))


def test_o_onnx_runtime_de_verdade_grava_q_dq_simetrico_que_ainda_roda(tmp_path: Path) -> None:
    ort = pytest.importorskip("onnxruntime")
    onnx = pytest.importorskip("onnx")
    fonte = tmp_path / "pequeno.onnx"
    _modelo_pequeno(fonte)
    tensores = _calibracao_aleatoria(tmp_path / "cal", (1, 3, 8, 8), 4)
    saida = tmp_path / "pequeno-int8.onnx"

    manifesto = quantiza(fonte, tmp_path / "cal", saida)

    modelo = onnx.load(str(saida))
    tipos = {no.op_type for no in modelo.graph.node}
    assert {"QuantizeLinear", "DequantizeLinear"} <= tipos
    iniciais = {i.name: onnx.numpy_helper.to_array(i) for i in modelo.graph.initializer}
    pontos_zero = [iniciais[no.input[2]] for no in modelo.graph.node if no.op_type in QDQ_OPS]
    assert all(not ponto.any() for ponto in pontos_zero)
    assert "b1" in iniciais
    assert iniciais["b1"].dtype == np.float32
    assert "w2" in iniciais
    assert iniciais["w2"].dtype == np.float32
    assert manifesto["fora_do_int8"] == {"bloco": "/model.1/", "convolucoes": 1}
    assert manifesto["onnx_sha256"] == sha256_of(fonte)
    assert manifesto["saida_sha256"] == sha256_of(saida)
    assert manifesto["tensores_sha256"] == [sha256_of(t) for t in tensores]
    assert manifesto["semente"] == 13
    assert manifesto["onnxruntime"] == ort.__version__
    assert not _chaves(manifesto) & PROIBIDAS
    entrada = {"images": np.load(tensores[0])}
    antes = ort.InferenceSession(str(fonte), providers=["CPUExecutionProvider"]).run(None, entrada)
    depois = ort.InferenceSession(str(saida), providers=["CPUExecutionProvider"]).run(None, entrada)
    assert np.abs(antes[0] - depois[0]).max() < 0.1 * np.abs(antes[0]).max()


def _tensorrt_de_mentira(visto: dict[str, Any]) -> SimpleNamespace:
    """So o que `build_engine.build` toca, para o ONNX de trabalho ser o do onnx de verdade."""
    tensor = SimpleNamespace(name="images", shape=(1, 3, 8, 8))

    def io(indice: int) -> SimpleNamespace:
        return tensor

    def camada(indice: int) -> SimpleNamespace:
        return SimpleNamespace(type="quantize")

    def nada(*argumentos: object) -> None: ...

    class Builder:
        def __init__(self, logger: object) -> None: ...

        def create_network(self, flags: int) -> SimpleNamespace:
            visto["rede"] = flags
            return SimpleNamespace(
                num_inputs=1,
                get_input=io,
                num_outputs=1,
                get_output=io,
                num_layers=1,
                get_layer=camada,
            )

        def create_builder_config(self) -> SimpleNamespace:
            return SimpleNamespace(set_memory_pool_limit=nada, set_flag=nada)

        def build_serialized_network(self, network: object, config: object) -> bytes:
            return b"plano"

    class Parser:
        num_errors = 0

        def __init__(self, network: object, logger: object) -> None: ...

        def parse(self, dados: bytes) -> bool:
            visto["onnx"] = dados
            return True

    class Logger:
        WARNING = 2

        def __init__(self, nivel: int) -> None: ...

    return SimpleNamespace(
        Logger=Logger,
        Builder=Builder,
        OnnxParser=Parser,
        MemoryPoolType=SimpleNamespace(WORKSPACE="workspace"),
        LayerType=SimpleNamespace(QUANTIZE="quantize"),
        NetworkDefinitionCreationFlag=SimpleNamespace(STRONGLY_TYPED=1),
        BuilderFlag=SimpleNamespace(),
        __version__="11.3.0.99",
    )


def test_o_q_dq_com_fp16_sai_valido_no_opset_19_com_entrada_e_saida_em_fp32(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """O ONNX que o parser do TensorRT recebe, com o onnx e o conversor de verdade.

    Abaixo do opset 19 a escala fp16 do Q/DQ e invalida pela especificacao, e o `check_model`
    completo a recusaria.
    """
    pytest.importorskip("onnxruntime")
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxconverter_common")
    fonte = tmp_path / "pequeno.onnx"
    _modelo_pequeno(fonte)
    _calibracao_aleatoria(tmp_path / "cal", (1, 3, 8, 8), 2)
    quantizado = tmp_path / "pequeno-int8.onnx"
    quantiza(fonte, tmp_path / "cal", quantizado)
    visto: dict[str, Any] = {}
    trt = _tensorrt_de_mentira(visto)
    reais = build_engine.import_module

    def importa(nome: str) -> Any:
        return trt if nome == "tensorrt" else reais(nome)

    def distribuicao(modulo: object) -> str:
        return "onnx"

    monkeypatch.setattr(build_engine, "import_module", importa)
    monkeypatch.setattr(build_engine, "_distribuicao", distribuicao)
    destino = tmp_path / "pequeno-int8.engine"

    relatorio = build_engine.build(quantizado, destino, fp16=True, saida="predictions")

    trabalho = onnx.load_from_string(visto["onnx"])
    onnx.checker.check_model(trabalho, full_check=True)
    assert visto["rede"] == 1 << 1
    assert relatorio["opset"] == "17 -> 19"
    assert relatorio["onnx_sha256"] == sha256_of(quantizado)
    assert [o.version for o in trabalho.opset_import if o.domain in ("", "ai.onnx")] == [19]
    iniciais = {i.name: i.data_type for i in trabalho.graph.initializer}
    escalas = [iniciais[no.input[1]] for no in trabalho.graph.node if no.op_type in QDQ_OPS]
    assert escalas
    assert set(escalas) == {onnx.TensorProto.FLOAT16}
    assert trabalho.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert trabalho.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert trabalho.graph.output[0].name == "predictions"
