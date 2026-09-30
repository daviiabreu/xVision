"""Rota INT8 versionada de `ball` e `field`: calibracao pelo pipeline, ONNX com Q/DQ e manifesto.

O TensorRT 11.3 da placa nao calibra: nao tem `IInt8Calibrator` nem `BuilderFlag.INT8`, e toda rede
e de tipo fixo. O INT8 so chega pelos nos QuantizeLinear e DequantizeLinear que o ONNX ja traz, e
quem os grava aqui e a quantizacao estatica do ONNX Runtime em formato QDQ, o caminho pelo qual o
INT8 do Danilo foi construido e rodou na placa.

A calibracao sai do proprio pipeline, e nao de um script a parte: decode do `FrameProvider`,
`PreprocessSpec` da especie, conjunto de quadros declarado no TOML e, na bola, os ladrilhos em lote
pela mesma funcao que monta o despacho do detector. O tensor calibrado e byte a byte o que o plano
recebe em producao, o que o script de `models/quantizacao`, a 1280 px, nao era. So entram clipes
das dobras de ajuste, 0 a 2; o pacote reservado nao calibra nada.

Calibracao MinMax, porque ela reduz cada ativacao a dois escalares dentro do grafo e cabe na Orin.

Na placa, em `src/`, com `uv sync --frozen --all-extras --group jetson --group quantizacao`. O ONNX
e o mesmo de onde saiu o plano fp16; se a pasta da exportacao ja existir, pule o primeiro comando,
porque o exportador recusa pasta existente:

  python -m mlops.export_onnx \
    --manifest models/manifests/ball/yolo-football-ball-detection-0.1.0.toml \
    --height 992 --width 1600 --batch 4 --output ../var/engines/ball-b4-onnx
  python -m mlops.quantize_int8 calibracao --especie ball \
    --config config/pipeline.models.jetson.toml --output ../var/calibracao/ball
  python -m mlops.quantize_int8 quantiza --onnx ../var/engines/ball-b4-onnx/model-1.onnx \
    --calibracao ../var/calibracao/ball --output ../var/engines/ball-b4-int8.onnx
  python -m mlops.build_engine --onnx ../var/engines/ball-b4-int8.onnx \
    --output ../var/engines/ball-b4-int8.engine --fp16 --workspace-gib 2 --saida predictions

e para o campo, com 544x960 e lote 1:

  python -m mlops.export_onnx \
    --manifest models/manifests/field/pitch-keypoints-yolo11n-pose-0.1.0.toml \
    --height 544 --width 960 --output ../var/engines/field-onnx
  python -m mlops.quantize_int8 calibracao --especie field \
    --config config/pipeline.models.jetson.toml --output ../var/calibracao/field
  python -m mlops.quantize_int8 quantiza --onnx ../var/engines/field-onnx/model-1.onnx \
    --calibracao ../var/calibracao/field --output ../var/engines/field-int8.onnx
  python -m mlops.build_engine --onnx ../var/engines/field-int8.onnx \
    --output ../var/engines/field-int8.engine --fp16 --workspace-gib 2 --saida predictions

A calibracao da bola le `letterbox = "cuda"` do TOML da Jetson e precisa do torch com CUDA, como em
producao. Os 64 tensores da bola ocupam 4,9 GB em disco, e os do campo 0,4 GB. O `--fp16` deixa em
fp16 o que ficou fora dos Q/DQ, cabeca inclusive; sem ele essas camadas ficam em fp32, e so essa
variante tem concordancia medida fora da placa, porque a de fp16 nao se confere no ONNX Runtime em
CPU (ver `mlops.build_engine`). O `engine_onnx_sha256` de um plano INT8 e o `onnx_sha256` do
`run.json` do plano, que e o sha256 do ONNX com Q/DQ.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import numpy as np

from edge.artifacts import sha256_of
from edge.frames.provider import FrameProvider, VideoSource, probe
from edge.graph.framesets import UniformFrames, read_framesets, uniform_indices
from edge.modules.ball.detector import entradas_do_backend
from edge.modules.ball.module import YoloBall
from edge.modules.base import ModuleSpec
from edge.modules.field.module import YoloField
from edge.modules.field.yolo import tensor_de_entrada
from edge.modules.registry import load_modules, read_config
from edge.paths import var_dir
from mlops.catalog import ShotRecord, load_catalog
from mlops.split import DOBRAS

Tensor = np.ndarray[Any, Any]

ESPECIES = ("ball", "field")
QUADROS_PADRAO = 64
SEMENTE_PADRAO = 0
CALIBRACAO = "calibracao.json"
CALIBRATION_FIELDS = frozenset(
    {
        "especie",
        "config_sha256",
        "frames",
        "fps",
        "preprocess",
        "ajuste",
        "dobras",
        "semente",
        "quadros_pedidos",
        "quadros",
        "clipes",
        "shape",
        "dtype",
        "sha256",
        "versoes",
        "extrator_sha256",
    }
)
QDQ = ("QuantizeLinear", "DequantizeLinear")
OPERADORES = ("Conv",)
BLOCO = re.compile(r"^/model\.(\d+)/")

METODO = "MinMax"
"""A calibracao MinMax do ONNX Runtime, e o motivo e memoria.

Ela acrescenta ReduceMin e ReduceMax ao grafo, entao cada passada devolve dois escalares por tensor
e a ativacao some quando a camada seguinte a consome. Entropy e Percentile devolvem cada ativacao
inteira como saida do grafo: medido no ONNX da bola em lote 4, as entradas e saidas das convolucoes
somam 2,18 GiB em fp32 por tensor de calibracao, numa placa de 8 GB dividida entre CPU e GPU.

O preco e a sensibilidade a extremo: um quadro atipico alarga a escala do tensor inteiro. No campo
ele nao foi o que pesou, medido no ONNX Runtime em CPU contra o fp32: com a cabeca fora do INT8 o
keypoint desloca 1,4 px na mediana, e o Percentile 99,999 com a cabeca dentro so chegou a 6,5 px.
Quem diz se custou deteccao no plano e a medida contra o fp16 na placa, e nao este arquivo.
"""


class CalibracaoError(ValueError): ...


def opcoes_do_tensorrt() -> dict[str, object]:
    """As opcoes do `quantize_static` que fazem o ONNX ser construivel pelo TensorRT.

    Ponto zero 0 nas ativacoes e nos pesos, porque o TensorRT so aceita INT8 simetrico. Vies fora
    da quantizacao, porque o TensorRT so quantiza para 8 bits ou menos e o vies quantizado do ONNX
    Runtime e int32. Q/DQ so na entrada da convolucao, que e a colocacao que a NVIDIA recomenda: na
    saida ela separaria a Conv do SiLU que o TensorRT funde nela. Um tensor de calibracao por vez,
    por `CalibStridedMinMax`; o `CalibMaxIntermediateOutputs`, que prometeria o mesmo, descarta a
    faixa acumulada no ONNX Runtime 1.23.2 em vez de junta-la.

    Funcao, e nao constante, porque o `quantize_static` recebe o dicionario e a lista dentro dele,
    e nada garante que ele nao os altere.
    """
    return {
        "ActivationSymmetric": True,
        "WeightSymmetric": True,
        "QuantizeBias": False,
        "OpTypesToExcludeOutputQuantization": list(OPERADORES),
        "CalibStridedMinMax": 1,
    }


@dataclass(frozen=True)
class Especie:
    """O que a calibracao precisa saber de uma especie para montar o tensor do plano."""

    spec: ModuleSpec
    entradas: Callable[[Tensor], tuple[Tensor, ...]]
    ajuste: dict[str, object]


def especie_de(nome: str, config: Path) -> Especie:
    """A especie configurada como o pipeline a configura, pelo registro e sem carregar peso."""
    if nome not in ESPECIES:
        raise CalibracaoError(f"especie {nome!r}: a rota INT8 existe para {', '.join(ESPECIES)}")
    entrada = next((c for c in read_config(config) if c.kind == nome), None)
    if entrada is None:
        raise CalibracaoError(f"{config} nao declara modules.{nome}")
    modulo = load_modules((entrada,))[nome]
    if isinstance(modulo, YoloBall):
        detector = modulo.ajuste_do_detector()
        ajuste: dict[str, object] = {
            campo: getattr(detector, campo)
            for campo in (
                "imgsz",
                "ladrilhos",
                "sobreposicao_px",
                "ladrilhos_por_lote",
                "letterbox",
            )
        }
        return Especie(modulo.spec, lambda quadro: entradas_do_backend(quadro, detector), ajuste)
    if isinstance(modulo, YoloField):
        return Especie(modulo.spec, lambda quadro: (tensor_de_entrada(quadro),), {})
    raise CalibracaoError(
        f"modules.{nome} aponta para {type(modulo).__name__}, e o tensor do plano so e conhecido "
        "para YoloBall e YoloField"
    )


def sorteia_clipes(
    registros: Sequence[ShotRecord],
    quantos: int,
    rng: random.Random,
    dobras: frozenset[int] = DOBRAS["ajuste"],
) -> tuple[tuple[ShotRecord, int], ...]:
    """Os clipes da calibracao e quantos quadros sai de cada um, espalhados pelo maximo de clipes.

    A ordem de partida e por `shot_id`, e nao a do catalogo, para a mesma semente dar a mesma
    amostra venha o catalogo na ordem que vier. Dobra fora do ajuste e recusada, e nao filtrada:
    pedir o reservado e erro de quem chamou, e filtrar calado esconderia o erro.
    """
    fora = sorted(dobras - DOBRAS["ajuste"])
    if fora:
        raise CalibracaoError(
            f"dobras {fora}: a calibracao so ve as de ajuste, {sorted(DOBRAS['ajuste'])}, porque "
            "o pacote reservado mede o que ninguem ajustou"
        )
    if quantos <= 0:
        raise CalibracaoError(f"quantos={quantos}: a calibracao precisa de ao menos um quadro")
    elegiveis = sorted(
        (r for r in registros if r.big_enough and r.fold in dobras), key=lambda r: r.shot_id
    )
    if not elegiveis:
        raise CalibracaoError(f"nenhum clipe abrivel nas dobras {sorted(dobras)}")
    ordem = rng.sample(elegiveis, len(elegiveis))
    base, resto = divmod(quantos, len(ordem))
    cotas = [base + (1 if posicao < resto else 0) for posicao in range(len(ordem))]
    return tuple((r, cota) for r, cota in zip(ordem, cotas, strict=True) if cota)


def sorteia_quadros(
    disponiveis: Sequence[int], quantos: int, rng: random.Random
) -> tuple[int, ...]:
    return tuple(sorted(rng.sample(list(disponiveis), min(quantos, len(disponiveis)))))


def extrai(
    nome: str,
    config: Path,
    destino: Path,
    *,
    quantos: int = QUADROS_PADRAO,
    semente: int = SEMENTE_PADRAO,
    registros: Sequence[ShotRecord] | None = None,
) -> dict[str, object]:
    """Grava um `.npy` por despacho do plano e o `calibracao.json` que diz de onde eles vieram.

    O nome de cada arquivo e a posicao na amostra, e nao o clipe: o tensor e derivado de video do
    parceiro e mora em `var/`, mas nem o nome dele carrega identificador da Footstats.
    """
    especie = especie_de(nome, config)
    fps = _fps_do_conjunto(config, especie.spec.frames)
    rng = random.Random(semente)
    sorteio = sorteia_clipes(load_catalog() if registros is None else registros, quantos, rng)
    destino.mkdir(parents=True, exist_ok=False)
    digests: list[str] = []
    formas: set[tuple[int, ...]] = set()
    quadros = 0
    for registro, cota in sorteio:
        for quadro in _quadros(registro, cota, fps, especie, rng):
            quadros += 1
            for tensor in especie.entradas(quadro):
                caminho = destino / f"{len(digests):04d}.npy"
                np.save(caminho, tensor)
                digests.append(sha256_of(caminho))
                formas.add(tuple(int(d) for d in tensor.shape))
    if len(formas) != 1:
        raise CalibracaoError(f"a amostra saiu com {len(formas)} shapes, e o plano tem um so")
    relatorio: dict[str, object] = {
        "especie": nome,
        "config_sha256": sha256_of(config),
        "frames": especie.spec.frames,
        "fps": fps,
        "preprocess": asdict(especie.spec.preprocess),
        "ajuste": especie.ajuste,
        "dobras": sorted(DOBRAS["ajuste"]),
        "semente": semente,
        "quadros_pedidos": quantos,
        "quadros": quadros,
        "clipes": len(sorteio),
        "shape": list(formas.pop()),
        "dtype": "float32",
        "sha256": digests,
        "versoes": {pacote: version(pacote) for pacote in ("numpy", "opencv-python-headless")},
        "extrator_sha256": sha256_of(Path(__file__)),
    }
    (destino / CALIBRACAO).write_text(json.dumps(relatorio, indent=2) + "\n", encoding="utf-8")
    return relatorio


def _fps_do_conjunto(config: Path, frames: str) -> float | None:
    conjunto = read_framesets(config).get(frames)
    if not isinstance(conjunto, UniformFrames):
        raise CalibracaoError(
            f"frames.{frames} nao e um conjunto uniform em {config}: uma janela depende do "
            "instante de outro modulo, e a calibracao nao roda o pipeline inteiro"
        )
    return conjunto.fps


def _quadros(
    registro: ShotRecord, cota: int, fps: float | None, especie: Especie, rng: random.Random
) -> Tensor:
    clip = probe(registro.clip_path, clip_id=registro.shot_id)
    indices = sorteia_quadros(uniform_indices(clip, fps), cota, rng)
    provider = FrameProvider(VideoSource(registro.clip_path))
    try:
        return provider.frames(indices, especie.spec.preprocess).tensor
    finally:
        provider.close()


class LeitorDeCalibracao:
    """O `CalibrationDataReader` do ONNX Runtime, lendo um tensor por vez do disco.

    Nao herda da classe do ONNX Runtime porque ela e dependencia opcional, e nao precisa: o
    `quantize_static` so chama `get_next`, e com `CalibStridedMinMax` tambem `__len__` e
    `set_range`.
    """

    def __init__(self, entrada: str, tensores: Sequence[Path]) -> None:
        self._entrada = entrada
        self._tensores = tuple(tensores)
        self._fila = iter(self._tensores)

    def get_next(self) -> dict[str, Tensor] | None:
        caminho = next(self._fila, None)
        return None if caminho is None else {self._entrada: np.load(caminho)}

    def __len__(self) -> int:
        return len(self._tensores)

    def set_range(self, start_index: int, end_index: int) -> None:
        self._fila = iter(self._tensores[start_index:end_index])


def le_calibracao(pasta: Path) -> tuple[tuple[Path, ...], list[str], dict[str, Any]]:
    """Os tensores da pasta, o sha256 de cada um e o relatorio da extracao, conferidos entre si.

    O relatorio completo da extracao e obrigatorio, com configuracao, versoes e dobras de ajuste.
    Contagens, shape e dtype precisam casar com os tensores, assim como o sha256 de cada arquivo.
    """
    relatorio_path = pasta / CALIBRACAO
    if not relatorio_path.is_file():
        raise CalibracaoError(
            f"{pasta} nao tem {CALIBRACAO}: extraia com `python -m mlops.quantize_int8 calibracao`"
        )
    relatorio = _read_calibration_report(relatorio_path)
    _validate_calibration_metadata(relatorio)
    tensores = tuple(sorted(pasta.glob("*.npy")))
    if not tensores:
        raise CalibracaoError(f"{pasta} nao tem tensor .npy nenhum")
    lidos = [sha256_of(t) for t in tensores]
    if lidos != relatorio.get("sha256"):
        raise CalibracaoError(f"os tensores de {pasta} nao sao os que a extracao gravou")
    _validate_sampling(relatorio, len(tensores))
    _check_calibration_tensors(tensores, relatorio)
    semente = relatorio.get("semente")
    if isinstance(semente, bool) or not isinstance(semente, int):
        raise CalibracaoError(f"{relatorio_path} nao declara a semente como inteiro")
    return tensores, lidos, relatorio


def _read_calibration_report(path: Path) -> dict[str, Any]:
    try:
        document: object = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as error:
        raise CalibracaoError(f"{path.name}: JSON invalido: {error}") from error
    if not isinstance(document, dict):
        raise CalibracaoError(f"{path.name}: o relatorio precisa ser um objeto JSON")
    report = cast("dict[str, Any]", document)
    missing = CALIBRATION_FIELDS - report.keys()
    if missing:
        raise CalibracaoError(f"{path.name}: faltam os campos {', '.join(sorted(missing))}")
    return report


def _validate_calibration_metadata(report: dict[str, Any]) -> None:
    if report["especie"] not in ESPECIES:
        raise CalibracaoError(f"especie precisa ser {' ou '.join(ESPECIES)}")
    for name in ("config_sha256", "extrator_sha256"):
        value = report[name]
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise CalibracaoError(f"{name} precisa ser um SHA-256 hexadecimal de 64 caracteres")
    _nonempty_text(report["frames"], "frames")
    fps = report["fps"]
    if fps is not None and (
        isinstance(fps, bool)
        or not isinstance(fps, int | float)
        or not math.isfinite(fps)
        or fps <= 0
    ):
        raise CalibracaoError("fps precisa ser nulo ou um numero finito positivo")
    _validate_preprocess(_json_object(report["preprocess"], "preprocess"))
    _json_object(report["ajuste"], "ajuste")
    if report["dtype"] != "float32":
        raise CalibracaoError("dtype precisa ser float32")
    shape = report["shape"]
    if not isinstance(shape, list) or len(cast("list[object]", shape)) != 4:
        raise CalibracaoError("shape precisa declarar as quatro dimensoes NCHW")
    for index, dimension in enumerate(cast("list[object]", shape)):
        _integer(dimension, f"shape[{index}]")
    versions = _json_object(report["versoes"], "versoes")
    for package in ("numpy", "opencv-python-headless"):
        _nonempty_text(versions.get(package), f"versoes.{package}")


def _validate_preprocess(preprocess: dict[str, Any]) -> None:
    for name in ("width", "height"):
        _integer(preprocess.get(name), f"preprocess.{name}")
    if preprocess.get("color") not in ("rgb", "bgr", "gray", "ycrcb"):
        raise CalibracaoError("preprocess.color precisa declarar uma cor conhecida")
    if not isinstance(preprocess.get("letterbox"), bool):
        raise CalibracaoError("preprocess.letterbox precisa ser booleano")
    if preprocess.get("normalize") not in ("none", "unit", "imagenet"):
        raise CalibracaoError("preprocess.normalize precisa declarar uma normalizacao conhecida")
    ops = preprocess.get("ops")
    if not isinstance(ops, list):
        raise CalibracaoError("preprocess.ops precisa ser uma lista")
    for op in cast("list[object]", ops):
        _nonempty_text(op, "preprocess.ops")


def _validate_sampling(report: dict[str, Any], tensors: int) -> None:
    folds = report["dobras"]
    if (
        not isinstance(folds, list)
        or any(isinstance(d, bool) or not isinstance(d, int) for d in cast("list[object]", folds))
        or folds != sorted(DOBRAS["ajuste"])
    ):
        raise CalibracaoError(
            "dobras precisa declarar somente o ajuste [0, 1, 2], nunca o reservado"
        )
    requested = _integer(report["quadros_pedidos"], "quadros_pedidos")
    frame_count = _integer(report["quadros"], "quadros")
    clip_count = _integer(report["clipes"], "clipes")
    if frame_count > requested:
        raise CalibracaoError("quadros nao pode exceder quadros_pedidos")
    if clip_count > frame_count:
        raise CalibracaoError("clipes nao pode exceder quadros")
    expected = frame_count * _dispatches_per_frame(report)
    if tensors != expected:
        raise CalibracaoError(f"a contagem declara {expected} tensores, mas a pasta tem {tensors}")


def _dispatches_per_frame(report: dict[str, Any]) -> int:
    settings = _json_object(report["ajuste"], "ajuste")
    if report["especie"] == "field":
        if settings or report["shape"][0] != 1:
            raise CalibracaoError("ajuste do campo e vazio e o lote de cada tensor tem tamanho 1")
        return 1
    for name in ("imgsz", "ladrilhos", "ladrilhos_por_lote"):
        _integer(settings.get(name), f"ajuste.{name}")
    _integer(settings.get("sobreposicao_px"), "ajuste.sobreposicao_px", minimum=0)
    if settings.get("letterbox") not in ("cv2", "cuda"):
        raise CalibracaoError("ajuste.letterbox precisa ser cv2 ou cuda")
    tile_count = settings["ladrilhos"] ** 2
    batch_size = min(tile_count, settings["ladrilhos_por_lote"])
    if tile_count % batch_size or report["shape"][0] != batch_size:
        raise CalibracaoError("ajuste dos ladrilhos nao corresponde ao lote declarado no shape")
    return tile_count // batch_size


def _json_object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CalibracaoError(f"{name} precisa ser um objeto JSON")
    return cast("dict[str, Any]", value)


def _nonempty_text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise CalibracaoError(f"{name} precisa ser texto nao vazio")


def _integer(value: object, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CalibracaoError(f"{name} precisa ser inteiro a partir de {minimum}")
    return value


def _check_calibration_tensors(tensors: Sequence[Path], report: dict[str, Any]) -> None:
    expected = tuple(report["shape"])
    for path in tensors:
        try:
            tensor = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError, EOFError) as error:
            raise CalibracaoError(f"{path.name}: tensor .npy invalido: {error}") from error
        if tensor.shape != expected or tensor.dtype != np.float32:
            raise CalibracaoError(
                f"{path.name} e {tensor.shape} {tensor.dtype}, e o relatorio declara "
                f"{expected} float32"
            )


def quantiza(onnx: Path, calibracao: Path, output: Path) -> dict[str, object]:
    """Grava o ONNX com Q/DQ e, ao lado, o `.run.json` que o amarra ao ONNX e a calibracao.

    O manifesto guarda sha256 e contagem, e nada que identifique clipe, arquivo ou partida.
    """
    if not onnx.is_file():
        raise CalibracaoError(f"{onnx} nao existe")
    tensores, lidos, relatorio = le_calibracao(calibracao)
    modulo_onnx = cast("Any", import_module("onnx"))
    modelo = modulo_onnx.load(str(onnx))
    entrada = _entrada_unica(modelo, onnx)
    _confere_shapes(tensores, entrada[1])
    bloco, cabeca = convolucoes_da_cabeca(modelo)
    ort = cast("Any", import_module("onnxruntime.quantization"))
    opcoes = opcoes_do_tensorrt()
    output.parent.mkdir(parents=True, exist_ok=True)
    ort.quantize_static(
        model_input=str(onnx),
        model_output=str(output),
        calibration_data_reader=LeitorDeCalibracao(entrada[0], tensores),
        quant_format=ort.QuantFormat.QDQ,
        op_types_to_quantize=list(OPERADORES),
        per_channel=True,
        activation_type=ort.QuantType.QInt8,
        weight_type=ort.QuantType.QInt8,
        calibrate_method=getattr(ort.CalibrationMethod, METODO),
        nodes_to_exclude=list(cabeca),
        extra_options=opcoes,
    )
    nos = contagem_qdq(modulo_onnx.load(str(output)))
    if not any(nos.values()):
        raise CalibracaoError(f"{output.name} saiu sem no Q/DQ: o ONNX de entrada tem Conv?")
    manifesto: dict[str, object] = {
        "onnx_sha256": sha256_of(onnx),
        "saida_sha256": sha256_of(output),
        "saida_bytes": output.stat().st_size,
        "especie": relatorio["especie"],
        "calibracao_config_sha256": relatorio["config_sha256"],
        "tensores": len(lidos),
        "tensores_sha256": lidos,
        "semente": relatorio["semente"],
        "metodo": METODO,
        "formato": "QDQ",
        "por_canal": True,
        "tipos": {"ativacao": "QInt8", "peso": "QInt8"},
        "operadores": list(OPERADORES),
        "fora_do_int8": {"bloco": bloco, "convolucoes": len(cabeca)},
        "opcoes": opcoes,
        "nos_qdq": nos,
        "onnxruntime": version("onnxruntime"),
        "onnx": version("onnx"),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "quantizador_sha256": sha256_of(Path(__file__)),
    }
    output.with_suffix(".run.json").write_text(
        json.dumps(manifesto, indent=2) + "\n", encoding="utf-8"
    )
    return manifesto


def convolucoes_da_cabeca(modelo: Any) -> tuple[str | None, tuple[str, ...]]:
    """O ultimo bloco do Ultralytics, a cabeca, e as convolucoes dele, que ficam fora do INT8.

    Medido no campo, no ONNX Runtime em CPU contra o fp32, com 64 quadros de calibracao e 16 de
    ajuste sorteados com outra semente: com a cabeca quantizada o keypoint desloca 12,1 px na
    mediana e 34,6 px no p95; com as 34 convolucoes de `/model.23/` fora, 1,4 e 5,5. Tirar a
    primeira convolucao nao mudou nada, 12,8 e 35,5. Na bola, com 16 quadros de calibracao e 8 de
    teste, das 75 deteccoes do fp32 a conf 0,10, 68 casam a 4 px com a cabeca fora e 19 com ela
    dentro. As duas medidas deixam o resto da rede em fp32 e nao passam pelo TensorRT.

    O nome do bloco vem do exportador do Ultralytics, `/model.<bloco>/`. ONNX sem esse padrao sai
    sem exclusao nenhuma, e o manifesto diz isso com `bloco` nulo.
    """
    blocos = [int(achado.group(1)) for no in modelo.graph.node if (achado := BLOCO.match(no.name))]
    if not blocos:
        return None, ()
    prefixo = f"/model.{max(blocos)}/"
    nomes = tuple(
        str(no.name)
        for no in modelo.graph.node
        if no.op_type == "Conv" and str(no.name).startswith(prefixo)
    )
    return prefixo, nomes


def contagem_qdq(modelo: Any) -> dict[str, int]:
    return {op: sum(1 for no in modelo.graph.node if no.op_type == op) for op in QDQ}


def _entrada_unica(modelo: Any, onnx: Path) -> tuple[str, tuple[int, ...]]:
    iniciais = {inicial.name for inicial in modelo.graph.initializer}
    entradas = [e for e in modelo.graph.input if e.name not in iniciais]
    if len(entradas) != 1:
        raise CalibracaoError(f"{onnx.name} tem {len(entradas)} entradas, e o plano recebe uma")
    dims = tuple(int(d.dim_value) for d in entradas[0].type.tensor_type.shape.dim)
    return str(entradas[0].name), dims


def _confere_shapes(tensores: Sequence[Path], esperado: tuple[int, ...]) -> None:
    """Recusa antes do ONNX Runtime o tensor que nao casa com a entrada, dizendo qual e por que."""
    for caminho in tensores:
        tensor = np.load(caminho, mmap_mode="r")
        forma = tuple(int(d) for d in tensor.shape)
        casa = len(forma) == len(esperado) and all(
            e in (0, f) for f, e in zip(forma, esperado, strict=True)
        )
        if not casa or tensor.dtype != np.float32:
            raise CalibracaoError(
                f"{caminho.name} e {forma} {tensor.dtype}, e o ONNX pede {esperado} float32"
            )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    comandos = parser.add_subparsers(dest="comando", required=True)
    calibracao = comandos.add_parser("calibracao", help="extrai os tensores das dobras de ajuste")
    calibracao.add_argument("--especie", choices=ESPECIES, required=True)
    calibracao.add_argument("--config", type=Path, required=True)
    calibracao.add_argument("--output", type=Path, default=None, help="padrao: var/calibracao/")
    calibracao.add_argument("--quadros", type=int, default=QUADROS_PADRAO)
    calibracao.add_argument("--semente", type=int, default=SEMENTE_PADRAO)
    quantizacao = comandos.add_parser("quantiza", help="grava o ONNX com Q/DQ e o manifesto")
    quantizacao.add_argument("--onnx", type=Path, required=True)
    quantizacao.add_argument("--calibracao", type=Path, required=True)
    quantizacao.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.comando == "calibracao":
        destino = args.output or var_dir() / "calibracao" / args.especie
        relatorio = extrai(
            args.especie, args.config, destino, quantos=args.quadros, semente=args.semente
        )
    else:
        relatorio = quantiza(args.onnx, args.calibracao, args.output)
    print(json.dumps(relatorio, indent=2))


if __name__ == "__main__":
    main()
