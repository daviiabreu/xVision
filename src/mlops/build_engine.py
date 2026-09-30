"""Constroi um plano TensorRT a partir de um ONNX, com os dois sha256 registrados.

O plano nao entra no Git: tem de 85 a 190 MB e nao atravessa versao de TensorRT nem placa. O que
entra e a configuracao que aponta para ele com `engine_onnx_sha256`, e esse numero so vale se o
plano tiver saido exatamente daquele ONNX. Por isso a construcao e um comando do repositorio, que
le o ONNX, calcula o sha256 dele e grava tudo num `run.json` ao lado do plano, em vez de um
`trtexec` digitado a mao que ninguem consegue reconstituir depois.

Constroi com o binding Python do TensorRT instalado no ambiente, e nao com o `trtexec` do sistema,
porque o plano so abre na mesma versao que o gerou, e quem abre e o binding do ambiente.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

from edge.artifacts import sha256_of

GIB = 1 << 30


class EngineBuildError(RuntimeError): ...


def build(
    onnx: Path,
    output: Path,
    *,
    fp16: bool,
    workspace_gib: float = 1.0,
    saida: str | None = None,
) -> dict[str, object]:
    if not onnx.is_file():
        raise EngineBuildError(f"{onnx} nao existe")
    if workspace_gib <= 0:
        raise EngineBuildError("workspace precisa ser positivo")
    trt = cast("Any", import_module("tensorrt"))
    por_flag = hasattr(trt.BuilderFlag, "FP16")
    modelo = cast("Any", import_module("onnx")).load(str(onnx))
    qdq = tem_qdq(modelo)
    converte = fp16 and (qdq or not por_flag)
    entrada = onnx
    conversao: dict[str, object] = {}
    if converte or saida is not None:
        entrada, conversao = _prepara_onnx(
            modelo, output, fp16=converte, saida=saida, qdq=qdq, origem=onnx.name
        )
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(_criacao_da_rede(trt, qdq=qdq, por_flag=por_flag))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(entrada.read_bytes()):
        erros = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise EngineBuildError(f"o parser do TensorRT recusou {entrada.name}: {'; '.join(erros)}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gib * GIB))
    if fp16 and por_flag and not qdq:
        config.set_flag(trt.BuilderFlag.FP16)
    inicio = time.perf_counter()
    plano = builder.build_serialized_network(network, config)
    duracao = time.perf_counter() - inicio
    if plano is None:
        raise EngineBuildError(f"o TensorRT nao construiu o plano de {onnx.name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(bytes(plano))
    report: dict[str, object] = {
        "onnx": str(onnx),
        "onnx_sha256": sha256_of(onnx),
        "engine": str(output),
        "engine_sha256": sha256_of(output),
        "engine_bytes": output.stat().st_size,
        "fp16": fp16,
        "precisao": _precisao(fp16, por_flag=por_flag, int8=_quantizado(trt, network)),
        **conversao,
        "workspace_gib": workspace_gib,
        "inputs": _tensores(network.num_inputs, network.get_input),
        "outputs": _tensores(network.num_outputs, network.get_output),
        "build_s": round(duracao, 1),
        "tensorrt": version(_distribuicao(trt)),
        "python": platform.python_version(),
        "machine": platform.machine(),
    }
    output.with_suffix(".run.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _prepara_onnx(
    modelo: Any, output: Path, *, fp16: bool, saida: str | None, qdq: bool, origem: str
) -> tuple[Path, dict[str, object]]:
    """O ONNX de trabalho: convertido para float16 e com a saida renomeada, conforme o pedido.

    O TensorRT 11 tirou `BuilderFlag.FP16`: toda rede passou a ter tipo fixo, e a precisao sai do
    que o ONNX declara. A conversao mantem entrada e saida em float32, para o plano falar o mesmo
    tensor que o detector ja monta.

    O nome da saida vem do exportador do Ultralytics, `output0`, e o contrato dos detectores chama o
    binding de `predictions`. Renomear aqui, e nao no ONNX publicado, mantem o artefato versionado
    identico ao que o manifesto descreve. O relatorio guarda o sha256 dos dois ONNX para o preparo
    poder ser repetido e conferido.
    """
    modulo_onnx = cast("Any", import_module("onnx"))
    info: dict[str, object] = {}
    if fp16:
        if qdq:
            modelo, info["opset"] = _opset_da_escala_fp16(modulo_onnx, modelo)
        float16 = cast("Any", import_module("onnxconverter_common.float16"))
        modelo = float16.convert_float_to_float16(modelo, keep_io_types=True)
        info["conversor"] = f"onnxconverter-common {version('onnxconverter-common')}"
    if saida is not None:
        saidas = list(modelo.graph.output)
        if len(saidas) != 1:
            raise EngineBuildError(f"renomear exige uma saida so, e {origem} tem {len(saidas)}")
        antigo = saidas[0].name
        for no in modelo.graph.node:
            no.output[:] = [saida if nome == antigo else nome for nome in no.output]
        saidas[0].name = saida
        info["saida_renomeada"] = f"{antigo} -> {saida}"
    destino = output.with_suffix(".trabalho.onnx")
    destino.parent.mkdir(parents=True, exist_ok=True)
    modulo_onnx.save(modelo, str(destino))
    info.update({"onnx_trabalho": str(destino), "onnx_trabalho_sha256": sha256_of(destino)})
    return destino, info


def _precisao(fp16: bool, *, por_flag: bool, int8: bool) -> str:
    """A precisao do plano, dita pelo que a construcao realmente usou.

    INT8 no TensorRT 11 nao tem flag: vem dos nos Q/DQ que o ONNX ja traz, e as demais camadas ficam
    nos tipos que o ONNX declara. Chamar esse plano de fp32 so porque nenhuma flag foi ligada
    descreveria um plano que nao existe.
    """
    if int8:
        resto = "em fp16 pelo ONNX convertido" if fp16 else "nos tipos do ONNX"
        return f"int8 pelos nos Q/DQ do ONNX, demais camadas {resto}"
    if fp16:
        return "fp16 por flag" if por_flag else "fp16 pelo ONNX convertido"
    return "fp32"


QDQ = frozenset({"QuantizeLinear", "DequantizeLinear"})
OPSET_ESCALA_FP16 = 19


def tem_qdq(modelo: Any) -> bool:
    return any(no.op_type in QDQ for no in modelo.graph.node)


def _criacao_da_rede(trt: Any, *, qdq: bool, por_flag: bool) -> int:
    """Rede de tipo fixo para o ONNX com Q/DQ, e a criacao de sempre para o resto.

    Com Q/DQ a precisao de cada camada tem de vir do ONNX, que e o que a rede de tipo fixo faz, e
    `STRONGLY_TYPED` pede isso explicitamente em vez de depender do padrao da versao. Rede de tipo
    fixo recusa flag de precisao, por isso a de fp16 nunca e ligada neste caminho. Onde a flag de
    criacao nao existe e a de fp16 tambem nao, toda rede ja e de tipo fixo.
    """
    if not qdq:
        return 0
    tipado = getattr(getattr(trt, "NetworkDefinitionCreationFlag", None), "STRONGLY_TYPED", None)
    if tipado is not None:
        return 1 << int(tipado)
    if not por_flag:
        return 0
    raise EngineBuildError(
        f"TensorRT {getattr(trt, '__version__', '?')} nao cria rede de tipo fixo, e o INT8 por "
        "Q/DQ depende dela"
    )


def _opset_da_escala_fp16(modulo_onnx: Any, modelo: Any) -> tuple[Any, str]:
    """Sobe o opset para 19 antes da conversao, porque so dali a escala de Q/DQ pode ser fp16.

    O conversor de fp16 troca toda constante float, inclusive a escala dos QuantizeLinear e
    DequantizeLinear, e ate o opset 18 essa escala so pode ser float32: o ONNX sairia invalido pela
    especificacao. Com escala fp16 o DequantizeLinear entrega fp16 e o plano fica em fp16 com as
    convolucoes em INT8, o formato do INT8 do Danilo que o TensorRT 11.3 construiu e rodou.

    Esse ONNX nao se confere no ONNX Runtime em CPU. Medido no 1.23.2, em macOS arm64, o
    QuantizeLinear de fp16 dele devolve um a menos que o arredondamento em 30% dos elementos, todos
    com parte fracionaria acima de meio: arredonda para baixo em vez de para o mais proximo. O vies
    se acumula e o campo perde a instancia nos 16 quadros de teste, enquanto a mesma quantizacao
    com o resto em fp32 desloca o keypoint 1,4 px. Quem confere a variante fp16 e o TensorRT.
    """
    atual = next(int(o.version) for o in modelo.opset_import if o.domain in ("", "ai.onnx"))
    if atual >= OPSET_ESCALA_FP16:
        return modelo, str(atual)
    convertido = modulo_onnx.version_converter.convert_version(modelo, OPSET_ESCALA_FP16)
    return convertido, f"{atual} -> {OPSET_ESCALA_FP16}"


def _quantizado(trt: Any, network: Any) -> bool:
    quantize = getattr(getattr(trt, "LayerType", None), "QUANTIZE", None)
    if quantize is None:
        return False
    return any(network.get_layer(i).type == quantize for i in range(network.num_layers))


def _tensores(quantos: int, pega: Any) -> list[dict[str, object]]:
    return [{"name": pega(i).name, "shape": list(pega(i).shape)} for i in range(quantos)]


def _distribuicao(trt: Any) -> str:
    """O nome da distribuicao que trouxe o modulo, para a versao sair do pacote instalado.

    `tensorrt.__version__` e o que o binding declara; o pacote pode vir como `tensorrt`, como
    `tensorrt-cu13` ou como `python3-libnvinfer` do apt, e so o primeiro existe sempre no pip.
    """
    for nome in ("tensorrt-cu13", "tensorrt-cu12", "tensorrt"):
        try:
            version(nome)
        except PackageNotFoundError:
            continue
        return nome
    raise EngineBuildError(f"TensorRT {getattr(trt, '__version__', '?')} sem pacote instalado")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--workspace-gib", type=float, default=1.0)
    parser.add_argument("--saida", default=None, help="novo nome da unica saida do grafo")
    args = parser.parse_args()
    relatorio = build(
        args.onnx,
        args.output,
        fp16=args.fp16,
        workspace_gib=args.workspace_gib,
        saida=args.saida,
    )
    print(json.dumps(relatorio, indent=2))


if __name__ == "__main__":
    main()
