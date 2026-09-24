"""O plano TensorRT sai com o sha256 do ONNX que o gerou, sem precisar de GPU no teste."""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from edge.artifacts import sha256_of
from mlops import build_engine
from mlops.build_engine import EngineBuildError, build


def _trt(
    parse_ok: bool = True, plano: bytes | None = b"plano", flag_fp16: bool = True
) -> tuple[SimpleNamespace, dict[str, Any]]:
    visto: dict[str, Any] = {"flags": []}

    class Builder:
        def __init__(self, logger: object) -> None: ...

        def create_network(self, flags: int) -> SimpleNamespace:
            tensor = SimpleNamespace(name="images", shape=(4, 3, 992, 1600))
            saida = SimpleNamespace(name="output0", shape=(4, 5, 32550))

            def entrada(i: int) -> SimpleNamespace:
                return tensor

            def resultado(i: int) -> SimpleNamespace:
                return saida

            return SimpleNamespace(
                num_inputs=1, get_input=entrada, num_outputs=1, get_output=resultado
            )

        def create_builder_config(self) -> SimpleNamespace:
            def limite(tipo: str, valor: int) -> None:
                visto["workspace"] = valor

            def flag(nome: str) -> None:
                visto["flags"].append(nome)

            return SimpleNamespace(set_memory_pool_limit=limite, set_flag=flag)

        def build_serialized_network(self, network: object, config: object) -> bytes | None:
            return plano

    class Parser:
        num_errors = 1

        def __init__(self, network: object, logger: object) -> None: ...

        def parse(self, dados: bytes) -> bool:
            visto["onnx"] = dados
            return parse_ok

        def get_error(self, i: int) -> str:
            return "no de entrada sem shape"

    class Logger:
        WARNING = 2

        def __init__(self, nivel: int) -> None: ...

    modulo = SimpleNamespace(
        Logger=Logger,
        Builder=Builder,
        OnnxParser=Parser,
        MemoryPoolType=SimpleNamespace(WORKSPACE="workspace"),
        BuilderFlag=SimpleNamespace(FP16="fp16") if flag_fp16 else SimpleNamespace(TF32="tf32"),
        __version__="11.3.0.99",
    )
    return modulo, visto


@pytest.fixture
def onnx(tmp_path: Path) -> Path:
    caminho = tmp_path / "ball-b4.onnx"
    caminho.write_bytes(b"onnx de teste")
    return caminho


def _instala(monkeypatch: pytest.MonkeyPatch, modulo: SimpleNamespace, pacote: str) -> None:
    def carrega(caminho: str) -> SimpleNamespace:
        saida = SimpleNamespace(name="output0")
        no = SimpleNamespace(output=["interno", "output0"])
        grafo = SimpleNamespace(output=[saida], node=[no])
        return SimpleNamespace(texto=Path(caminho).read_text(), graph=grafo, tipo="fp32")

    def salva(modelo: SimpleNamespace, caminho: str) -> None:
        nomes = ",".join(modelo.graph.node[0].output)
        Path(caminho).write_text(
            f"{modelo.texto}|{modelo.tipo}|{modelo.graph.output[0].name}|{nomes}"
        )

    def converte(modelo: SimpleNamespace, keep_io_types: bool) -> SimpleNamespace:
        assert keep_io_types
        modelo.tipo = "fp16"
        return modelo

    modulos = {
        "tensorrt": modulo,
        "onnx": SimpleNamespace(load=carrega, save=salva),
        "onnxconverter_common.float16": SimpleNamespace(convert_float_to_float16=converte),
    }

    def importa(nome: str) -> SimpleNamespace:
        return modulos[nome]

    monkeypatch.setattr(build_engine, "import_module", importa)

    def versao(nome: str) -> str:
        if nome == "onnxconverter-common":
            return "1.16.0"
        if nome != pacote:
            raise PackageNotFoundError(nome)
        return "11.3.0.99"

    monkeypatch.setattr(build_engine, "version", versao)


def test_o_plano_sai_com_os_dois_sha256_e_a_versao(
    onnx: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    modulo, visto = _trt()
    _instala(monkeypatch, modulo, "tensorrt-cu13")
    destino = tmp_path / "engines" / "ball-b4.engine"

    relatorio = build(onnx, destino, fp16=True, workspace_gib=2)

    assert destino.read_bytes() == b"plano"
    assert relatorio["onnx_sha256"] == sha256_of(onnx)
    assert relatorio["engine_sha256"] == sha256_of(destino)
    assert relatorio["tensorrt"] == "11.3.0.99"
    assert relatorio["inputs"] == [{"name": "images", "shape": [4, 3, 992, 1600]}]
    assert visto["flags"] == ["fp16"]
    assert visto["workspace"] == 2 << 30
    assert visto["onnx"] == b"onnx de teste"
    assert json.loads(destino.with_suffix(".run.json").read_text()) == relatorio


def test_sem_fp16_nenhuma_flag_de_precisao_e_ligada(
    onnx: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    modulo, visto = _trt()
    _instala(monkeypatch, modulo, "tensorrt")

    build(onnx, tmp_path / "field.engine", fp16=False)

    assert visto["flags"] == []


def test_onnx_recusado_pelo_parser_nao_gera_plano(
    onnx: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    modulo, _ = _trt(parse_ok=False)
    _instala(monkeypatch, modulo, "tensorrt-cu13")

    with pytest.raises(EngineBuildError, match="no de entrada sem shape"):
        build(onnx, tmp_path / "x.engine", fp16=True)
    assert not (tmp_path / "x.engine").exists()


def test_construcao_que_falha_nao_deixa_arquivo(
    onnx: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    modulo, _ = _trt(plano=None)
    _instala(monkeypatch, modulo, "tensorrt-cu13")

    with pytest.raises(EngineBuildError, match="nao construiu"):
        build(onnx, tmp_path / "x.engine", fp16=True)
    assert not (tmp_path / "x.engine").exists()


def test_tensorrt_sem_pacote_instalado_e_recusado(
    onnx: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    modulo, _ = _trt()
    _instala(monkeypatch, modulo, "outro-pacote")

    with pytest.raises(EngineBuildError, match="sem pacote"):
        build(onnx, tmp_path / "x.engine", fp16=True)


@pytest.mark.parametrize(
    ("nome", "workspace", "erro"), [("nada.onnx", 1, "nao existe"), (None, 0, "positivo")]
)
def test_entrada_invalida_falha_antes_do_tensorrt(
    onnx: Path, tmp_path: Path, nome: str | None, workspace: float, erro: str
) -> None:
    alvo = tmp_path / nome if nome else onnx
    with pytest.raises(EngineBuildError, match=erro):
        build(alvo, tmp_path / "x.engine", fp16=True, workspace_gib=workspace)


def test_sem_a_flag_fp16_o_onnx_e_convertido_antes_do_parser(
    onnx: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    modulo, visto = _trt(flag_fp16=False)
    _instala(monkeypatch, modulo, "tensorrt-cu13")
    destino = tmp_path / "ball-b4.engine"

    relatorio = build(onnx, destino, fp16=True)

    trabalho = destino.with_suffix(".trabalho.onnx")
    assert visto["flags"] == []
    assert visto["onnx"] == trabalho.read_bytes() == b"onnx de teste|fp16|output0|interno,output0"
    assert relatorio["precisao"] == "fp16 pelo ONNX convertido"
    assert relatorio["onnx_sha256"] == sha256_of(onnx)
    assert relatorio["onnx_trabalho_sha256"] == sha256_of(trabalho)
    assert relatorio["conversor"] == "onnxconverter-common 1.16.0"


def test_a_saida_e_renomeada_no_onnx_de_trabalho_e_nao_no_publicado(
    onnx: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    modulo, visto = _trt(flag_fp16=False)
    _instala(monkeypatch, modulo, "tensorrt-cu13")
    original = onnx.read_bytes()

    relatorio = build(onnx, tmp_path / "field.engine", fp16=True, saida="predictions")

    assert relatorio["saida_renomeada"] == "output0 -> predictions"
    assert visto["onnx"] == b"onnx de teste|fp16|predictions|interno,predictions"
    assert onnx.read_bytes() == original


def test_com_a_flag_fp16_so_renomear_ainda_gera_onnx_de_trabalho(
    onnx: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    modulo, visto = _trt()
    _instala(monkeypatch, modulo, "tensorrt")

    relatorio = build(onnx, tmp_path / "field.engine", fp16=True, saida="predictions")

    assert visto["flags"] == ["fp16"]
    assert visto["onnx"] == b"onnx de teste|fp32|predictions|interno,predictions"
    assert "conversor" not in relatorio
