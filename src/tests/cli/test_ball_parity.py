"""`xvision eval-ball-paridade`: dois runtimes da bola no contato, sem GPU e sem video."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner, Result

from cli import app
from contracts import BallDetection, Box, Clip
from edge.graph.framesets import UniformFrames
from edge.modules.base import FrameBatch, ModuleInputs, ModuleOutput, ModuleSpec, PreprocessSpec
from edge.runtime import ArtifactIncompatible, RuntimeInfo
from mlops.catalog import ShotRecord

PEQUENO = PreprocessSpec(64, 64, "bgr", True, "none")


def _info(backend: str) -> RuntimeInfo:
    return RuntimeInfo(
        backend="tensorrt" if backend == "tensorrt" else "pytorch",
        provider=backend,
        device="cuda",
        precision="float16",
        runtime_versions=((backend, "1.0"),),
        artifact_sha256="ab" * 32,
        parent_onnx_sha256=None,
        inputs=(),
        outputs=(),
        inference_count=0,
        fallback_reason=None,
    )


def _caixa(x: float, y: float) -> Box:
    return Box(x - 2, y - 2, x + 2, y + 2)


class _Bola:
    """A bola num x fixo e um segundo candidato parado, com runtime e falha lidos da config."""

    def __init__(self, config: dict[str, Any], eventos: list[str]) -> None:
        self.nome = str(config["nome"])
        self.x = float(config["x"])
        self.config = config
        self.eventos = eventos
        self.spec = ModuleSpec(
            kind="ball", name=self.nome, version="1", frames="dense", preprocess=PEQUENO
        )

    @property
    def runtime_info(self) -> RuntimeInfo | None:
        return _info(self.nome) if self.config.get("runtime", True) else None

    def load(self) -> None:
        if "falha" in self.config:
            raise ArtifactIncompatible(str(self.config["falha"]))
        self.eventos.append(f"load:{self.nome}")

    def close(self) -> None:
        self.eventos.append(f"close:{self.nome}")

    def infer(self, batch: FrameBatch, ctx: ModuleInputs) -> ModuleOutput:
        caixa = _caixa(self.x, 30.0)
        registro = {"box": (caixa.x1, caixa.y1, caixa.x2, caixa.y2), "confidence": 0.9}
        return ModuleOutput(
            kind="ball",
            frames=batch.indices,
            records=tuple(({**registro, "source": "detected"},) for _ in batch.indices),
        )

    def candidatos(self, batch: FrameBatch) -> tuple[tuple[BallDetection, ...], ...]:
        return tuple(
            (BallDetection(_caixa(self.x, 30.0), 0.9), BallDetection(_caixa(50.0, 50.0), 0.4))
            for _ in batch.indices
        )


class _Fonte:
    fps = 30.0

    def __init__(self, path: Path) -> None: ...

    def read(self, indices: Sequence[int]) -> dict[int, Any]:
        return {i: np.zeros((64, 64, 3), dtype=np.uint8) for i in indices}

    def close(self) -> None: ...


class Paridade:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.tmp_path = tmp_path
        self.eventos: list[str] = []
        registros = [
            ShotRecord(c, c, tmp_path / f"{c}.mp4", True, 0, {})
            for c in ("synthetic-a", "synthetic-b")
        ]

        def contatos(_: object) -> dict[str, int]:
            return {"synthetic-a": 2, "synthetic-b": 3}

        def probe(path: Path, *, clip_id: str) -> Clip:
            return Clip(clip_id, str(path), 30.0, 5, 64, 64)

        def config(path: Path) -> dict[str, Any]:
            return json.loads(path.read_text())

        def modulos(config: dict[str, Any]) -> dict[str, _Bola]:
            return {"ball": _Bola(config, self.eventos)}

        def quadros(path: Path) -> dict[str, UniformFrames]:
            return {"dense": UniformFrames(fps=json.loads(path.read_text()).get("fps"))}

        monkeypatch.setattr("cli.read", lambda: ())
        monkeypatch.setattr("cli.consensus", contatos)
        monkeypatch.setattr("cli.load_catalog", lambda: tuple(registros))
        monkeypatch.setattr("cli.read_config", config)
        monkeypatch.setattr("cli.load_modules", modulos)
        monkeypatch.setattr("cli.read_framesets", quadros)
        monkeypatch.setattr("cli.probe", probe)
        monkeypatch.setattr("cli.VideoSource", _Fonte)

    def run(self, referencia: dict[str, Any], comparada: dict[str, Any]) -> Result:
        caminhos: list[str] = []
        for nome, config in (("pytorch", referencia), ("tensorrt", comparada)):
            caminho = self.tmp_path / f"{nome}.toml"
            caminho.write_text(json.dumps({"nome": nome, **config}))
            caminhos.append(str(caminho))
        return CliRunner().invoke(
            app,
            [
                "eval-ball-paridade",
                "--config",
                caminhos[0],
                "--contra",
                caminhos[1],
                "--saida",
                str(self.tmp_path / "paridade.json"),
            ],
        )


def test_runtimes_iguais_no_contato_passam_no_criterio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paridade = Paridade(tmp_path, monkeypatch)

    resultado = paridade.run({"x": 20.0}, {"x": 20.5})

    assert resultado.exit_code == 0, resultado.output
    assert "referencia:" in resultado.output and "pytorch cuda float16" in resultado.output
    assert "comparada:" in resultado.output and "tensorrt cuda float16" in resultado.output
    assert "4 pares a ate 4 px, 0 so na referencia, 0 so na comparada" in resultado.output
    assert "ponto escolhido mudou em 0 de 2 contatos" in resultado.output
    assert resultado.output.rstrip().splitlines()[-2].endswith(": passa")
    salvo = json.loads((tmp_path / "paridade.json").read_text())
    assert salvo["resumo"]["centro_distancia_max_px"] == pytest.approx(0.5)
    assert salvo["por_contato"]["synthetic-a"]["frame_index"] == 2
    assert salvo["por_contato"]["synthetic-b"]["frame_index"] == 3
    assert salvo["runtimes"]["comparada"]["backend"] == "tensorrt"
    assert sorted(paridade.eventos) == sorted(
        ["load:pytorch", "load:tensorrt", "close:pytorch", "close:tensorrt"]
    )


def test_bola_trocada_no_contato_reprova_e_diz_quantos_contatos_mudaram(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resultado = Paridade(tmp_path, monkeypatch).run({"x": 20.0}, {"x": 30.0})

    assert resultado.exit_code == 0, resultado.output
    assert "2 pares a ate 4 px, 2 so na referencia, 2 so na comparada" in resultado.output
    assert "ponto escolhido mudou em 2 de 2 contatos" in resultado.output
    assert "nao passa" in resultado.output


def test_plano_que_nao_abre_encerra_com_o_arquivo_e_fecha_o_que_abriu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paridade = Paridade(tmp_path, monkeypatch)

    resultado = paridade.run({"x": 20.0}, {"x": 20.0, "falha": "plano de outra versao"})

    assert resultado.exit_code == 1
    assert "tensorrt.toml: a bola nao carregou" in resultado.output
    assert "ArtifactIncompatible: plano de outra versao" in resultado.output
    assert paridade.eventos == ["load:pytorch", "close:pytorch"]
    assert not (tmp_path / "paridade.json").exists()


def test_bola_sem_runtime_declarado_e_recusada(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sem detector o modulo sai sem candidato, e dois lados vazios passariam por paridade."""
    paridade = Paridade(tmp_path, monkeypatch)

    resultado = paridade.run({"x": 20.0}, {"x": 20.0, "runtime": False})

    assert resultado.exit_code == 1
    assert "sem runtime declarado" in resultado.output
    assert sorted(paridade.eventos) == sorted(
        ["load:pytorch", "load:tensorrt", "close:pytorch", "close:tensorrt"]
    )


def test_configuracoes_com_quadros_diferentes_sao_recusadas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resultado = Paridade(tmp_path, monkeypatch).run({"x": 20.0}, {"x": 20.0, "fps": 15.0})

    assert resultado.exit_code == 1
    assert "mesmos quadros" in resultado.output
