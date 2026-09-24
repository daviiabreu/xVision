"""As duas etapas do instante: gravar so o `shot`, e depois ler o instante gravado."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from edge.graph.framesets import UniformFrames
from edge.modules.base import (
    FrameBatch,
    ModuleError,
    ModuleInputs,
    ModuleNotLoaded,
    ModuleOutput,
    ModuleSpec,
    PerceptionModule,
    PreprocessSpec,
)
from edge.modules.conformance import ModuleConformance
from edge.testing.make_clip import make_synthetic_clip
from mlops.catalog import ShotRecord
from mlops.spots import PREPROCESS, PrecomputedShot, SpotsError, gera, lidos

LOTE = PreprocessSpec(width=64, height=36, color="bgr", letterbox=False, normalize="none")


class ShotFalso:
    """Aponta o quadro do meio e registra o que recebeu, para o teste conferir o recorte."""

    spec = ModuleSpec(kind="shot", name="falso", version="1", frames="coarse", preprocess=LOTE)

    def __init__(self, falha_em: str | None = None) -> None:
        self.falha_em = falha_em
        self.vistos: list[tuple[str | None, tuple[int, ...]]] = []
        self.aberto = False

    def load(self) -> None:
        self.aberto = True

    def close(self) -> None:
        self.aberto = False

    def infer(self, batch: FrameBatch, ctx: ModuleInputs) -> ModuleOutput:
        self.vistos.append((batch.clip_id, batch.indices))
        if batch.clip_id == self.falha_em:
            raise ModuleError("sem memoria")
        meio = batch.indices[len(batch.indices) // 2]
        registros = tuple(
            ({"frame_index": meio, "confidence": 0.7, "source": "falso"},) if i == meio else ()
            for i in batch.indices
        )
        return ModuleOutput(kind="shot", frames=batch.indices, records=registros)


@pytest.fixture(scope="module")
def video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_synthetic_clip(tmp_path_factory.mktemp("spots") / "sintetico.mp4")


def registro(clip_id: str, video: Path) -> ShotRecord:
    return ShotRecord(clip_id, "m0", video, True, 0, {})


def test_a_primeira_etapa_grava_um_instante_por_clipe(video: Path, tmp_path: Path) -> None:
    destino = tmp_path / "spots" / "falso.jsonl"
    modulo = ShotFalso()

    resumo = gera([registro("c1", video), registro("c2", video)], modulo, UniformFrames(5), destino)

    assert (resumo.feitos, resumo.pulados, resumo.falhas) == (2, 0, ())
    gravados = lidos(destino)
    assert set(gravados) == {"c1", "c2"}
    assert gravados["c1"]["source"] == "falso"
    assert gravados["c1"]["duration_ms"] >= 0
    assert [clip for clip, _ in modulo.vistos] == ["c1", "c2"]
    assert not modulo.aberto


def test_a_primeira_etapa_retoma_sem_refazer_o_que_ja_gravou(video: Path, tmp_path: Path) -> None:
    destino = tmp_path / "falso.jsonl"
    gera([registro("c1", video)], ShotFalso(), UniformFrames(5), destino)
    modulo = ShotFalso()

    resumo = gera([registro("c1", video), registro("c2", video)], modulo, UniformFrames(5), destino)

    assert (resumo.feitos, resumo.pulados) == (1, 1)
    assert [clip for clip, _ in modulo.vistos] == ["c2"]


def test_falha_de_um_clipe_fica_no_resumo_e_nao_derruba_os_outros(
    video: Path, tmp_path: Path
) -> None:
    destino = tmp_path / "falso.jsonl"

    resumo = gera(
        [registro("c1", video), registro("c2", video)],
        ShotFalso(falha_em="c1"),
        UniformFrames(5),
        destino,
    )

    assert resumo.feitos == 1
    assert resumo.falhas == (("c1", "ModuleError: sem memoria"),)
    assert set(lidos(destino)) == {"c2"}


def test_a_primeira_etapa_so_aceita_a_especie_shot(video: Path, tmp_path: Path) -> None:
    class Bola(ShotFalso):
        spec = ModuleSpec(kind="ball", name="b", version="1", frames="coarse", preprocess=LOTE)

    with pytest.raises(SpotsError, match="especie shot"):
        gera([registro("c1", video)], Bola(), UniformFrames(5), tmp_path / "x.jsonl")


@pytest.mark.parametrize(
    ("linhas", "erro"),
    [
        (['{"clip_id": "c1", "frame_index": 3}', '{"clip_id": "c1", "frame_index": 4}'], "duas"),
        (['{"clip_id": "c1", "frame_index": "3"}'], "inteiro"),
        (['{"frame_index": 3}'], "inteiro"),
    ],
)
def test_arquivo_invalido_e_recusado(tmp_path: Path, linhas: list[str], erro: str) -> None:
    caminho = tmp_path / "spots.jsonl"
    caminho.write_text("\n".join(linhas) + "\n")

    with pytest.raises(SpotsError, match=erro):
        lidos(caminho)


def test_arquivo_ausente_e_vazio(tmp_path: Path) -> None:
    assert lidos(tmp_path / "nada.jsonl") == {}


def _spots(tmp_path: Path) -> Path:
    caminho = tmp_path / "tdeed.jsonl"
    linha = {
        "clip_id": "c1",
        "frame_index": 40,
        "confidence": 0.61,
        "source": "tdeed",
        "dominant_class": "SHOT",
        "candidates": [{"frame_index": 40, "sample_index": 8, "score": 0.61}],
        "ball_support": "absent",
        "ball_detected_frames": 0,
        "ball_interpolated_frames": 0,
        "duration_ms": 6900.0,
    }
    caminho.write_text(json.dumps(linha) + "\n")
    return caminho


def _lote(clip_id: str | None, indices: tuple[int, ...] = (30, 35, 40, 45)) -> FrameBatch:
    tensor = np.zeros((len(indices), PREPROCESS.height, PREPROCESS.width, 1), dtype=np.uint8)
    return FrameBatch(indices=indices, tensor=tensor, preprocess=PREPROCESS, clip_id=clip_id)


class TestPrecomputedShot(ModuleConformance):
    @pytest.fixture
    def module(self, tmp_path: Path) -> PerceptionModule:
        return PrecomputedShot(_spots(tmp_path))


def test_a_segunda_etapa_le_o_instante_e_recalcula_o_apoio_da_bola(tmp_path: Path) -> None:
    modulo = PrecomputedShot()
    modulo.configure({"spots": str(_spots(tmp_path))})
    modulo.load()
    bola = ModuleOutput(
        kind="ball",
        frames=(35, 40),
        records=(({"box": (0, 0, 2, 2), "source": "detected"},),) * 2,
    )

    saida = modulo.infer(_lote("c1"), {"ball": bola})

    registro = [r for grupo in saida.records for r in grupo][0]
    assert registro["frame_index"] == 40
    assert registro["source"] == "tdeed"
    assert registro["precomputed"] is True
    assert registro["ball_support"] == "detected"
    assert registro["ball_detected_frames"] == 2
    assert "duration_ms" not in registro
    assert "clip_id" not in registro


def test_clipe_fora_do_arquivo_sai_sem_instante(tmp_path: Path) -> None:
    modulo = PrecomputedShot(_spots(tmp_path))
    modulo.load()

    saida = modulo.infer(_lote("c9"), {})

    assert all(grupo == () for grupo in saida.records)


def test_sem_arquivo_a_segunda_etapa_nao_carrega(tmp_path: Path) -> None:
    with pytest.raises(ModuleError, match="spots"):
        PrecomputedShot().load()
    with pytest.raises(ModuleError, match="xvision spots"):
        PrecomputedShot(tmp_path / "nada.jsonl").load()
    with pytest.raises(ModuleNotLoaded):
        PrecomputedShot(_spots(tmp_path)).infer(_lote("c1"), {})


@pytest.mark.parametrize(("ajustes", "erro"), [({"x": 1}, "nao existe"), ({"spots": 3}, "texto")])
def test_ajuste_invalido_e_recusado(ajustes: dict[str, object], erro: str) -> None:
    with pytest.raises(ValueError, match=erro):
        PrecomputedShot().configure(ajustes)


def test_a_impressao_segue_o_conteudo_do_arquivo(tmp_path: Path) -> None:
    caminho = _spots(tmp_path)
    modulo = PrecomputedShot(caminho)
    antes = modulo.fingerprint()

    caminho.write_text(caminho.read_text().replace('"frame_index": 40', '"frame_index": 41'))

    assert modulo.fingerprint() != antes
    assert PrecomputedShot().fingerprint() == "spots=ausente"
