"""O comando de pose valida o video, declara a populacao e publica o numero que a feature consome.

Ele entra pela caixa anotada e nao pela saida da especie inteira, porque `pose` escolhe o
finalizador e estima o corpo dele, e um numero que some as duas nao diz qual mudou.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import cli
from edge.artifacts import sha256_of
from edge.modules.pose import COCO_KEYPOINTS, YoloPose
from edge.modules.registry import ModuleConfig
from edge.testing.make_clip import make_synthetic_clip
from mlops.catalog import ShotRecord

CAIXA = (100.0, 60.0, 140.0, 190.0)


def _posicao(nome: str) -> tuple[float, float]:
    if nome == "ankle_left":
        return (112.0, 185.0)
    if nome == "ankle_right":
        return (123.0, 185.0)
    return (120.0, 65.0 + COCO_KEYPOINTS.index(nome) * 5.0)


class _PoseFalsa:
    """Responde os keypoints anotados, com a opcao de trocar os dois tornozelos de lado."""

    spec = YoloPose.spec

    def __init__(self, *, trocado: bool = False) -> None:
        self._trocado = trocado
        self.caixas: list[Any] = []
        self.carregado = False

    def load(self) -> None:
        self.carregado = True

    def close(self) -> None:
        self.carregado = False

    def estimate_boxes(self, batch: Any, caixas: Any) -> dict[int, Any]:
        self.caixas.append(dict(caixas))
        pontos: list[tuple[str, float, float, float]] = []
        for nome in COCO_KEYPOINTS:
            origem = nome
            if self._trocado and nome == "ankle_left":
                origem = "ankle_right"
            elif self._trocado and nome == "ankle_right":
                origem = "ankle_left"
            x, y = _posicao(origem)
            pontos.append((nome, x, y, 0.9))
        return {indice: tuple(pontos) for indice in caixas}


def _substitui_modulos(monkeypatch: pytest.MonkeyPatch, modulos: dict[str, object]) -> None:
    def carregar(_config: tuple[ModuleConfig, ...]) -> dict[str, object]:
        return modulos

    monkeypatch.setattr(cli, "load_modules", carregar)


@pytest.fixture
def pose_population(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    video = make_synthetic_clip(tmp_path / "synthetic.mp4")
    record = ShotRecord("synthetic", "synthetic-match", video, True, 0, {})
    monkeypatch.setattr(cli, "load_catalog", lambda: (record,))
    comum = {
        "schema_version": "1.0",
        "clip_id": "synthetic",
        "match_id": "synthetic-match",
        "video_sha256": sha256_of(video),
        "frame_width": 320,
        "frame_height": 240,
        "shooter_box": list(CAIXA),
        "keypoints": [
            {"name": n, "x": _posicao(n)[0], "y": _posicao(n)[1], "visibility": "visible"}
            for n in COCO_KEYPOINTS
        ],
        "annotator": "reviewer",
        "reviewed_by": None,
        "created_at": "2026-09-22T12:00:00Z",
    }
    return [
        {**comum, "frame_index": 0, "quality_flags": ["human-annotated"]},
        {**comum, "frame_index": 1, "quality_flags": ["requires-human-review"]},
    ]


def _roda(tmp_path: Path, populacao: list[dict[str, Any]], *extra: str) -> dict[str, Any]:
    labels = tmp_path / "pose.jsonl"
    labels.write_text("\n".join(map(json.dumps, populacao)), encoding="utf-8")
    saida = tmp_path / "report.json"
    resultado = CliRunner().invoke(
        cli.app, ["eval-pose", "--labels", str(labels), "--saida", str(saida), *extra]
    )
    assert resultado.exit_code == 0, resultado.output
    return json.loads(saida.read_text(encoding="utf-8"))


@pytest.mark.parametrize("include_proposals", [False, True])
def test_o_comando_declara_o_denominador_que_mediu(
    tmp_path: Path,
    pose_population: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    include_proposals: bool,
) -> None:
    _substitui_modulos(monkeypatch, {"pose": _PoseFalsa()})
    extra = ["--include-proposals"] if include_proposals else []

    relatorio = _roda(tmp_path, pose_population, *extra)

    assert relatorio["n_labels"] == 2
    assert relatorio["n_frames"] == (2 if include_proposals else 1)
    assert relatorio["n_excluded_proposals"] == (0 if include_proposals else 1)
    assert relatorio["evaluation_kind"] == (
        "exploratory-with-proposals" if include_proposals else "human-confirmed"
    )
    assert relatorio["population"] == "folds=[0, 1, 2]"


def test_o_estimador_exato_sai_utilizavel(
    tmp_path: Path, pose_population: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _substitui_modulos(monkeypatch, {"pose": _PoseFalsa()})

    relatorio = _roda(tmp_path, pose_population)

    assert relatorio["completion"] == 1.0
    assert relatorio["ankle_side_accuracy"] == 1.0
    assert relatorio["usable_rate"] == 1.0
    assert relatorio["score"] == relatorio["usable_rate"]


def test_a_troca_de_lado_chega_ao_relatorio_como_zero(
    tmp_path: Path, pose_population: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """O defeito que o comando existe para pegar, e que a OKS nao pega."""
    _substitui_modulos(monkeypatch, {"pose": _PoseFalsa(trocado=True)})

    relatorio = _roda(tmp_path, pose_population)

    assert relatorio["oks_median"] > 0.9, "a premissa caiu: a troca tem de ser invisivel para a OKS"
    assert relatorio["ankle_availability"] == 1.0
    assert relatorio["ankle_side_accuracy"] == 0.0
    assert relatorio["usable_rate"] == 0.0


def test_a_caixa_medida_e_a_da_anotacao(
    tmp_path: Path, pose_population: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A escolha do finalizador nao entra aqui: se entrasse, o numero mediria as duas coisas."""
    modulo = _PoseFalsa()
    _substitui_modulos(monkeypatch, {"pose": modulo})

    _roda(tmp_path, pose_population)

    assert modulo.caixas, "o comando nao chamou estimate_boxes"
    caixa = next(iter(modulo.caixas[0].values()))
    assert (caixa.x1, caixa.y1, caixa.x2, caixa.y2) == CAIXA


def test_o_video_diferente_do_anotado_encerra_com_erro(
    tmp_path: Path, pose_population: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _substitui_modulos(monkeypatch, {"pose": _PoseFalsa()})
    pose_population[0]["video_sha256"] = "b" * 64
    labels = tmp_path / "pose.jsonl"
    labels.write_text(json.dumps(pose_population[0]), encoding="utf-8")

    resultado = CliRunner().invoke(cli.app, ["eval-pose", "--labels", str(labels)])

    assert resultado.exit_code == 1
    assert "video_sha256" in resultado.output


def test_a_configuracao_sem_modulo_pose_encerra_com_erro(
    tmp_path: Path, pose_population: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _substitui_modulos(monkeypatch, {})
    labels = tmp_path / "pose.jsonl"
    labels.write_text(json.dumps(pose_population[0]), encoding="utf-8")

    resultado = CliRunner().invoke(cli.app, ["eval-pose", "--labels", str(labels)])

    assert resultado.exit_code == 1
    assert "modules.pose" in resultado.output


def test_o_modulo_sem_o_seam_encerra_com_erro(
    tmp_path: Path, pose_population: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sem `estimate_boxes` o comando teria de refazer recorte e margem por fora, e divergir."""

    class _SemSeam:
        spec = YoloPose.spec

        def load(self) -> None: ...
        def close(self) -> None: ...

    _substitui_modulos(monkeypatch, {"pose": _SemSeam()})
    labels = tmp_path / "pose.jsonl"
    labels.write_text(json.dumps(pose_population[0]), encoding="utf-8")

    resultado = CliRunner().invoke(cli.app, ["eval-pose", "--labels", str(labels)])

    assert resultado.exit_code == 1
    assert "estimate_boxes" in resultado.output


def test_a_dobra_vazia_encerra_com_erro(
    tmp_path: Path, pose_population: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _substitui_modulos(monkeypatch, {"pose": _PoseFalsa()})
    labels = tmp_path / "pose.jsonl"
    labels.write_text(json.dumps(pose_population[0]), encoding="utf-8")

    resultado = CliRunner().invoke(
        cli.app, ["eval-pose", "--labels", str(labels), "--dobra", "reservado"]
    )

    assert resultado.exit_code == 1
    assert "dobra" in resultado.output
