"""O modulo `shot` que chama o T-DEED por processo separado, testado com um executor falso.

O executor falso cumpre o mesmo protocolo de arquivo do verdadeiro, le o `.npy` e escreve o JSON,
e faz a coluna `SHOT` seguir o brilho do quadro. Assim o teste controla onde fica o pico sem torch,
sem peso e sem o codigo GPL do T-DEED.
"""

from __future__ import annotations

import json
import textwrap
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from edge.cache.perception import tuning_of
from edge.features.base import parse_shot_actions
from edge.modules.base import (
    FrameBatch,
    ModuleError,
    ModuleNotLoaded,
    ModuleOutput,
    PerceptionModule,
)
from edge.modules.conformance import ModuleConformance
from edge.modules.records import check_records
from edge.modules.registry import load_modules, read_config
from edge.modules.shot import TdeedShot
from edge.modules.shot.tdeed import CANDIDATOS_REGISTRADOS, FONTE, PREPROCESS, SUPRESSAO

EXECUTOR = """
import argparse, json, sys
import numpy as np
p = argparse.ArgumentParser()
for nome in ("--source", "--checkpoint", "--frames", "--saida", "--device"):
    p.add_argument(nome)
a = p.parse_args()
quadros = np.load(a.frames)
brilho = quadros.reshape(len(quadros), -1).mean(axis=1) / 255.0
classes = ["BACKGROUND", "PASS", "SHOT"]
linhas = [[1.0 - b, b / 2.0, b] for b in brilho]
if len(linhas) > 1:
    linhas[0][1] = 0.99
json.dump({"classes": classes, "probabilities": linhas, "device": a.device}, open(a.saida, "w"))
"""


def _executor(pasta: Path, codigo: str = EXECUTOR) -> Path:
    caminho = pasta / "run.py"
    caminho.write_text(textwrap.dedent(codigo))
    return caminho


def _fonte(pasta: Path) -> Path:
    (pasta / "T-DEED" / "model").mkdir(parents=True)
    (pasta / "T-DEED" / "model" / "model.py").write_text("# falso\n")
    return pasta / "T-DEED"


def _modulo(pasta: Path, codigo: str = EXECUTOR, **ajustes: object) -> TdeedShot:
    checkpoint = pasta / "checkpoint_best.pt"
    checkpoint.write_bytes(b"peso")
    modulo = TdeedShot(runner=_executor(pasta, codigo), source=_fonte(pasta), checkpoint=checkpoint)
    if ajustes:
        modulo.configure(ajustes)
    return modulo


def _lote(brilho: Sequence[float], inicio: int = 100, passo: int = 5) -> FrameBatch:
    tensor = np.stack(
        [
            np.full((PREPROCESS.height, PREPROCESS.width, 3), round(b * 255), dtype=np.uint8)
            for b in brilho
        ]
    )
    indices = tuple(inicio + passo * i for i in range(len(brilho)))
    return FrameBatch(indices=indices, tensor=tensor, preprocess=PREPROCESS)


def _registro(saida: ModuleOutput) -> dict[str, object]:
    registros = [r for grupo in saida.records for r in grupo]
    assert len(registros) == 1
    return dict(registros[0])


class TestTdeedShot(ModuleConformance):
    @pytest.fixture
    def module(self, tmp_path: Path) -> PerceptionModule:
        return _modulo(tmp_path)


def test_o_instante_e_o_pico_de_shot_em_quadro_nativo(tmp_path: Path) -> None:
    modulo = _modulo(tmp_path)
    modulo.load()

    registro = _registro(modulo.infer(_lote([0.1, 0.2, 0.9, 0.3]), {}))

    assert registro["frame_index"] == 110
    assert registro["source"] == FONTE
    assert registro["confidence"] == pytest.approx(0.9, abs=0.01)
    assert registro["ball_support"] == "absent"


def test_a_classe_dominante_fica_registrada_mesmo_quando_nao_e_shot(tmp_path: Path) -> None:
    modulo = _modulo(tmp_path)
    modulo.load()

    registro = _registro(modulo.infer(_lote([0.1, 0.2, 0.5]), {}))

    assert registro["dominant_class"] == "PASS"
    assert registro["frame_index"] == 110


def test_o_registro_carrega_a_matriz_inteira_de_passo_por_classe(tmp_path: Path) -> None:
    modulo = _modulo(tmp_path)
    modulo.load()

    saida = modulo.infer(_lote([0.1, 0.2, 0.9, 0.3]), {})
    check_records(saida)
    acoes = parse_shot_actions(json.loads(json.dumps(_registro(saida)["actions"])))

    assert acoes is not None
    assert acoes.classes == ("BACKGROUND", "PASS", "SHOT")
    assert acoes.frame_indices == (100, 105, 110, 115)
    assert [p for _, p in acoes.serie("SHOT")] == pytest.approx([0.1, 0.2, 0.9, 0.3], abs=0.01)
    assert acoes.serie("PASS")[0] == (100, 0.99)


def test_os_candidatos_sao_picos_separados_pela_supressao(tmp_path: Path) -> None:
    brilho = [0.1] * 60
    brilho[10], brilho[12], brilho[40] = 0.9, 0.8, 0.7
    modulo = _modulo(tmp_path)
    modulo.load()

    registro = _registro(modulo.infer(_lote(brilho, inicio=0, passo=1), {}))

    candidatos = cast("list[dict[str, int]]", registro["candidates"])
    quadros = [c["frame_index"] for c in candidatos]
    assert quadros[:2] == [10, 40]
    assert len(quadros) == CANDIDATOS_REGISTRADOS
    assert all(abs(a - b) > SUPRESSAO for a in quadros for b in quadros if a != b)


def test_o_apoio_da_bola_vem_da_especie_ball_quando_ela_rodou(tmp_path: Path) -> None:
    bola = ModuleOutput(
        kind="ball",
        frames=(105, 110),
        records=(
            ({"box": (0, 0, 4, 4), "source": "detected"},),
            ({"box": (0, 0, 4, 4), "source": "detected"},),
        ),
    )
    modulo = _modulo(tmp_path)
    modulo.load()

    registro = _registro(modulo.infer(_lote([0.1, 0.2, 0.9]), {"ball": bola}))

    assert registro["ball_support"] == "detected"
    assert registro["ball_detected_frames"] == 2
    check_records(modulo.infer(_lote([0.1, 0.9]), {"ball": bola}))


def test_executor_que_falha_levanta_com_o_fim_do_stderr(tmp_path: Path) -> None:
    modulo = _modulo(tmp_path, "import sys\nsys.stderr.write('sem cuda aqui')\nsys.exit(3)\n")
    modulo.load()

    with pytest.raises(ModuleError, match="saiu com 3: sem cuda aqui"):
        modulo.infer(_lote([0.5]), {})


def test_executor_que_devolve_outra_forma_e_recusado(tmp_path: Path) -> None:
    codigo = EXECUTOR.replace("linhas = [", "linhas = [[0.0, 0.0, 1.0]] + [")
    modulo = _modulo(tmp_path, codigo)
    modulo.load()

    with pytest.raises(ModuleError, match="devolveu"):
        modulo.infer(_lote([0.5, 0.6]), {})


def test_executor_sem_classe_shot_e_recusado(tmp_path: Path) -> None:
    modulo = _modulo(tmp_path, EXECUTOR.replace('"SHOT"]', '"GOAL"]'))
    modulo.load()

    with pytest.raises(ModuleError, match="classe SHOT"):
        modulo.infer(_lote([0.5]), {})


def test_executor_lento_demais_e_interrompido(tmp_path: Path) -> None:
    modulo = _modulo(tmp_path, "import time\ntime.sleep(5)\n", timeout_s=0.5)
    modulo.load()

    with pytest.raises(ModuleError, match="passou de"):
        modulo.infer(_lote([0.5]), {})


def test_carregar_sem_executor_sem_codigo_ou_sem_peso_diz_o_que_falta(tmp_path: Path) -> None:
    sem_peso = TdeedShot(runner=_executor(tmp_path), source=_fonte(tmp_path))
    with pytest.raises(ModuleError, match="artifact"):
        sem_peso.load()

    peso = tmp_path / "peso.pt"
    peso.write_bytes(b"x")
    sem_executor = TdeedShot(runner=tmp_path / "nada.py", source=tmp_path, checkpoint=peso)
    with pytest.raises(ModuleError, match="executor"):
        sem_executor.load()

    sem_codigo = TdeedShot(runner=_executor(tmp_path), source=tmp_path / "x", checkpoint=peso)
    with pytest.raises(ModuleError, match="README"):
        sem_codigo.load()


def test_fechado_nao_infere(tmp_path: Path) -> None:
    modulo = _modulo(tmp_path)
    with pytest.raises(ModuleNotLoaded):
        modulo.infer(_lote([0.5]), {})


@pytest.mark.parametrize(
    ("ajustes", "erro"),
    [
        ({"limiar": 0.5}, "nao existe"),
        ({"runner": 3}, "caminho"),
        ({"device": "tpu"}, "device"),
        ({"timeout_s": 0}, "positivos"),
        ({"timeout_s": True}, "positivos"),
    ],
)
def test_ajuste_invalido_e_recusado(ajustes: dict[str, object], erro: str) -> None:
    with pytest.raises(ValueError, match=erro):
        TdeedShot().configure(ajustes)


def test_a_impressao_muda_quando_o_executor_ou_o_dispositivo_muda(tmp_path: Path) -> None:
    modulo = _modulo(tmp_path)
    antes = tuning_of(modulo)

    modulo.configure({"device": "cpu"})
    depois_do_dispositivo = tuning_of(modulo)
    modulo.runner.write_text(EXECUTOR + "\n# outra versao\n")

    assert len({antes, depois_do_dispositivo, tuning_of(modulo)}) == 3


def test_os_padroes_apontam_para_external_e_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XVISION_VAR_DIR", str(tmp_path))
    modulo = TdeedShot()

    assert modulo.runner.parts[-3:] == ("external", "tdeed", "run.py")
    assert modulo.source == tmp_path / "external" / "T-DEED"


def test_a_configuracao_de_producao_escolhe_o_tdeed() -> None:
    for nome in ("config/pipeline.models.toml", "config/pipeline.models.cpu.toml"):
        modulos = load_modules(read_config(Path(nome)))
        shot = modulos["shot"]
        assert isinstance(shot, TdeedShot), nome
        assert shot.spec.frames == "tdeed"
        assert shot.spec.artifact is not None and "tdeed" in shot.spec.artifact
