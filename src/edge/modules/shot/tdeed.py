"""O instante da finalizacao pelo T-DEED, um detector temporal de eventos publicado.

Medido na sprint 4 nos 24 clipes de ajuste, com o checkpoint publicado e sem treino local, ele pos
24 de 24 instantes a ate 30 quadros do gabarito, com mediana de 5, contra 0,292 e mediana de 46 da
regra cinematica. A regra continua em `BallBreakShot`, e e ela que roda na configuracao sem modelo.

O codigo do T-DEED e GPL-3.0 e este arquivo nao o importa. O executor mora em `external/tdeed/`,
com a licenca dele, e roda em outro processo: este modulo grava os quadros num `.npy`, chama o
executor e le as probabilidades de volta em JSON. Importar o modelo aqui faria de `src/` obra
derivada, e o produto inteiro teria de mudar de licenca para responder quando foi o chute.

O modelo localiza um chute que se sabe existir, ele nao decide se houve chute. Por isso o instante e
o maximo da coluna `SHOT` mesmo quando outra classe pontua mais alto no clipe, o que acontece em 14
dos 24 clipes de ajuste. O corte do parceiro sempre contem a finalizacao, e `dominant_class` fica no
registro para quem quiser medir o caso em que isso deixar de ser verdade.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from edge.artifacts import entrypoint_of
from edge.modules.base import (
    FrameBatch,
    ModuleError,
    ModuleInputs,
    ModuleNotLoaded,
    ModuleOutput,
    ModuleSpec,
    PreprocessSpec,
    check_preprocess,
)
from edge.modules.shot.kinematics import (
    BALL,
    CANDIDATOS_REGISTRADOS,
    procedencias,
    saida_unica,
    suporte,
)
from edge.paths import var_dir

FONTE = "tdeed"
SHOT = "SHOT"
FUNDO = "BACKGROUND"
RUNNER = Path("..") / "external" / "tdeed" / "run.py"
DISPOSITIVOS = frozenset({"auto", "cpu", "cuda"})
AJUSTES = frozenset({"runner", "source", "device", "timeout_s"})
SUPRESSAO = 6
"""Passos em volta de um pico que nao contam como outro candidato: meio segundo a 12 por segundo.

Sem isso os cinco candidatos registrados seriam o pico e os quatro passos vizinhos dele, e a lista
deixaria de responder se havia outro momento do clipe parecido com um chute.
"""

PREPROCESS = PreprocessSpec(width=796, height=448, color="rgb", letterbox=False, normalize="none")
"""O quadro que o checkpoint publicado viu no treino. A normalizacao e feita dentro do modelo."""


class TdeedShot:
    spec = ModuleSpec(
        kind="shot",
        name="tdeed",
        version="0.1.0",
        frames="tdeed",
        preprocess=PREPROCESS,
        optional_needs=(BALL,),
    )

    def __init__(
        self,
        *,
        runner: Path | None = None,
        source: Path | None = None,
        checkpoint: Path | None = None,
        device: str = "auto",
        timeout_s: float = 600.0,
    ) -> None:
        self._runner = runner
        self._source = source
        self._checkpoint_fixo = checkpoint
        self._device = device
        self._timeout = timeout_s
        self._checkpoint: Path | None = None

    def configure(self, options: Mapping[str, object]) -> None:
        for nome, valor in options.items():
            if nome not in AJUSTES:
                raise ValueError(
                    f"shot.{nome} nao existe: os ajustes sao {', '.join(sorted(AJUSTES))}"
                )
            if nome in {"runner", "source"} and not isinstance(valor, str):
                raise ValueError(f"shot.{nome} espera caminho em texto, recebeu {valor!r}")
            if nome == "device" and valor not in DISPOSITIVOS:
                raise ValueError(f"shot.device aceita {', '.join(sorted(DISPOSITIVOS))}")
            if nome == "timeout_s" and (
                isinstance(valor, bool) or not isinstance(valor, int | float) or valor <= 0
            ):
                raise ValueError(f"shot.timeout_s espera segundos positivos, recebeu {valor!r}")
        if "runner" in options:
            self._runner = Path(cast("str", options["runner"]))
        if "source" in options:
            self._source = Path(cast("str", options["source"]))
        if "device" in options:
            self._device = cast("str", options["device"])
        if "timeout_s" in options:
            self._timeout = float(cast("float", options["timeout_s"]))

    def fingerprint(self) -> str:
        """Executor, codigo do modelo e dispositivo, que mudam a saida sem mudar este arquivo.

        O cache de percepcao ja cobre o peso pelo digest e este arquivo pelo hash da fonte. O que
        fica fora dos dois e o executor em `external/` e o checkout do T-DEED, e trocar qualquer um
        deles serviria probabilidade velha sob a mesma chave.
        """
        return ";".join(
            (
                f"runner={_hash_de((self.runner,))}",
                f"source={_hash_de(sorted((self.source / 'model').glob('*.py')))}",
                f"device={self._device}",
            )
        )

    @property
    def runner(self) -> Path:
        return self._runner if self._runner is not None else RUNNER

    @property
    def source(self) -> Path:
        return self._source if self._source is not None else var_dir() / "external" / "T-DEED"

    def load(self) -> None:
        checkpoint = self._checkpoint_fixo or entrypoint_of(self.spec.artifact)
        if checkpoint is None:
            raise ModuleError("shot.tdeed precisa de artifact com o checkpoint publicado")
        if not self.runner.is_file():
            raise ModuleError(f"o executor do T-DEED nao esta em {self.runner}")
        if not (self.source / "model").is_dir():
            raise ModuleError(
                f"o codigo do T-DEED nao esta em {self.source}: veja external/tdeed/README.md"
            )
        self._checkpoint = checkpoint

    def close(self) -> None:
        self._checkpoint = None

    def infer(self, batch: FrameBatch, ctx: ModuleInputs) -> ModuleOutput:
        if self._checkpoint is None:
            raise ModuleNotLoaded(self.spec.name)
        check_preprocess(batch, self.spec.preprocess)
        if not batch.indices:
            return ModuleOutput(kind=self.spec.kind, frames=(), records=())
        classes, probabilidades = self._executa(batch.tensor, self._checkpoint)
        if probabilidades.shape != (len(batch.indices), len(classes)):
            raise ModuleError(
                f"o executor devolveu {probabilidades.shape} para {len(batch.indices)} quadros"
            )
        registro = _registro(batch.indices, classes, probabilidades, procedencias(ctx.get(BALL)))
        return saida_unica(self.spec.kind, batch.indices, registro)

    def _executa(
        self, tensor: np.ndarray[Any, Any], checkpoint: Path
    ) -> tuple[tuple[str, ...], np.ndarray[Any, np.dtype[np.float64]]]:
        with tempfile.TemporaryDirectory(prefix="xvision-tdeed-") as pasta:
            quadros, saida = Path(pasta) / "quadros.npy", Path(pasta) / "saida.json"
            np.save(quadros, tensor)
            comando = [
                sys.executable,
                str(self.runner),
                "--source",
                str(self.source),
                "--checkpoint",
                str(checkpoint),
                "--frames",
                str(quadros),
                "--saida",
                str(saida),
                "--device",
                self._device,
            ]
            try:
                feito = subprocess.run(
                    comando, capture_output=True, text=True, timeout=self._timeout, check=False
                )
            except subprocess.TimeoutExpired as erro:
                raise ModuleError(f"o executor do T-DEED passou de {self._timeout:.0f} s") from erro
            if feito.returncode != 0 or not saida.is_file():
                cauda = "\n".join(feito.stderr.strip().splitlines()[-5:])
                raise ModuleError(f"o executor do T-DEED saiu com {feito.returncode}: {cauda}")
            documento = cast("dict[str, Any]", json.loads(saida.read_text(encoding="utf-8")))
        classes = tuple(str(nome) for nome in cast("list[object]", documento["classes"]))
        if SHOT not in classes:
            raise ModuleError(f"o executor nao devolveu a classe {SHOT}")
        return classes, np.asarray(documento["probabilities"], dtype=np.float64)


def _registro(
    indices: Sequence[int],
    classes: Sequence[str],
    probabilidades: np.ndarray[Any, np.dtype[np.float64]],
    procedencia: Mapping[int, str],
) -> dict[str, object]:
    coluna = probabilidades[:, classes.index(SHOT)]
    passo = int(coluna.argmax())
    quadro = indices[passo]
    eventos = [i for i, nome in enumerate(classes) if nome != FUNDO]
    dominante = max(eventos, key=lambda i: float(probabilidades[:, i].max()))
    return {
        "frame_index": quadro,
        "confidence": round(float(coluna[passo]), 6),
        "source": FONTE,
        "dominant_class": classes[dominante],
        "candidates": _picos(indices, coluna),
        **suporte(quadro, procedencia),
    }


def _picos(
    indices: Sequence[int], coluna: np.ndarray[Any, np.dtype[np.float64]]
) -> list[dict[str, object]]:
    """Os `CANDIDATOS_REGISTRADOS` maiores picos de `SHOT`, cada um suprimindo a vizinhanca."""
    livres = np.ones(len(coluna), dtype=bool)
    picos: list[dict[str, object]] = []
    for passo in np.argsort(-coluna, kind="stable"):
        if len(picos) == CANDIDATOS_REGISTRADOS:
            break
        if not livres[passo]:
            continue
        livres[max(0, passo - SUPRESSAO) : passo + SUPRESSAO + 1] = False
        picos.append(
            {
                "frame_index": indices[int(passo)],
                "sample_index": int(passo),
                "score": round(float(coluna[passo]), 6),
            }
        )
    return picos


def _hash_de(arquivos: Sequence[Path]) -> str:
    digestor = hashlib.sha256()
    for arquivo in arquivos:
        digestor.update(arquivo.name.encode())
        digestor.update(arquivo.read_bytes() if arquivo.is_file() else b"ausente")
    return digestor.hexdigest()[:16]
