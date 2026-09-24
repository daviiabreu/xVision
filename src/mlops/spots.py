"""O instante em duas etapas: primeiro so a especie `shot`, depois o resto do pipeline.

Na Jetson Orin Nano de 8 GB, o pipeline e o T-DEED nao cabem na memoria ao mesmo tempo. Medido em
24/09 com a interface grafica ligada: a placa parada usa 2,4 GB, o pipeline sem o T-DEED leva o
pico a 5,4 GB, e o executor do T-DEED sozinho leva a 5,7 GB. Os dois juntos passam dos 7,3 GB, e
a memoria que o CUDA pede fica presa na RAM fisica, entao swap nao resolve.

A primeira etapa roda so o `shot` da configuracao, sem outra especie carregada, e grava um instante
por clipe num JSONL que `mlops.instant.read_spots` e `xvision eval-spots` ja leem. A segunda roda o
pipeline com `PrecomputedShot` no lugar do `shot`, e cada etapa tem a GPU inteira para si.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

from edge.frames.provider import FrameProvider, VideoSource, probe
from edge.graph.framesets import FrameSet, resolve
from edge.modules.base import (
    FrameBatch,
    ModuleError,
    ModuleInputs,
    ModuleNotLoaded,
    ModuleOutput,
    ModuleSpec,
    PerceptionModule,
    PreprocessSpec,
    check_preprocess,
)
from edge.modules.shot.kinematics import BALL, procedencias, saida_unica, suporte
from mlops.catalog import ShotRecord

PREPROCESS = PreprocessSpec(width=32, height=32, color="gray", letterbox=False, normalize="none")
"""O menor lote que o contrato aceita: o instante ja vem do arquivo, e o quadro nao e lido."""

DE_FORA = frozenset(
    {"duration_ms", "ball_support", "ball_detected_frames", "ball_interpolated_frames"}
)
"""Campos da primeira etapa que a segunda recalcula ou descarta.

O apoio da bola sai `absent` na primeira etapa, porque ali a especie `ball` nao roda. Copiar isso
para o evento diria que o instante foi escolhido sem bola nenhuma por perto, o que e falso: a bola
roda na segunda etapa, e e dela que o apoio sai.
"""


class SpotsError(ValueError): ...


@dataclass(frozen=True)
class Resumo:
    feitos: int
    pulados: int
    falhas: tuple[tuple[str, str], ...]


def lidos(caminho: Path) -> dict[str, dict[str, Any]]:
    """Os registros por clipe, recusando duplicata e linha sem `frame_index` inteiro."""
    if not caminho.is_file():
        return {}
    registros: dict[str, dict[str, Any]] = {}
    for numero, linha in enumerate(caminho.read_text(encoding="utf-8").splitlines(), start=1):
        if not linha.strip():
            continue
        documento = cast("dict[str, Any]", json.loads(linha))
        clip = str(documento.get("clip_id", ""))
        quadro = documento.get("frame_index")
        if not clip or isinstance(quadro, bool) or not isinstance(quadro, int):
            raise SpotsError(f"{caminho}:{numero}: precisa de clip_id e frame_index inteiro")
        if clip in registros:
            raise SpotsError(f"{caminho}:{numero}: {clip} aparece duas vezes")
        registros[clip] = documento
    return registros


def gera(
    escolhidos: Iterable[ShotRecord],
    modulo: PerceptionModule,
    frameset: FrameSet,
    destino: Path,
    *,
    relogio: Callable[[], float] = time.perf_counter,
) -> Resumo:
    """Roda so a especie `shot` em cada clipe e acrescenta um instante por linha.

    Clipe que ja esta no arquivo e pulado, entao uma execucao interrompida continua de onde parou.
    Falha de um clipe fica registrada no resumo e nao derruba os outros, como no `xvision run`.
    """
    if modulo.spec.kind != "shot":
        raise SpotsError(f"a primeira etapa roda a especie shot, e recebeu {modulo.spec.kind}")
    feitos_antes = set(lidos(destino))
    destino.parent.mkdir(parents=True, exist_ok=True)
    feitos = pulados = 0
    falhas: list[tuple[str, str]] = []
    modulo.load()
    try:
        for registro in escolhidos:
            if registro.shot_id in feitos_antes:
                pulados += 1
                continue
            try:
                linha = _um_clipe(registro, modulo, frameset, relogio)
            except (ModuleError, OSError, ValueError) as erro:
                falhas.append((registro.shot_id, f"{type(erro).__name__}: {erro}"))
                continue
            with destino.open("a", encoding="utf-8") as arquivo:
                arquivo.write(json.dumps(linha, ensure_ascii=False) + "\n")
            feitos += 1
    finally:
        modulo.close()
    return Resumo(feitos=feitos, pulados=pulados, falhas=tuple(falhas))


def _um_clipe(
    registro: ShotRecord,
    modulo: PerceptionModule,
    frameset: FrameSet,
    relogio: Callable[[], float],
) -> dict[str, object]:
    clip = probe(registro.clip_path, clip_id=registro.shot_id)
    indices = resolve(frameset, clip)
    provider = FrameProvider(VideoSource(clip.path))
    try:
        lote = replace(provider.frames(indices, modulo.spec.preprocess), clip_id=registro.shot_id)
    finally:
        provider.close()
    inicio = relogio()
    saida = modulo.infer(lote, {})
    duracao = (relogio() - inicio) * 1000
    instantes = [dict(r) for grupo in saida.records for r in grupo]
    if not instantes:
        raise ModuleError(f"{modulo.spec.name} nao apontou instante em {registro.shot_id}")
    return {"clip_id": registro.shot_id, **instantes[0], "duration_ms": round(duracao, 3)}


class PrecomputedShot:
    """A segunda etapa: o instante lido do arquivo que a primeira gravou.

    O registro sai com os campos da primeira etapa, `source` inclusive, e com o apoio da bola
    recalculado sobre a especie `ball` desta execucao. Clipe que nao esta no arquivo sai sem
    instante, e o `process_clip` recusa o clipe, em vez de inventar um quadro.
    """

    spec = ModuleSpec(
        kind="shot",
        name="precomputed",
        version="0.1.0",
        frames="coarse",
        preprocess=PREPROCESS,
        optional_needs=(BALL,),
    )

    def __init__(self, spots: Path | None = None) -> None:
        self._caminho = spots
        self._registros: dict[str, dict[str, Any]] | None = None

    def configure(self, options: Mapping[str, object]) -> None:
        for nome, valor in options.items():
            if nome != "spots":
                raise ValueError(f"shot.{nome} nao existe: o unico ajuste e spots")
            if not isinstance(valor, str):
                raise ValueError(f"shot.spots espera caminho em texto, recebeu {valor!r}")
            self._caminho = Path(valor)

    def fingerprint(self) -> str:
        """O conteudo do arquivo entra na chave do cache: outra primeira etapa, outro instante."""
        if self._caminho is None or not self._caminho.is_file():
            return "spots=ausente"
        return f"spots={hashlib.sha256(self._caminho.read_bytes()).hexdigest()[:16]}"

    def load(self) -> None:
        if self._caminho is None:
            raise ModuleError("shot.precomputed precisa de spots, o arquivo da primeira etapa")
        if not self._caminho.is_file():
            raise ModuleError(f"{self._caminho} nao existe: rode `xvision spots` antes")
        self._registros = lidos(self._caminho)

    def close(self) -> None:
        self._registros = None

    def infer(self, batch: FrameBatch, ctx: ModuleInputs) -> ModuleOutput:
        if self._registros is None:
            raise ModuleNotLoaded(self.spec.name)
        check_preprocess(batch, self.spec.preprocess)
        if not batch.indices:
            return ModuleOutput(kind=self.spec.kind, frames=(), records=())
        gravado = self._registros.get(batch.clip_id or "")
        if gravado is None:
            return ModuleOutput(
                kind=self.spec.kind, frames=batch.indices, records=((),) * len(batch.indices)
            )
        quadro = int(gravado["frame_index"])
        registro = {k: v for k, v in gravado.items() if k not in DE_FORA and k != "clip_id"}
        registro.update(suporte(quadro, procedencias(ctx.get(BALL))))
        registro["precomputed"] = True
        return saida_unica(self.spec.kind, batch.indices, registro)
