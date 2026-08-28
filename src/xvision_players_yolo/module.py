"""Os dois `PerceptionModule` deste pacote, e o motivo de a chave ainda nao estar virada.

O `config/pipeline.toml` continua apontando `players` e `tracking` para os stubs, e nao para as
duas classes daqui, por uma razao que nao e de acabamento: **o modulo nao tem como saber a
resolucao do clipe de origem**.

O `records.py` da borda cobra que `box` saia em pixel do quadro original. O `FrameBatch` que o
runner entrega carrega `indices`, `tensor` e `preprocess`, e o tensor ja veio redimensionado e
com letterbox aplicado. Nenhum dos tres diz quanto media o quadro antes. Os stubs contornam isso
lendo o gabarito da fixture, que um modelo de verdade nao tem. A conta de volta esta escrita aqui
em `to_source_box` e e exata: falta so o par de numeros.

Enquanto o `FrameBatch` nao carregar esse par, `load()` recusa. Devolver box no espaco do tensor
seria a falha que este repositorio mais odeia: o runner monta `Perception` sem reclamar, as sete
etapas saem `ok`, e a coordenada esta errada por um fator que depende do clipe. Recusar no `load`
custa uma etapa vermelha e diz exatamente onde por a mao.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from edge.modules.base import (
    FrameBatch,
    ModuleInputs,
    ModuleNotLoaded,
    ModuleOutput,
    ModuleSpec,
    PreprocessSpec,
    check_preprocess,
)
from xvision_players_yolo.detector import DetectionBackend, Frame, UltralyticsBackend
from xvision_players_yolo.tracking import TrackingBackend, UltralyticsTrackingBackend
from xvision_players_yolo.types import Box, clip_box

PREPROCESS = PreprocessSpec(
    width=1280,
    height=1280,
    color="bgr",
    letterbox=True,
    normalize="none",
)
"""BGR uint8 com letterbox: e exatamente o que o Ultralytics consome sem reconverter."""

VERSION = "0.1.0"

SEM_TAMANHO = (
    "o modulo nao recebeu a resolucao do clipe de origem, e sem ela a caixa sairia em pixel do "
    "tensor pre-processado, e nao do quadro original que `edge.modules.records` cobra. O "
    "`FrameBatch` nao carrega essa medida: construa o modulo com `source_size=(largura, altura)` "
    "ou estenda o `FrameBatch` para leva-la"
)


def to_source_box(box: Box, preprocess: PreprocessSpec, source: tuple[int, int]) -> Box:
    """Desfaz o enquadramento que `edge.frames.preprocess.prepare` aplicou.

    E o inverso literal do `_fit` de la, letterbox inclusive, e nao uma aproximacao: a escala e a
    mesma `min(largura_alvo / largura, altura_alvo / altura)` e a margem e a mesma divisao inteira
    por dois. Sair da conta exata deslocaria toda caixa por ate um pixel em cada eixo.
    """
    largura, altura = source
    if preprocess.letterbox:
        escala = min(preprocess.width / largura, preprocess.height / altura)
        esquerda = (preprocess.width - max(1, round(largura * escala))) // 2
        topo = (preprocess.height - max(1, round(altura * escala))) // 2
        devolvida = Box(
            x1=(box.x1 - esquerda) / escala,
            y1=(box.y1 - topo) / escala,
            x2=(box.x2 - esquerda) / escala,
            y2=(box.y2 - topo) / escala,
        )
    else:
        por_x = largura / preprocess.width
        por_y = altura / preprocess.height
        devolvida = Box(x1=box.x1 * por_x, y1=box.y1 * por_y, x2=box.x2 * por_x, y2=box.y2 * por_y)
    return clip_box(devolvida, width=largura, height=altura)


class _SourceSpace:
    """A metade comum das duas especies: recusar sem tamanho e converter a caixa de volta."""

    spec: ModuleSpec

    def __init__(self, source_size: tuple[int, int] | None) -> None:
        self._source_size = source_size

    def _require_source_size(self) -> tuple[int, int]:
        if self._source_size is None:
            raise ModuleNotLoaded(f"{self.spec.name}: {SEM_TAMANHO}")
        return self._source_size

    def _box(self, box: Box) -> tuple[float, float, float, float]:
        convertida = to_source_box(box, self.spec.preprocess, self._require_source_size())
        return (convertida.x1, convertida.y1, convertida.x2, convertida.y2)


def _frames(batch: FrameBatch) -> tuple[Frame, ...]:
    """Garante quadro contiguo: o lote pode chegar de uma view, e o Ultralytics entrega ao
    torch sem conferir. Em lote ja contiguo a chamada nao copia nada.
    """
    return tuple(np.ascontiguousarray(frame) for frame in batch.tensor)


class PlayersModule(_SourceSpace):
    """A especie `players`: caixa, confianca e papel, sem identificador de trajetoria."""

    spec = ModuleSpec(
        kind="players",
        name="football-player-detection",
        version=VERSION,
        frames="coarse",
        preprocess=PREPROCESS,
    )

    def __init__(
        self,
        weights: Path | None = None,
        *,
        source_size: tuple[int, int] | None = None,
        image_size: int = PREPROCESS.width,
        confidence_threshold: float = 0.25,
        device: str = "auto",
        backend: DetectionBackend | None = None,
    ) -> None:
        super().__init__(source_size)
        self._weights = weights
        self._image_size = image_size
        self._confidence_threshold = confidence_threshold
        self._device = device
        self._injected = backend
        self._backend: DetectionBackend | None = None

    def load(self) -> None:
        self._require_source_size()
        if self._injected is not None:
            self._backend = self._injected
            return
        if self._weights is None:
            raise ModuleNotLoaded(f"{self.spec.name}: nenhum peso e nenhum backend fornecido")
        self._backend = UltralyticsBackend(
            self._weights,
            image_size=self._image_size,
            confidence_threshold=self._confidence_threshold,
            device=self._device,
        )

    def close(self) -> None:
        self._backend = None

    def infer(self, batch: FrameBatch, ctx: ModuleInputs) -> ModuleOutput:
        if self._backend is None:
            raise ModuleNotLoaded(self.spec.name)
        check_preprocess(batch, self.spec.preprocess)
        papeis = self._backend.class_roles
        registros = tuple(
            tuple(
                {
                    "box": self._box(raw.box),
                    "confidence": raw.confidence,
                    "role": papeis[raw.class_id],
                }
                for raw in self._backend.infer(frame)
                if raw.class_id in papeis
            )
            for frame in _frames(batch)
        )
        return ModuleOutput(kind=self.spec.kind, frames=batch.indices, records=registros)


class TrackingModule(_SourceSpace):
    """A especie `tracking`: o identificador de trajetoria, ligado a deteccao por posicao.

    `detection_index` e a posicao do registro dentro do quadro, igual ao `StubTracking`. O runner
    nunca preenche um a partir do outro, entao a ordem daqui precisa ser a mesma que `players`
    emitiria para o mesmo quadro: por isso o filtro de papel e identico nas duas classes.
    """

    spec = ModuleSpec(
        kind="tracking",
        name="football-player-detection",
        version=VERSION,
        frames="window",
        preprocess=PREPROCESS,
    )

    def __init__(
        self,
        weights: Path | None = None,
        *,
        source_size: tuple[int, int] | None = None,
        image_size: int = PREPROCESS.width,
        confidence_threshold: float = 0.25,
        device: str = "auto",
        backend: TrackingBackend | None = None,
    ) -> None:
        super().__init__(source_size)
        self._weights = weights
        self._image_size = image_size
        self._confidence_threshold = confidence_threshold
        self._device = device
        self._injected = backend
        self._backend: TrackingBackend | None = None

    def load(self) -> None:
        self._require_source_size()
        if self._injected is not None:
            self._backend = self._injected
        elif self._weights is None:
            raise ModuleNotLoaded(f"{self.spec.name}: nenhum peso e nenhum backend fornecido")
        else:
            self._backend = UltralyticsTrackingBackend(
                self._weights,
                image_size=self._image_size,
                confidence_threshold=self._confidence_threshold,
                device=self._device,
            )
        self._backend.start_clip()

    def close(self) -> None:
        self._backend = None

    def infer(self, batch: FrameBatch, ctx: ModuleInputs) -> ModuleOutput:
        if self._backend is None:
            raise ModuleNotLoaded(self.spec.name)
        check_preprocess(batch, self.spec.preprocess)
        papeis = self._backend.class_roles
        registros: list[tuple[Mapping[str, object], ...]] = []
        for frame in _frames(batch):
            aceitas = [raw for raw in self._backend.track(frame) if raw.class_id in papeis]
            registros.append(
                tuple(
                    {
                        "detection_index": posicao,
                        "track_id": raw.track_id,
                        "box": self._box(raw.box),
                    }
                    for posicao, raw in enumerate(aceitas)
                )
            )
        return ModuleOutput(kind=self.spec.kind, frames=batch.indices, records=tuple(registros))


__all__ = ["PREPROCESS", "PlayersModule", "TrackingModule", "to_source_box"]
