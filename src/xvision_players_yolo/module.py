"""Os dois `PerceptionModule` deste pacote, agora que a chave pode ser virada.

Este arquivo nasceu bloqueado, e o bloqueio esta resolvido. O que ele dizia: `records.py` cobra
que `box` saia em pixel do quadro original, o `FrameBatch` carregava `indices`, `tensor` e
`preprocess`, e nenhum dos tres dizia quanto media o quadro antes do redimensionamento. A conta de
volta ja estava escrita em `to_source_box` e era exata; faltava so o par de numeros. Sem ele
`load()` recusava, porque devolver box no espaco do tensor e a falha que este repositorio mais
odeia: o runner monta `Perception` sem reclamar, as sete etapas saem `ok`, e a coordenada esta
errada por um fator que depende do clipe.

O par chegou. O `FrameBatch` carrega `source_size`, e a recusa saiu do `load()` para o `infer()`,
que e onde o lote esta em maos e a medida pode ser lida. Recusar no `load` deixaria de fazer
sentido: em producao o registro constroi com `_resolve(...)()`, sem argumento nenhum, e o tamanho
nunca chegaria por ali.

O `source_size` do construtor continua valendo, como excecao e nao como caminho: lote montado a
mao em teste nao tem clipe de origem, e `make_batch` nao inventa um. Quando os dois existem vale o
do lote, porque ele e a medida do quadro que de fato entrou.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from edge.frames.preprocess import to_source_box as source_box
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
    "o lote nao trouxe `source_size` e o modulo nao recebeu um no construtor, e sem a resolucao do "
    "clipe de origem a caixa sairia em pixel do tensor pre-processado, e nao do quadro original "
    "que `edge.modules.records` cobra. Em producao quem preenche e `edge.frames.provider`; em "
    "teste, construa o modulo com `source_size=(largura, altura)`"
)


def to_source_box(box: Box, preprocess: PreprocessSpec, source: tuple[int, int]) -> Box:
    """Desfaz o enquadramento que `edge.frames.preprocess.prepare` aplicou, e limita ao quadro.

    A conta agora e a de `edge.frames.preprocess.to_source_box`, e nao uma segunda copia dela. A
    daqui foi escrita quando a fundacao nao tinha a funcao, e as duas eram o inverso literal do
    mesmo `_fit`: mesma escala `min(largura_alvo / largura, altura_alvo / altura)` e mesma margem
    por divisao inteira por dois. Duas copias de uma conta exata sao duas oportunidades de
    divergir por um pixel, e o pixel nao apareceria em teste nenhum das duas.

    O que sobra aqui e o que a fundacao nao faz: o vocabulario `Box` deste pacote, e o `clip_box`,
    porque caixa de deteccao pode passar a borda do quadro e quem consome espera pixel valido.
    """
    largura, altura = source
    devolvida = Box(
        *source_box(
            (box.x1, box.y1, box.x2, box.y2),
            preprocess,
            source,
        )
    )
    return clip_box(devolvida, width=largura, height=altura)


class _SourceSpace:
    """A metade comum das duas especies: achar o tamanho do quadro e converter a caixa de volta."""

    spec: ModuleSpec

    def __init__(self, source_size: tuple[int, int] | None) -> None:
        self._source_size = source_size

    def _source_of(self, batch: FrameBatch) -> tuple[int, int]:
        """O lote ganha do construtor: ele e a medida do quadro que de fato entrou.

        O construtor descreve um clipe suposto e nao muda entre chamadas, enquanto o lote vem do
        provedor com a resolucao daquele arquivo. Quando as duas divergem, seguir a do construtor
        deslocaria toda caixa por um fator que ninguem veria.
        """
        tamanho = batch.source_size or self._source_size
        if tamanho is None:
            raise ModuleNotLoaded(f"{self.spec.name}: {SEM_TAMANHO}")
        return tamanho

    def _box(self, box: Box, source: tuple[int, int]) -> tuple[float, float, float, float]:
        convertida = to_source_box(box, self.spec.preprocess, source)
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
        fonte = self._source_of(batch)
        papeis = self._backend.class_roles
        registros = tuple(
            tuple(
                {
                    "box": self._box(raw.box, fonte),
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
        fonte = self._source_of(batch)
        papeis = self._backend.class_roles
        registros: list[tuple[Mapping[str, object], ...]] = []
        for frame in _frames(batch):
            aceitas = [raw for raw in self._backend.track(frame) if raw.class_id in papeis]
            registros.append(
                tuple(
                    {
                        "detection_index": posicao,
                        "track_id": raw.track_id,
                        "box": self._box(raw.box, fonte),
                    }
                    for posicao, raw in enumerate(aceitas)
                )
            )
        return ModuleOutput(kind=self.spec.kind, frames=batch.indices, records=tuple(registros))


__all__ = ["PREPROCESS", "PlayersModule", "TrackingModule", "to_source_box"]
