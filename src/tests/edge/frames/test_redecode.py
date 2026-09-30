"""O modo `redecode` do provider: os mesmos quadros do `keep`, sem guardar quadro entre especies.

A identidade e cobrada byte a byte no lote e na entrada de cada modulo, e nao so na saida, porque
os stubs leem o gabarito da fixture e nao o tensor: comparar so a saida passaria com pixel errado.
"""

from __future__ import annotations

import gc
import hashlib
import weakref
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from contracts import Clip
from edge.frames import provider as provider_mod
from edge.frames.preprocess import Frame
from edge.frames.provider import (
    ClipUnavailable,
    Decode,
    DecodeConfigError,
    FrameProvider,
    MissingFrame,
    VideoSource,
    probe,
    read_decode,
)
from edge.graph.framesets import read_framesets
from edge.graph.runner import RunResult, run_pipeline
from edge.modules.base import (
    MODULE_KINDS,
    FrameBatch,
    ModuleInputs,
    ModuleOutput,
    ModuleSpec,
    PerceptionModule,
    PreprocessSpec,
    Temporal,
)
from edge.modules.registry import load_modules, read_config
from edge.testing.make_clip import make_synthetic_clip
from edge.testing.pitch import FPS, SECONDS

CONFIG = Path("config/pipeline.toml")
QUADROS = SECONDS * FPS

RGB = PreprocessSpec(width=32, height=32, color="rgb", letterbox=False, normalize="none")
GRAY = PreprocessSpec(width=16, height=16, color="gray", letterbox=False, normalize="none")
TRATADO = replace(RGB, ops=("bilateral:5:50:50",))
DETECTOR = PreprocessSpec(
    width=64, height=48, color="bgr", letterbox=True, normalize="imagenet", ops=("clahe:2.0:8",)
)

ESPECIES: tuple[tuple[tuple[int, ...], PreprocessSpec], ...] = (
    (tuple(range(0, QUADROS, 5)), RGB),
    (tuple(range(QUADROS)), DETECTOR),
    (tuple(range(3, 24)), TRATADO),
    ((7, 3, 3, 20), GRAY),
    (tuple(range(8, 16)), GRAY),
    (tuple(range(4, 12)), GRAY),
    ((), GRAY),
)
"""Uma passada por especie, na ordem em que o runner as chamaria.

Amostra esparsa, clipe inteiro, janela em volta do chute, pedido fora de ordem com repeticao, dois
lotes temporais em que o segundo volta atras do primeiro, e lote vazio.
"""


class CapturaVigiada:
    """Captura de verdade que conta busca e avanco e guarda referencia fraca a cada quadro lido."""

    def __init__(self, alvo: cv2.VideoCapture, lidos: list[weakref.ref[Frame]]) -> None:
        self._alvo = alvo
        self._lidos = lidos
        self.buscas = 0
        self.avancos = 0
        self.pico: list[int] = []

    def isOpened(self) -> bool:  # noqa: N802
        return bool(self._alvo.isOpened())

    def get(self, prop: int) -> float:
        return float(self._alvo.get(prop))

    def set(self, prop: int, valor: float) -> bool:  # noqa: A003
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.buscas += 1
        return bool(self._alvo.set(prop, valor))

    def grab(self) -> bool:
        self.avancos += 1
        return bool(self._alvo.grab())

    def read(self) -> tuple[bool, Frame]:
        ok, quadro = self._alvo.read()
        if ok:
            self._lidos.append(weakref.ref(quadro))
            self.pico.append(_vivos(self._lidos, coletar=False))
        return ok, quadro

    def release(self) -> None:
        self._alvo.release()


def _vivos(lidos: Sequence[weakref.ref[Frame]], *, coletar: bool = True) -> int:
    if coletar:
        gc.collect()
    return sum(1 for ref in lidos if ref() is not None)


@pytest.fixture
def clipe(tmp_path: Path) -> Path:
    return make_synthetic_clip(tmp_path / "s.mp4", seed=3)


@pytest.fixture
def capturas(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[CapturaVigiada], list[weakref.ref[Frame]]]:
    abertas: list[CapturaVigiada] = []
    lidos: list[weakref.ref[Frame]] = []
    original = cv2.VideoCapture

    def captura(caminho: str) -> CapturaVigiada:
        vigiada = CapturaVigiada(original(caminho), lidos)
        abertas.append(vigiada)
        return vigiada

    monkeypatch.setattr(provider_mod.cv2, "VideoCapture", captura)
    return abertas, lidos


def _lotes(clipe: Path, decode: Decode) -> list[FrameBatch]:
    provider = FrameProvider(VideoSource(clipe), decode=decode)
    try:
        return [provider.frames(indices, spec) for indices, spec in ESPECIES]
    finally:
        provider.close()


def test_redecode_delivers_the_same_batch_as_keep_byte_for_byte(
    clipe: Path,
    capturas: tuple[list[CapturaVigiada], list[weakref.ref[Frame]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Com o limiar de busca em 2, o `keep` chega a parte dos quadros por busca e o `redecode` a
    nenhum, entao a igualdade cobre tambem o quadro que um modo busca e o outro le em sequencia."""
    abertas, _ = capturas
    monkeypatch.setattr(provider_mod, "SEEK_WORTH_IN_FRAMES", 2)

    guardado = _lotes(clipe, "keep")
    buscas_do_keep = sum(c.buscas for c in abertas)
    abertas.clear()
    redecodificado = _lotes(clipe, "redecode")

    assert buscas_do_keep > 0
    assert sum(c.buscas for c in abertas) == 0
    for antes, depois in zip(guardado, redecodificado, strict=True):
        assert depois.indices == antes.indices
        assert depois.preprocess == antes.preprocess
        assert depois.tensor.dtype == antes.tensor.dtype
        assert depois.tensor.shape == antes.tensor.shape
        assert depois.tensor.tobytes() == antes.tensor.tobytes()
        assert depois.source_size == antes.source_size
        assert depois.source_fps == antes.source_fps


def test_redecode_keeps_no_decoded_frame_between_species(
    clipe: Path, capturas: tuple[list[CapturaVigiada], list[weakref.ref[Frame]]]
) -> None:
    abertas, lidos = capturas
    provider = FrameProvider(VideoSource(clipe), decode="redecode")
    try:
        for indices, spec in ESPECIES:
            lote = provider.frames(indices, spec)
            assert lote.indices == indices
            assert _vivos(lidos) == 0, f"quadro decodificado sobreviveu ao lote {indices}"
    finally:
        provider.close()

    assert len(lidos) == sum(len(set(indices)) for indices, _ in ESPECIES), (
        "cada passada decodifica"
    )
    assert max(p for c in abertas for p in c.pico) <= 2, "o anterior e o atual, e nunca o lote"


def test_keep_holds_every_decoded_frame_until_the_clip_ends(
    clipe: Path, capturas: tuple[list[CapturaVigiada], list[weakref.ref[Frame]]]
) -> None:
    """A contraprova do teste acima: a mesma vigilancia enxerga a retencao quando ela existe."""
    abertas, lidos = capturas
    provider = FrameProvider(VideoSource(clipe))
    try:
        for indices, spec in ESPECIES:
            provider.frames(indices, spec)
        assert _vivos(lidos) == QUADROS
    finally:
        provider.close()

    assert len(lidos) == QUADROS, "o padrao decodifica cada quadro uma vez por clipe"
    assert max(p for c in abertas for p in c.pico) == QUADROS
    assert _vivos(lidos) == 0


def test_redecode_never_seeks_and_restarts_from_the_first_frame_when_asked_to_go_back(
    clipe: Path, capturas: tuple[list[CapturaVigiada], list[weakref.ref[Frame]]]
) -> None:
    abertas, _ = capturas
    provider = FrameProvider(VideoSource(clipe), decode="redecode")
    try:
        provider.frames(tuple(range(10)), GRAY)
        provider.frames(tuple(range(10, 20)), GRAY)
        assert len(abertas) == 1, "pedido adiante continua a mesma leitura"
        provider.frames((5,), GRAY)
        assert len(abertas) == 2, "pedido atras reabre o video"
    finally:
        provider.close()

    assert [c.buscas for c in abertas] == [0, 0]
    assert abertas[1].avancos == 5, "do quadro 0 ao 5 em sequencia"


def test_a_redecode_provider_opens_the_video_again_after_close(clipe: Path) -> None:
    provider = FrameProvider(VideoSource(clipe), decode="redecode")
    try:
        primeiro = provider.frames((2, 9), RGB)
        provider.close()
        de_novo = provider.frames((2, 9), RGB)
    finally:
        provider.close()

    assert de_novo.tensor.tobytes() == primeiro.tensor.tobytes()


def test_an_empty_batch_in_redecode_carries_the_shape_the_spec_declares(clipe: Path) -> None:
    provider = FrameProvider(VideoSource(clipe), decode="redecode")
    try:
        lote = provider.frames((), GRAY)
    finally:
        provider.close()

    assert lote.indices == ()
    assert lote.tensor.shape == (0, 16, 16, 1)
    assert lote.source_size is None


def test_redecode_refuses_a_frame_the_clip_does_not_have_and_recovers(clipe: Path) -> None:
    provider = FrameProvider(VideoSource(clipe), decode="redecode")
    try:
        with pytest.raises(MissingFrame, match=str(QUADROS)):
            provider.frames((QUADROS - 1, QUADROS), RGB)
        with pytest.raises(MissingFrame, match="-1"):
            provider.frames((-1, 0), RGB)
        assert provider.frames((0,), RGB).indices == (0,)
    finally:
        provider.close()


def test_redecode_reports_a_clip_that_does_not_open(tmp_path: Path) -> None:
    provider = FrameProvider(VideoSource(tmp_path / "ausente.mp4"), decode="redecode")

    with pytest.raises(ClipUnavailable, match="ausente.mp4"):
        provider.frames((0,), RGB)


class LeituraQueFalha(CapturaVigiada):
    def read(self) -> tuple[bool, Frame]:
        return False, np.empty((0,), dtype=np.uint8)


class AvancoQueFalha(CapturaVigiada):
    def grab(self) -> bool:
        return False


@pytest.mark.parametrize("quebrada", [LeituraQueFalha, AvancoQueFalha])
def test_redecode_reports_the_frame_a_truncated_clip_could_not_deliver(
    clipe: Path, monkeypatch: pytest.MonkeyPatch, quebrada: type[CapturaVigiada]
) -> None:
    original = cv2.VideoCapture

    def captura(caminho: str) -> CapturaVigiada:
        return quebrada(original(caminho), [])

    monkeypatch.setattr(provider_mod.cv2, "VideoCapture", captura)
    provider = FrameProvider(VideoSource(clipe), decode="redecode")
    try:
        with pytest.raises(MissingFrame, match="nao entrega o quadro 3"):
            provider.frames((3,), RGB)
    finally:
        provider.close()


class SoPorIndice:
    def read(self, indices: Sequence[int]) -> Mapping[int, Frame]:
        return {}

    def close(self) -> None: ...


def test_redecode_refuses_a_source_that_only_reads_by_index() -> None:
    with pytest.raises(TypeError, match="SoPorIndice"):
        FrameProvider(SoPorIndice(), decode="redecode")


def test_a_mode_the_provider_does_not_know_is_refused_instead_of_running_as_keep() -> None:
    """Achado ao medir: um script passou `"passada"` como modo e rodou em `keep` calado."""
    modo: Any = "passada"

    with pytest.raises(ValueError, match="passada"):
        FrameProvider(SoPorIndice(), decode=modo)


class PulaUmQuadro(SoPorIndice):
    def stream(self, indices: Sequence[int]) -> Iterator[tuple[int, Frame]]:
        for indice in sorted(set(indices))[1:]:
            yield indice, np.zeros((24, 32, 3), dtype=np.uint8)


def test_a_source_that_skips_a_frame_fails_instead_of_leaving_garbage_in_the_batch() -> None:
    """`np.empty` nao zera: sem a conferencia, a posicao do quadro pulado levaria lixo ao modelo."""
    provider = FrameProvider(PulaUmQuadro(), decode="redecode")

    with pytest.raises(MissingFrame, match=r"\[1\]"):
        provider.frames((1, 2), RGB)


class Gravador:
    """Envolve o modulo de verdade e grava o lote que ele recebeu, com o tensor resumido."""

    def __init__(self, interno: PerceptionModule, entradas: list[tuple[object, ...]]) -> None:
        self.interno = interno
        self.spec: ModuleSpec = interno.spec
        self._entradas = entradas

    def load(self) -> None:
        self.interno.load()

    def close(self) -> None:
        self.interno.close()

    def infer(self, batch: FrameBatch, ctx: ModuleInputs) -> ModuleOutput:
        self._entradas.append(
            (
                self.spec.kind,
                batch.indices,
                batch.tensor.dtype.str,
                batch.tensor.shape,
                hashlib.sha256(batch.tensor.tobytes()).hexdigest(),
                batch.source_size,
                batch.source_fps,
                batch.clip_id,
            )
        )
        return self.interno.infer(batch, ctx)


def _execucao(clip: Clip, decode: Decode) -> tuple[RunResult, list[tuple[object, ...]]]:
    """Os sete stubs da configuracao padrao, com a bola no `coarse` e a pose em lote temporal na
    janela, para as especies pedirem conjuntos diferentes e a pose voltar atras a cada lote."""
    entradas: list[tuple[object, ...]] = []
    modulos = load_modules(read_config(CONFIG))
    modulos["ball"].spec = replace(modulos["ball"].spec, frames="coarse")
    modulos["pose"].spec = replace(
        modulos["pose"].spec, frames="window", batching=Temporal(size=8, stride=4)
    )
    provider = FrameProvider(VideoSource(Path(clip.path)), decode=decode)
    try:
        execucao = run_pipeline(
            clip=clip,
            modules={kind: Gravador(modulo, entradas) for kind, modulo in modulos.items()},
            framesets=read_framesets(CONFIG),
            provider=provider,
        )
    finally:
        provider.close()
    return execucao, entradas


def test_every_species_receives_the_same_input_in_both_modes(clipe: Path) -> None:
    clip = probe(clipe, clip_id="clipe-do-teste")

    guardado, entradas_do_keep = _execucao(clip, "keep")
    redecodificado, entradas_do_redecode = _execucao(clip, "redecode")

    assert all(etapa.status == "ok" for etapa in guardado.stages), guardado.stages
    assert {entrada[0] for entrada in entradas_do_keep} == MODULE_KINDS
    assert entradas_do_redecode == entradas_do_keep
    assert [(e.kind, e.status) for e in redecodificado.stages] == [
        (e.kind, e.status) for e in guardado.stages
    ]
    assert redecodificado.outputs == guardado.outputs
    assert redecodificado.perception == guardado.perception


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("", "keep"),
        ('[frames.dense]\nstrategy = "uniform"\nfps = 15\n', "keep"),
        ('[decode]\nmode = "keep"\n', "keep"),
        ('[decode]\nmode = "redecode"\n', "redecode"),
    ],
)
def test_read_decode_defaults_to_keep_and_reads_the_declared_mode(
    texto: str, esperado: Decode
) -> None:
    assert read_decode(texto) == esperado


@pytest.mark.parametrize(
    ("texto", "motivo"),
    [
        ('[decode]\nmode = "per_species"\n', "mode = 'per_species'"),
        ('[decode]\nmode = "Redecode"\n', "mode = 'Redecode'"),
        ('[decode]\nmode = ""\n', "mode = ''"),
        ("[decode]\nmode = true\n", "mode = True"),
        ("[decode]\nmode = 1\n", "mode = 1"),
        ('[decode]\nmdoe = "redecode"\n', "declara mdoe"),
        ('decode = "redecode"\n', "secao"),
    ],
)
def test_read_decode_refuses_what_it_does_not_know(texto: str, motivo: str) -> None:
    with pytest.raises(DecodeConfigError, match=motivo):
        read_decode(texto)


def test_read_decode_reads_a_file_like_the_other_readers(tmp_path: Path) -> None:
    arquivo = tmp_path / "pipeline.toml"
    arquivo.write_text('[decode]\nmode = "redecode"\n', encoding="utf-8")

    assert read_decode(arquivo) == "redecode"


def test_no_versioned_configuration_changes_the_default() -> None:
    configuracoes = sorted(Path("config").glob("pipeline*.toml"))

    assert configuracoes
    assert {c.name: read_decode(c) for c in configuracoes} == {
        c.name: "keep" for c in configuracoes
    }
