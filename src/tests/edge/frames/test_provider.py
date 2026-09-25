"""Decode unico e cache de quadro, os dois em memoria e vivos por execucao."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from edge.frames import provider as provider_mod
from edge.frames.preprocess import Frame, prepare
from edge.frames.provider import (
    ClipUnavailable,
    FrameProvider,
    MissingFrame,
    VideoSource,
    probe,
)
from edge.modules.base import PreprocessSpec
from edge.testing.make_clip import make_synthetic_clip
from edge.testing.pitch import FPS, HEIGHT, SECONDS, WIDTH, clip_truth


def iguais(a: Frame, b: Frame) -> bool:
    return a.shape == b.shape and bool((a == b).all())


RGB = PreprocessSpec(width=32, height=32, color="rgb", letterbox=False, normalize="none")
GRAY = PreprocessSpec(width=16, height=16, color="gray", letterbox=False, normalize="none")
TRATADO = replace(RGB, ops=("bilateral:5:50:50",))


class SourceEspia:
    def __init__(self, total: int = 8) -> None:
        rng = np.random.default_rng(11)
        self.quadros = rng.integers(0, 256, size=(total, 24, 32, 3), dtype=np.uint8)
        self.pedidos: list[tuple[int, ...]] = []
        self.fechado = False

    def read(self, indices: Sequence[int]) -> Mapping[int, Frame]:
        self.pedidos.append(tuple(indices))
        for indice in indices:
            if indice >= len(self.quadros):
                raise MissingFrame(f"a fonte de teste nao entrega o quadro {indice}")
        return {indice: self.quadros[indice] for indice in indices}

    def close(self) -> None:
        self.fechado = True

    @property
    def decodificados(self) -> list[int]:
        return [indice for pedido in self.pedidos for indice in pedido]


def test_decode_called_once_for_repeated_indices() -> None:
    espia = SourceEspia()
    provider = FrameProvider(espia)

    lote = provider.frames([2, 2, 5, 2], RGB)

    assert espia.pedidos == [(2, 5)]
    assert lote.indices == (2, 2, 5, 2)
    assert lote.tensor.shape == (4, 32, 32, 3)
    for posicao, indice in enumerate((2, 2, 5, 2)):
        assert iguais(lote.tensor[posicao], prepare(espia.quadros[indice], RGB))


def test_same_spec_hits_frame_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    preparos: list[int] = []

    def contado(frame: Frame, spec: PreprocessSpec) -> Frame:
        preparos.append(1)
        return prepare(frame, spec)

    monkeypatch.setattr(provider_mod, "prepare", contado)
    espia = SourceEspia()
    provider = FrameProvider(espia)

    primeiro = provider.frames([1, 3], RGB)
    segundo = provider.frames([1, 3], RGB)

    assert espia.pedidos == [(1, 3)]
    assert len(preparos) == 2
    assert iguais(primeiro.tensor, segundo.tensor)


def test_different_spec_does_not_collide_in_cache() -> None:
    provider = FrameProvider(SourceEspia())

    cru = provider.frames([4], RGB)
    tratado = provider.frames([4], TRATADO)

    assert cru.tensor.shape == tratado.tensor.shape == (1, 32, 32, 3)
    assert not iguais(cru.tensor, tratado.tensor)
    assert iguais(provider.frames([4], RGB).tensor, cru.tensor)
    assert iguais(provider.frames([4], TRATADO).tensor, tratado.tensor)


def test_an_empty_batch_carries_the_shape_the_spec_declares() -> None:
    lote = FrameProvider(SourceEspia()).frames([], GRAY)

    assert lote.indices == ()
    assert lote.tensor.shape == (0, 16, 16, 1)


def test_switching_spec_drops_the_prepared_frames_but_keeps_the_decoded_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparos: list[str] = []

    def contado(frame: Frame, spec: PreprocessSpec) -> Frame:
        preparos.append(spec.color)
        return prepare(frame, spec)

    monkeypatch.setattr(provider_mod, "prepare", contado)
    espia = SourceEspia()
    provider = FrameProvider(espia)

    primeiro = provider.frames([0, 1], RGB)
    provider.frames([0, 1], GRAY)
    de_novo = provider.frames([0, 1], RGB)

    assert espia.decodificados == [0, 1]
    assert preparos == ["rgb", "rgb", "gray", "gray", "rgb", "rgb"]
    assert iguais(primeiro.tensor, de_novo.tensor)


def test_second_spec_over_the_same_indices_does_not_decode_again() -> None:
    espia = SourceEspia()
    provider = FrameProvider(espia)

    provider.frames([0, 1], RGB)
    provider.frames([0, 1], GRAY)

    assert espia.decodificados == [0, 1]


def test_index_the_source_cannot_deliver_raises_instead_of_shrinking_the_batch(
    tmp_path: Path,
) -> None:
    clip = make_synthetic_clip(tmp_path / "s.mp4")
    provider = FrameProvider(VideoSource(clip))

    try:
        with pytest.raises(MissingFrame, match=str(SECONDS * FPS)):
            provider.frames([SECONDS * FPS], RGB)
        with pytest.raises(MissingFrame, match="-1"):
            provider.frames([-1], RGB)
    finally:
        provider.close()


def test_close_closes_the_source_and_empties_the_caches() -> None:
    espia = SourceEspia()
    provider = FrameProvider(espia)
    provider.frames([0], RGB)

    provider.close()

    assert espia.fechado
    provider.frames([0], RGB)
    assert espia.decodificados == [0, 0]


def test_video_source_reads_the_declared_indices_of_a_real_clip(tmp_path: Path) -> None:
    clip = make_synthetic_clip(tmp_path / "s.mp4", seed=2)
    source = VideoSource(clip)

    try:
        quadros = source.read([0, 7, 19])
    finally:
        source.close()

    assert sorted(quadros) == [0, 7, 19]
    assert all(quadro.shape == (HEIGHT, WIDTH, 3) for quadro in quadros.values())
    assert not iguais(quadros[0], quadros[19])


def test_video_source_seek_matches_a_sequential_read(tmp_path: Path) -> None:
    clip = make_synthetic_clip(tmp_path / "s.mp4", seed=5)
    sequencial = _le_do_inicio_ao_fim(clip)
    source = VideoSource(clip)

    try:
        salteado = source.read([13, 4])
    finally:
        source.close()

    assert len(sequencial) == SECONDS * FPS
    assert iguais(salteado[13], sequencial[13])
    assert iguais(salteado[4], sequencial[4])


def _le_do_inicio_ao_fim(clip: Path) -> list[Frame]:
    capture = cv2.VideoCapture(str(clip))
    quadros: list[Frame] = []
    try:
        while True:
            ok, quadro = capture.read()
            if not ok:
                return quadros
            quadros.append(quadro)
    finally:
        capture.release()


def test_clip_that_does_not_open_is_a_different_failure_from_a_missing_frame(
    tmp_path: Path,
) -> None:
    with pytest.raises(ClipUnavailable, match="ausente.mp4"):
        VideoSource(tmp_path / "ausente.mp4").read([0])

    assert not issubclass(ClipUnavailable, MissingFrame)


class CapturaContada:
    def __init__(self, alvo: cv2.VideoCapture) -> None:
        self._alvo = alvo
        self.buscas = 0
        self.leituras = 0

    def isOpened(self) -> bool:  # noqa: N802
        return bool(self._alvo.isOpened())

    def get(self, prop: int) -> float:
        return float(self._alvo.get(prop))

    def set(self, prop: int, valor: float) -> bool:  # noqa: A003
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.buscas += 1
        return bool(self._alvo.set(prop, valor))

    def grab(self) -> bool:
        return bool(self._alvo.grab())

    def read(self) -> tuple[bool, Frame]:
        self.leituras += 1
        return self._alvo.read()

    def release(self) -> None:
        self._alvo.release()


def test_reading_a_run_of_frames_does_not_seek_once_per_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = make_synthetic_clip(tmp_path / "s.mp4")
    contadas: list[CapturaContada] = []

    original = cv2.VideoCapture

    def captura(caminho: str) -> CapturaContada:
        contada = CapturaContada(original(caminho))
        contadas.append(contada)
        return contada

    monkeypatch.setattr(provider_mod.cv2, "VideoCapture", captura)
    source = VideoSource(clip)
    try:
        quadros = source.read(range(SECONDS * FPS))
    finally:
        source.close()

    assert len(quadros) == SECONDS * FPS
    assert contadas[0].buscas <= 1, "o clipe inteiro em ordem nao precisa de mais de uma busca"


def test_a_sparse_set_of_frames_seeks_instead_of_decoding_everything_in_between(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = make_synthetic_clip(tmp_path / "s.mp4")
    monkeypatch.setattr(provider_mod, "SEEK_WORTH_IN_FRAMES", 2)
    contadas: list[CapturaContada] = []

    original = cv2.VideoCapture

    def captura(caminho: str) -> CapturaContada:
        contada = CapturaContada(original(caminho))
        contadas.append(contada)
        return contada

    monkeypatch.setattr(provider_mod.cv2, "VideoCapture", captura)
    source = VideoSource(clip)
    try:
        source.read([0, SECONDS * FPS - 1])
    finally:
        source.close()

    assert contadas[0].buscas >= 1
    assert contadas[0].leituras == 2, "so os dois quadros pedidos sao convertidos"


def test_a_backend_that_overshoots_the_seek_fails_instead_of_returning_the_wrong_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = make_synthetic_clip(tmp_path / "s.mp4")
    monkeypatch.setattr(provider_mod, "SEEK_WORTH_IN_FRAMES", 2)

    class CapturaQuePassaDoAlvo(CapturaContada):
        def get(self, prop: int) -> float:
            if prop == cv2.CAP_PROP_POS_FRAMES and self.buscas:
                return 999.0
            return super().get(prop)

    original = cv2.VideoCapture

    def captura(caminho: str) -> CapturaQuePassaDoAlvo:
        return CapturaQuePassaDoAlvo(original(caminho))

    monkeypatch.setattr(provider_mod.cv2, "VideoCapture", captura)
    source = VideoSource(clip)
    try:
        with pytest.raises(MissingFrame, match="so anda para a frente"):
            source.read([0, SECONDS * FPS - 1])
    finally:
        source.close()


def test_a_clip_that_ends_mid_read_reports_the_frame_it_could_not_reach(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = make_synthetic_clip(tmp_path / "s.mp4")

    class CapturaTruncada(CapturaContada):
        def grab(self) -> bool:
            return False

    original = cv2.VideoCapture

    def captura(caminho: str) -> CapturaTruncada:
        return CapturaTruncada(original(caminho))

    monkeypatch.setattr(provider_mod.cv2, "VideoCapture", captura)
    source = VideoSource(clip)
    try:
        with pytest.raises(MissingFrame, match=f"nao entrega o quadro {SECONDS * FPS - 1}"):
            source.read([SECONDS * FPS - 1])
    finally:
        source.close()


def test_the_seek_rule_turns_over_exactly_at_the_declared_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = make_synthetic_clip(tmp_path / "s.mp4")
    monkeypatch.setattr(provider_mod, "SEEK_WORTH_IN_FRAMES", 4)
    original = cv2.VideoCapture
    contadas: list[CapturaContada] = []

    def captura(caminho: str) -> CapturaContada:
        contada = CapturaContada(original(caminho))
        contadas.append(contada)
        return contada

    monkeypatch.setattr(provider_mod.cv2, "VideoCapture", captura)

    for salto, buscas_esperadas in ((5, 0), (6, 1)):
        contadas.clear()
        source = VideoSource(clip)
        try:
            source.read([0, salto])
        finally:
            source.close()
        esperado = f"{salto - 1} avancos com limiar 4"
        assert contadas[0].buscas == buscas_esperadas, esperado


def test_frames_come_back_by_index_no_matter_the_order_they_were_asked_for(tmp_path: Path) -> None:
    clip = make_synthetic_clip(tmp_path / "s.mp4")
    referencia = VideoSource(clip)
    try:
        esperado = dict(referencia.read(range(SECONDS * FPS)))
    finally:
        referencia.close()

    source = VideoSource(clip)
    try:
        for pedido in ([7, 3, 3, 20], [0], [29, 1], [15, 15, 2], list(range(SECONDS * FPS))):
            for indice, quadro in source.read(pedido).items():
                assert (quadro == esperado[indice]).all(), f"quadro {indice} apos o pedido {pedido}"
    finally:
        source.close()


def test_probe_reads_the_clip_metadata_the_pipeline_needs(tmp_path: Path) -> None:
    clip = make_synthetic_clip(tmp_path / "s.mp4")

    lido = probe(clip)

    gabarito = clip_truth()
    assert (lido.width, lido.height) == (gabarito.width, gabarito.height)
    assert lido.frame_count == gabarito.frame_count
    assert lido.fps == pytest.approx(float(gabarito.fps))
    assert lido.clip_id == "s"


def test_probe_on_something_that_is_not_a_video_refuses_like_the_source_does(
    tmp_path: Path,
) -> None:
    quebrado = tmp_path / "truncado.mp4"
    quebrado.write_bytes(b"nao e video")

    with pytest.raises(ClipUnavailable, match="truncado.mp4"):
        probe(quebrado)


def test_probe_refuses_a_clip_that_opens_and_declares_nothing_usable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contêiner que abre e reporta zero quadro daria divisao por zero la no pipeline."""
    clip = make_synthetic_clip(tmp_path / "s.mp4")

    class SemMetadado(CapturaContada):
        def get(self, prop: int) -> float:
            return 0.0

    original = cv2.VideoCapture

    def captura(caminho: str) -> SemMetadado:
        return SemMetadado(original(caminho))

    monkeypatch.setattr(provider_mod.cv2, "VideoCapture", captura)

    with pytest.raises(ClipUnavailable, match="metadado"):
        probe(clip)


def test_probe_takes_the_clip_id_instead_of_inventing_one_from_the_file_name(
    tmp_path: Path,
) -> None:
    """O nome do arquivo do parceiro carrega o desfecho, e em xG o desfecho e o alvo.

    Medido: 78 dos 633 clipes tem `gol` no nome, e o quarto campo e sempre um de `off`, `on`,
    `block`, `gol` ou `wood`. Com o nome como identificador, o rotulo vira prefixo do `shot_id` de
    todo evento e chega ate a chave do cache de percepcao.
    """
    clip = make_synthetic_clip(tmp_path / "900_004_Mexico_gol_4995949.mp4")

    assert probe(clip, clip_id="4995949").clip_id == "4995949"


def test_probe_without_a_clip_id_refuses_a_partner_style_name(tmp_path: Path) -> None:
    clip = make_synthetic_clip(tmp_path / "900_004_Mexico_gol_4995949.mp4")

    with pytest.raises(ClipUnavailable, match="clip_id"):
        probe(clip)
