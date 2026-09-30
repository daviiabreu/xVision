"""Decode unico do clipe e entrega de quadro por indice, com cache em memoria.

E o unico componente que sabe abrir video, e por isso e o unico ponto a trocar quando o decode
passar a ser por hardware na Jetson.

Dois dicionarios, com propositos diferentes: o quadro cru por indice, que e o que faz o
decode acontecer uma vez em vez de uma vez por spec, e o quadro pronto por indice mais
impressao digital da spec, que e o que dois modulos com a mesma spec dividem.

Os dois vivem enquanto o provider viver e `close` os esvazia, entao quem limita o pico de
memoria e quem escolhe os conjuntos de quadros de uma execucao, que e o runner do grafo.

Medido no `config/pipeline.toml` de hoje contra os clipes reais do parceiro, 1920x1080 a
59,94 fps, mediana de 6 s e p95 de 11 s. No clipe de p95 o `coarse` resolve 55 quadros e a
`window` 121, que se sobrepoem em 11, entao `_decoded` guarda 165 quadros crus e 0,96 GiB.
O `_ready` some perto disso, 0,08 GiB, porque as duas specs em uso sao pequenas. Total de
1,03 GiB, num dispositivo de 8 GB.

Quem domina e o quadro cru de 1080p, a 5,93 MiB cada, e nao o tensor preprocessado. Trocando
as sete specs por 640x640 em float32, que e o tamanho tipico de detector, o total vai para
cerca de 3,6 GiB e ainda cabe. O que nao cabe e conjunto `uniform` a fps nativo: os 660
quadros do clipe de p95 sozinhos sao 3,82 GiB em `_decoded`, antes de qualquer spec. O teto real
so a medida no dispositivo responde, e ela ainda nao foi feita.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Literal, Protocol, cast, runtime_checkable

import cv2
import numpy as np

from contracts import Clip
from edge.frames.preprocess import Frame, prepare
from edge.modules.base import FrameBatch, ModuleError, PreprocessSpec, channels, tensor_dtype

if sys.version_info >= (3, 11):  # pragma: no cover - o piso e 3.10, e e nele que a suite roda
    import tomllib
else:
    import tomli as tomllib


class ClipUnavailable(OSError): ...


class MissingFrame(LookupError): ...


class FrameSource(Protocol):
    def read(self, indices: Sequence[int]) -> Mapping[int, Frame]: ...
    def close(self) -> None: ...


@runtime_checkable
class SequentialSource(FrameSource, Protocol):
    def stream(self, indices: Sequence[int]) -> Iterator[tuple[int, Frame]]: ...


Decode = Literal["keep", "redecode"]
DECODES: tuple[Decode, ...] = ("keep", "redecode")
CHAVES_DE_DECODE = frozenset({"mode"})


class DecodeConfigError(ModuleError): ...


def read_decode(source: str | Path) -> Decode:
    """O `[decode].mode` da configuracao, e `keep` quando a secao nao existe.

    Quem le e quem roda o grafo inteiro: `xvision run` e a API. Chave ou valor fora da lista e
    recusado, e nao tratado como padrao: `mdoe = "redecode"` cairia calado em `keep`, e a medida
    de memoria sairia do modo errado com a cara do certo.
    """
    text = source.read_text(encoding="utf-8") if isinstance(source, Path) else source
    documento = cast("dict[str, object]", tomllib.loads(text))
    secao = documento.get("decode", {})
    if not isinstance(secao, dict):
        raise DecodeConfigError(
            f"decode = {secao!r}, e o esperado e a secao [decode] com mode = {' ou '.join(DECODES)}"
        )
    entrada = cast("dict[str, object]", secao)
    sobrando = set(entrada) - CHAVES_DE_DECODE
    if sobrando:
        raise DecodeConfigError(
            f"[decode] declara {', '.join(sorted(sobrando))}, e a unica chave aceita e mode"
        )
    modo = entrada.get("mode", "keep")
    if modo not in DECODES:
        raise DecodeConfigError(f"[decode].mode = {modo!r}, e os aceitos sao {' e '.join(DECODES)}")
    return modo


SEEK_WORTH_IN_FRAMES = 64
"""Quantos quadros seguidos uma busca precisa pular para se pagar.

Medido num clipe real da Footstats, 360 quadros em 1920x1080 a 59,94 fps: ler o clipe
inteiro em sequencia leva 0,31 s, e buscar cada indice leva 27,78 s, 88 vezes mais. Sai
77 ms por busca contra 0,9 ms por quadro pulado, entao a busca so compensa depois de uns
85 quadros. O valor fica abaixo disso de proposito, porque errar para o lado de pular
custa milissegundos e errar para o lado de buscar custa dezenas deles.
"""


class VideoSource:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._capture: cv2.VideoCapture | None = None
        self._sequencial: cv2.VideoCapture | None = None
        self._posicao_sequencial = 0

    def read(self, indices: Sequence[int]) -> Mapping[int, Frame]:
        alvos = sorted(set(indices))
        for indice in alvos:
            if indice < 0:
                raise MissingFrame(f"{self._path} nao tem o quadro {indice}")
        capture = self._open()
        quadros: dict[int, Frame] = {}
        posicao = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        for indice in alvos:
            if indice < posicao or indice - posicao > SEEK_WORTH_IN_FRAMES:
                capture.set(cv2.CAP_PROP_POS_FRAMES, indice)
                posicao = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
                if posicao > indice:
                    raise MissingFrame(
                        f"{self._path} parou no quadro {posicao} ao buscar o {indice}, "
                        "e o decode so anda para a frente"
                    )
            while posicao < indice:
                if not capture.grab():
                    raise MissingFrame(f"{self._path} nao entrega o quadro {indice}")
                posicao += 1
            ok, quadro = capture.read()
            if not ok:
                raise MissingFrame(f"{self._path} nao entrega o quadro {indice}")
            posicao += 1
            quadros[indice] = quadro
        return quadros

    def stream(self, indices: Sequence[int]) -> Iterator[tuple[int, Frame]]:
        """Os quadros pedidos em ordem crescente, lidos em sequencia desde o quadro 0 e sem busca.

        A busca de `read` pousa no quadro-chave e confia no backend para chegar ao alvo exato, o
        que nao vale para todo conteiner. Aqui o quadro tem de ser o da leitura sequencial, entao
        pedido atras da posicao reabre o video e recomeca do inicio em vez de buscar.
        """
        alvos = sorted(set(indices))
        if alvos and alvos[0] < 0:
            raise MissingFrame(f"{self._path} nao tem o quadro {alvos[0]}")
        for indice in alvos:
            capture = self._sequencial
            if capture is None or indice < self._posicao_sequencial:
                capture = self._recomeca()
            while self._posicao_sequencial < indice:
                if not capture.grab():
                    self._fecha_sequencial()
                    raise MissingFrame(f"{self._path} nao entrega o quadro {indice}")
                self._posicao_sequencial += 1
            ok, quadro = capture.read()
            if not ok:
                self._fecha_sequencial()
                raise MissingFrame(f"{self._path} nao entrega o quadro {indice}")
            self._posicao_sequencial += 1
            yield indice, quadro

    @property
    def fps(self) -> float | None:
        """A taxa do arquivo, para o modulo poder raciocinar em segundos e nao em quadros.

        Sem ela um gate de velocidade em pixel por segundo vira pixel por quadro, e a diferenca
        nao aparece: o rastreador continua devolvendo trajetoria, so que sem rejeitar salto
        nenhum. Medido nos clipes do parceiro, o gate desligado deixa passar salto de 125 px por
        quadro onde a fisica da uns 14.
        """
        capture = self._sequencial if self._sequencial is not None else self._open()
        taxa = capture.get(cv2.CAP_PROP_FPS)
        return float(taxa) if taxa and taxa > 0 else None

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._fecha_sequencial()

    def _open(self) -> cv2.VideoCapture:
        if self._capture is None:
            capture = cv2.VideoCapture(str(self._path))
            if not capture.isOpened():
                raise ClipUnavailable(f"{self._path} nao abriu como video")
            self._capture = capture
        return self._capture

    def _recomeca(self) -> cv2.VideoCapture:
        self._fecha_sequencial()
        capture = cv2.VideoCapture(str(self._path))
        if not capture.isOpened():
            raise ClipUnavailable(f"{self._path} nao abriu como video")
        self._sequencial = capture
        return capture

    def _fecha_sequencial(self) -> None:
        if self._sequencial is not None:
            self._sequencial.release()
            self._sequencial = None
        self._posicao_sequencial = 0


class FrameProvider:
    """Decodifica cada quadro uma vez por clipe e prepara direto no lote, sem guardar o preparado.

    O quadro decodificado fica ate o fim do clipe, porque decodificar de novo custa uma passada
    sequencial pelo video. O preparado nao fica. Guardar o de cada especie somava uns 17 MB por
    quadro denso (cru, jogadores, bola e campo) e travou a Jetson no catalogo; guardar so o da
    especie atual ainda levou o unico clipe de 18 s alem dos 7,5 GB da placa. Preparar de novo
    custa milissegundos por quadro.

    Com `decode="redecode"` nenhum quadro decodificado sobrevive a chamada de `frames`: o pico fica
    no lote de uma especie, e o preco e uma leitura sequencial do clipe a mais por especie.
    """

    def __init__(self, source: FrameSource, *, decode: Decode = "keep") -> None:
        self._source = source
        self._decoded: dict[int, Frame] = {}
        self._sequencial: SequentialSource | None = None
        if decode not in DECODES:
            raise ValueError(f"decode={decode!r}, e os aceitos sao {' e '.join(DECODES)}")
        if decode == "redecode":
            if not isinstance(source, SequentialSource):
                raise TypeError(
                    f"decode='redecode' le o video em sequencia, e {type(source).__name__} so "
                    "entrega quadro por indice"
                )
            self._sequencial = source

    def frames(self, indices: Sequence[int], spec: PreprocessSpec) -> FrameBatch:
        if self._sequencial is not None:
            return self._redecode(self._sequencial, indices, spec)
        self._decode_missing(sorted(set(indices)))
        tensor = _tensor_vazio(len(indices), spec)
        for posicao, indice in enumerate(indices):
            tensor[posicao] = prepare(self._decoded[indice], spec)
        return FrameBatch(
            indices=tuple(indices),
            tensor=tensor,
            preprocess=spec,
            source_size=self._source_size(indices),
            source_fps=self._source_fps(),
        )

    def _redecode(
        self, source: SequentialSource, indices: Sequence[int], spec: PreprocessSpec
    ) -> FrameBatch:
        """Cada quadro e preparado assim que sai do decode e descartado antes do proximo.

        O quadro repetido no pedido e preparado uma vez e copiado para as posicoes dele, o que da
        o mesmo tensor de preparar de novo, porque o preparo e deterministico.
        """
        posicoes: dict[int, list[int]] = {}
        for posicao, indice in enumerate(indices):
            posicoes.setdefault(indice, []).append(posicao)
        tensor = _tensor_vazio(len(indices), spec)
        tamanho: tuple[int, int] | None = None
        pendentes = set(posicoes)
        for indice, quadro in source.stream(sorted(posicoes)):
            if indice == indices[0]:
                tamanho = (int(quadro.shape[1]), int(quadro.shape[0]))
            pronto = prepare(quadro, spec)
            for posicao in posicoes[indice]:
                tensor[posicao] = pronto
            pendentes.discard(indice)
        if pendentes:
            raise MissingFrame(f"a fonte nao entregou os quadros {sorted(pendentes)}")
        return FrameBatch(
            indices=tuple(indices),
            tensor=tensor,
            preprocess=spec,
            source_size=tamanho,
            source_fps=self._source_fps(),
        )

    def _source_fps(self) -> float | None:
        """Lido por `getattr` de proposito, e nao por metodo do protocolo `FrameSource`.

        Alargar o protocolo obrigaria todo duble de teste a crescer um metodo que ele nao usa, e
        fonte que nao sabe a taxa e caso legitimo: quem monta quadro em memoria nao tem arquivo.
        """
        taxa = getattr(self._source, "fps", None)
        return float(taxa) if isinstance(taxa, (int, float)) and taxa > 0 else None

    def _source_size(self, indices: Sequence[int]) -> tuple[int, int] | None:
        """O tamanho do quadro decodificado, que so este lado conhece.

        O modulo recebe o tensor ja redimensionado e precisa devolver caixa em pixel do quadro
        original, entao sem este numero nao ha volta possivel.
        """
        if not indices:
            return None
        quadro = self._decoded.get(indices[0])
        return None if quadro is None else (int(quadro.shape[1]), int(quadro.shape[0]))

    def close(self) -> None:
        self._source.close()
        self._decoded.clear()

    def _decode_missing(self, indices: Sequence[int]) -> None:
        pendentes = [indice for indice in indices if indice not in self._decoded]
        if pendentes:
            self._decoded.update(self._source.read(pendentes))


def _tensor_vazio(quantos: int, spec: PreprocessSpec) -> Frame:
    forma = (quantos, spec.height, spec.width, channels(spec.color))
    return np.empty(forma, dtype=tensor_dtype(spec.normalize))


DESFECHOS = frozenset({"off", "on", "block", "gol", "wood"})
CAMPOS_DO_PARCEIRO = 5


def _clip_id(caminho: Path, declarado: str | None) -> str:
    if declarado:
        return declarado
    campos = caminho.stem.split("_")
    if len(campos) == CAMPOS_DO_PARCEIRO and campos[3] in DESFECHOS:
        raise ClipUnavailable(
            f"{caminho.name} segue o padrao do parceiro, e o quarto campo e o desfecho da "
            f"finalizacao. Passe clip_id explicito, senao o rotulo vira prefixo do shot_id e "
            "chega ate a chave do cache"
        )
    return caminho.stem


def probe(path: str | Path, *, clip_id: str | None = None) -> Clip:
    caminho = Path(path)
    identificador = _clip_id(caminho, clip_id)
    capture = cv2.VideoCapture(str(caminho))
    try:
        if not capture.isOpened():
            raise ClipUnavailable(f"{caminho} nao abriu como video")
        largura = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        altura = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        quadros = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if quadros <= 0 or fps <= 0 or largura <= 0 or altura <= 0:
        raise ClipUnavailable(
            f"{caminho} abriu e nao declara metadado utilizavel: {largura}x{altura}, "
            f"{quadros} quadros, {fps} fps"
        )
    return Clip(
        clip_id=identificador,
        path=str(caminho),
        fps=fps,
        frame_count=quadros,
        width=largura,
        height=altura,
    )
