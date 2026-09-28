"""Adaptador do ByteTrack para as caixas produzidas pelo modulo central de jogadores.

O rastreador recebe apenas as deteccoes existentes, sem abrir video nem executar outro modelo. O
estado nasce dentro de `infer`, portanto duas chamadas com a mesma entrada continuam deterministas
e um clipe nunca herda trajetorias do anterior.

O lote e o mesmo quadro BGR que `players` recebe, e nao o cinza minimo do greedy, porque a costura
de fragmentos compara a cor da camisa. Como o preprocess e igual ao de `players`, o runner nao
prepara um segundo tensor.

Deteccao sem ID fica sem ID, decisao tomada depois de medir. O ByteTrack so devolve trajetoria
confirmada, entao a deteccao que ele nao casou nem confirmou sai sem `track_id`, e foram 1.265 de
11.927 no piloto de 8 clipes, 10,6%, todas com confianca acima de 0,25. Delas, 539 se sobrepoem com
IoU acima de 0,3 a uma deteccao rastreada no mesmo quadro, que e oclusao ou caixa duplicada, 576
estao perto de uma pessoa rastreada no quadro vizinho, inclusive o primeiro quadro de uma trajetoria
nova, e 150 estao sozinhas. Dar ID novo, mesmo so a confianca alta, criaria duplicata ou fragmento
de uma amostra em quase todas e pioraria os IDs por pessoa visivel, que a costura existe para
reduzir. A deteccao continua na percepcao com `track_id` nulo, sem equipe, numero nem pose, e as
features que precisam de trajetoria ja a ignoram.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from importlib import import_module
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from contracts import Box
from edge.frames.preprocess import to_tensor_box
from edge.modules.base import (
    FrameBatch,
    ModuleInputs,
    ModuleNotLoaded,
    ModuleOutput,
    ModuleSpec,
    check_preprocess,
)
from edge.modules.players import PREPROCESS as PLAYERS_PREPROCESS
from edge.modules.tracking.greedy import (
    PAPEIS,
    Deteccao,
    com_papeis_estaveis,
    deteccoes_do_quadro,
    papel_estavel,
    por_quadro,
)
from edge.participants.identity import (
    LabColor,
    extract_shirt_color,
    representative_shirt_color,
)

PREPROCESS = PLAYERS_PREPROCESS

TRACK_HIGH_THRESHOLD = 0.25
TRACK_LOW_THRESHOLD = 0.1
NEW_TRACK_THRESHOLD = 0.25
TRACK_BUFFER = 30
MATCH_THRESHOLD = 0.8

LACUNA_DE_COSTURA = 4
"""Quantas amostras depois do fim de A a trajetoria B ainda pode nascer para virar A."""

ALCANCE_DE_COSTURA = 0.75
"""Distancia maxima entre o fim de A e o inicio de B, em diagonais medias das duas caixas.

A lacuna e o alcance comecaram em 8 e 1,5 e foram apertados depois do teste em tres partidas fora
do piloto, 237079, 237107 e 237119: la a precisao caiu para 0,82, e os erros que a cor nao separa
sao colegas do mesmo time lado a lado. Juntar duas pessoas estraga toda feature que le a trajetoria,
e um fragmento a mais so custa um ID, entao a regra fica do lado conservador.
"""

AMOSTRAS_DE_COR = 3
"""Quantas amostras de cada ponta dao a cor, pela mediana, para um recorte ruim nao decidir."""

DISTANCIA_DE_COR = 10.0
"""Distancia maxima de crominancia, os canais a e b do Lab de 8 bits do OpenCV, entre as pontas.

A crominancia aguenta sol e sombra na mesma camisa, que mudam o brilho e nao o tom.
"""

DIFERENCA_DE_LUMINOSIDADE = 40.0
"""Diferenca maxima do canal L entre as pontas, porque camisa branca, preta ou cinza nao tem tom.

Sem este limite, branco e preto tem a mesma crominancia e passam juntos, que foi o erro mais comum
nas partidas novas. O folgado 40 deixa passar a mesma camisa entre sol e sombra.

Os limiares foram medidos no tensor do pipeline, com rotulo de agente, nos candidatos do piloto,
237053, e numa amostra de 350 pares das tres partidas novas. Com a regra inteira ficam 22 pares
da mesma pessoa e nenhum de outra no piloto, e 80 contra 3 nas partidas novas, precisao de 0,96.
Os mesmos rotulos escolheram e medem a regra, entao nao sobra amostra limpa, e o numero e de
calibracao, nao de teste.
"""

Amostra = tuple[int, Box]
"""Posicao no lote e caixa em pixel do quadro original."""


class _Results:
    """Subconjunto do objeto de resultados que o ByteTrack consome."""

    def __init__(
        self, xywh: NDArray[np.float32], conf: NDArray[np.float32], cls: NDArray[np.float32]
    ) -> None:
        self.xywh = np.asarray(xywh, dtype=np.float32).reshape((-1, 4))
        self.conf = np.asarray(conf, dtype=np.float32).reshape((-1,))
        self.cls = np.asarray(cls, dtype=np.float32).reshape((-1,))

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, index: Any) -> _Results:
        return _Results(self.xywh[index], self.conf[index], self.cls[index])


class ByteTrackTracking:
    spec = ModuleSpec(
        kind="tracking",
        name="bytetrack",
        version="0.2.0",
        frames="window",
        preprocess=PREPROCESS,
        needs=("players",),
    )

    def __init__(
        self,
        *,
        track_high_thresh: float = TRACK_HIGH_THRESHOLD,
        track_low_thresh: float = TRACK_LOW_THRESHOLD,
        new_track_thresh: float = NEW_TRACK_THRESHOLD,
        track_buffer: int = TRACK_BUFFER,
        match_thresh: float = MATCH_THRESHOLD,
        fuse_score: bool = True,
        costura: bool = True,
    ) -> None:
        self._costura = costura
        self._args = SimpleNamespace(
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            fuse_score=fuse_score,
        )
        self._tracker_class: Any | None = None

    def configure(self, options: Mapping[str, object]) -> None:
        """So `costura` e ajustavel, e desligar serve para rotular as trajetorias antes da costura.

        Rotular depois da costura esconderia justamente o erro que se quer medir, porque o par
        costurado errado ja viraria uma trajetoria so.
        """
        desconhecidas = sorted(set(options) - {"costura"})
        if desconhecidas:
            raise ValueError(f"tracking nao aceita {', '.join(desconhecidas)}, so costura")
        costura = options.get("costura", self._costura)
        if not isinstance(costura, bool):
            raise ValueError("tracking.costura espera booleano")
        self._costura = costura

    def fingerprint(self) -> str:
        """Entra na chave do cache, para a saida sem costura nao ser servida como a costurada."""
        return f"costura={self._costura}"

    def load(self) -> None:
        modulo = import_module("ultralytics.trackers.byte_tracker")
        self._tracker_class = cast("Any", modulo).BYTETracker

    def close(self) -> None:
        self._tracker_class = None

    def infer(self, batch: FrameBatch, ctx: ModuleInputs) -> ModuleOutput:
        if self._tracker_class is None:
            raise ModuleNotLoaded(self.spec.name)
        check_preprocess(batch, self.spec.preprocess)
        if not batch.indices:
            return ModuleOutput(kind=self.spec.kind, frames=(), records=())

        tracker = self._tracker_class(self._args)
        jogadores = por_quadro(ctx.get("players"))
        registros: list[tuple[Mapping[str, object], ...]] = []
        papeis: dict[int, dict[str, float]] = {}
        trajetorias: dict[int, list[Amostra]] = {}
        for posicao, indice in enumerate(batch.indices):
            deteccoes = deteccoes_do_quadro(jogadores.get(indice, ()))
            do_quadro, observacoes = _rastrear_quadro(tracker, deteccoes)
            registros.append(do_quadro)
            caixas = {d.detection_index: d.box for d in deteccoes}
            for registro in do_quadro:
                track_id = int(cast("Any", registro["track_id"]))
                caixa = caixas[int(cast("Any", registro["detection_index"]))]
                trajetorias.setdefault(track_id, []).append((posicao, caixa))
            for track_id, papel, confianca in observacoes:
                por_papel = papeis.setdefault(track_id, {})
                por_papel[papel] = por_papel.get(papel, 0.0) + confianca

        destino = (
            costurar(
                trajetorias,
                {track_id: papel_estavel(por_papel) for track_id, por_papel in papeis.items()},
                lambda amostras: _cor_representativa(batch, amostras),
            )
            if self._costura
            else {}
        )
        for origem, alvo in destino.items():
            por_papel = papeis.setdefault(alvo, {})
            for papel, confianca in papeis.pop(origem, {}).items():
                por_papel[papel] = por_papel.get(papel, 0.0) + confianca
        costurados = [
            tuple(
                {**r, "track_id": destino.get(int(cast("Any", r["track_id"])), r["track_id"])}
                for r in quadro
            )
            for quadro in registros
        ]
        return ModuleOutput(
            kind=self.spec.kind,
            frames=batch.indices,
            records=com_papeis_estaveis(costurados, papeis),
        )


def costurar(
    trajetorias: Mapping[int, Sequence[Amostra]],
    papeis: Mapping[int, str | None],
    cor: Callable[[Sequence[Amostra]], LabColor | None],
) -> dict[int, int]:
    """Junta a trajetoria que nasce logo depois do fim de outra, e devolve o ID que cada uma herda.

    B vira A quando nasce ate `LACUNA_DE_COSTURA` amostras depois do fim de A, a no maximo
    `ALCANCE_DE_COSTURA` diagonais de caixa, com o mesmo papel e cor compativel no tom e no brilho.
    Entre as candidatas fica a mais proxima. A cadeia continua: depois de B entrar em A, o fim de A
    passa a ser o fim de B, e uma terceira trajetoria pode entrar na mesma. Sem cor medida nas duas
    pontas a costura nao acontece, porque juntar duas pessoas custa mais que deixar uma fragmentada.
    """
    destino: dict[int, int] = {}
    cauda = {track_id: tuple(amostras) for track_id, amostras in trajetorias.items()}
    cores: dict[tuple[int, str], LabColor | None] = {}

    def cor_da_ponta(track_id: int, ponta: str) -> LabColor | None:
        if (track_id, ponta) not in cores:
            amostras = trajetorias[track_id]
            pedaco = (
                cauda[track_id][-AMOSTRAS_DE_COR:] if ponta == "fim" else amostras[:AMOSTRAS_DE_COR]
            )
            cores[(track_id, ponta)] = cor(pedaco)
        return cores[(track_id, ponta)]

    for b in sorted(trajetorias, key=lambda t: (trajetorias[t][0][0], t)):
        inicio = trajetorias[b][0]
        melhor: tuple[float, int] | None = None
        for a, amostras_a in cauda.items():
            fim = amostras_a[-1]
            if a == b or not 0 < inicio[0] - fim[0] <= LACUNA_DE_COSTURA:
                continue
            if papeis.get(a) != papeis.get(b):
                continue
            distancia = _em_diagonais(fim[1], inicio[1])
            if distancia > ALCANCE_DE_COSTURA or (melhor is not None and distancia >= melhor[0]):
                continue
            cor_a, cor_b = cor_da_ponta(a, "fim"), cor_da_ponta(b, "inicio")
            if not _cores_compativeis(cor_a, cor_b):
                continue
            melhor = (distancia, a)
        if melhor is None:
            continue
        a = melhor[1]
        destino[b] = a
        cauda[a] = tuple(trajetorias[b])
        cores.pop((a, "fim"), None)
        del cauda[b]
    return destino


def _rastrear_quadro(
    tracker: Any, deteccoes: Sequence[Deteccao]
) -> tuple[
    tuple[Mapping[str, object], ...],
    tuple[tuple[int, str, float], ...],
]:
    resultados = tracker.update(_resultados(deteccoes))
    registros: list[Mapping[str, object]] = []
    observacoes: list[tuple[int, str, float]] = []
    for linha in resultados:
        posicao = int(linha[-1])
        if not 0 <= posicao < len(deteccoes):
            continue
        deteccao = deteccoes[posicao]
        track_id = int(linha[4])
        registros.append(
            {
                "detection_index": deteccao.detection_index,
                "track_id": track_id,
            }
        )
        if deteccao.role in PAPEIS:
            observacoes.append((track_id, cast("str", deteccao.role), deteccao.confidence))
    registros.sort(key=lambda registro: int(cast("Any", registro["detection_index"])))
    return tuple(registros), tuple(observacoes)


def _resultados(deteccoes: Sequence[Deteccao]) -> _Results:
    xywh: list[tuple[float, float, float, float]] = []
    confiancas: list[float] = []
    classes: list[int] = []
    for deteccao in deteccoes:
        caixa = deteccao.box
        xywh.append(
            (
                (caixa.x1 + caixa.x2) / 2.0,
                (caixa.y1 + caixa.y2) / 2.0,
                caixa.x2 - caixa.x1,
                caixa.y2 - caixa.y1,
            )
        )
        confiancas.append(deteccao.confidence)
        classes.append(PAPEIS.index(deteccao.role) if deteccao.role in PAPEIS else len(PAPEIS))
    return _Results(np.asarray(xywh), np.asarray(confiancas), np.asarray(classes))


def cor_da_camisa(batch: FrameBatch, posicao: int, box: Box) -> LabColor | None:
    """Cor da camisa no quadro `posicao` do lote, com a caixa de `players` levada ao tensor."""
    caixa = (box.x1, box.y1, box.x2, box.y2)
    if batch.source_size is not None:
        caixa = to_tensor_box(caixa, batch.preprocess, batch.source_size)
    return extract_shirt_color(batch.tensor[posicao], Box(*caixa))


def _cores_compativeis(a: LabColor | None, b: LabColor | None) -> bool:
    if a is None or b is None:
        return False
    return (
        math.dist(a[1:], b[1:]) <= DISTANCIA_DE_COR
        and abs(a[0] - b[0]) <= DIFERENCA_DE_LUMINOSIDADE
    )


def _cor_representativa(batch: FrameBatch, amostras: Sequence[Amostra]) -> LabColor | None:
    cores = [cor_da_camisa(batch, posicao, caixa) for posicao, caixa in amostras]
    return representative_shirt_color([c for c in cores if c is not None])


def _em_diagonais(a: Box, b: Box) -> float:
    media = (math.hypot(a.x2 - a.x1, a.y2 - a.y1) + math.hypot(b.x2 - b.x1, b.y2 - b.y1)) / 2
    centro_a = ((a.x1 + a.x2) / 2, (a.y1 + a.y2) / 2)
    centro_b = ((b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2)
    return math.dist(centro_a, centro_b) / media if media > 0 else math.inf
