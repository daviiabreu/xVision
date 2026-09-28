"""Diagnostico da fragmentacao do tracking sobre os `detections.json` de um run de jogadores.

Mede, por clipe, quantos IDs o tracking gasta por pessoa visivel, quantas trajetorias morrem com ate
duas amostras e quantas parecem retomadas, que sao trajetorias que nascem logo depois do fim de
outra, perto dela e com o mesmo papel. A pessoa visivel e o pico de deteccoes num quadro.

Os rotulos do agente sao opcionais e foram feitos sobre um run de referencia. Como os runs usam as
mesmas deteccoes, cada deteccao do run medido e ligada pela caixa a trajetoria original, e assim a
troca de ID e a costura continuam medidas mesmo quando os IDs mudaram. O rotulo e de agente, nao de
pessoa, e o numero que sai daqui deve ser apresentado assim.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

CURTA = 2
LACUNA_DE_RETOMADA = 8
CORES = frozenset({"V", "A", "G", "R"})

Chave = tuple[int, int, int, int, int]


@dataclass(frozen=True)
class Amostra:
    posicao: int
    chave: Chave
    box: Mapping[str, float]
    role: str | None


@dataclass(frozen=True)
class Clipe:
    clip_id: str
    pico: int
    deteccoes: int
    sem_id: int
    trajetorias: Mapping[int, tuple[Amostra, ...]]


@dataclass(frozen=True)
class Rotulos:
    times: Mapping[tuple[str, int], frozenset[str]]
    pares: Mapping[tuple[str, int, int], str]
    origem: Mapping[str, Mapping[Chave, int]]


def ler_run(pasta: Path) -> tuple[Clipe, ...]:
    return tuple(_clipe(caminho) for caminho in sorted(pasta.glob("*/detections.json")))


def ler_rotulos(caminho: Path, referencia: Path) -> Rotulos:
    dados = cast("dict[str, Any]", json.loads(caminho.read_text(encoding="utf-8")))
    times = {
        (str(t["clip"]), int(t["track_id"])): frozenset(
            str(c["label"]) for c in t["crops"] if c["label"] in CORES
        )
        for t in dados["tracks"]
    }
    pares = {(str(p["clip"]), int(p["a"]), int(p["b"])): str(p["label"]) for p in dados["pairs"]}
    origem = {
        clipe.clip_id: {a.chave: track_id for track_id, v in clipe.trajetorias.items() for a in v}
        for clipe in ler_run(referencia)
    }
    return Rotulos(times=times, pares=pares, origem=origem)


def diagnosticar(clipe: Clipe, rotulos: Rotulos | None = None) -> dict[str, float | int | str]:
    comprimentos = [len(v) for v in clipe.trajetorias.values()]
    papeis: dict[str | None, set[int]] = defaultdict(set)
    for track_id, amostras in clipe.trajetorias.items():
        papeis[amostras[0].role].add(track_id)
    linha: dict[str, float | int | str] = {
        "clipe": clipe.clip_id,
        "ids": len(clipe.trajetorias),
        "pico": clipe.pico,
        "curtas": sum(1 for n in comprimentos if n <= CURTA),
        "mediana": statistics.median(comprimentos) if comprimentos else 0,
        "retomadas": len(retomadas(clipe.trajetorias)),
        "arbitros": len(papeis["referee"]),
        "goleiros": len(papeis["goalkeeper"]),
        "deteccoes": clipe.deteccoes,
        "sem_id": clipe.sem_id,
    }
    if rotulos is not None:
        linha.update(_contra_rotulos(clipe, rotulos))
    return linha


def retomadas(trajetorias: Mapping[int, Sequence[Amostra]]) -> tuple[tuple[int, int], ...]:
    """A mesma regra do diagnostico do handoff, para o numero de antes continuar comparavel."""
    usados: set[int] = set()
    pares: list[tuple[int, int]] = []
    nascimentos = sorted(trajetorias.items(), key=lambda item: item[1][0].posicao)
    for b, amostras_b in nascimentos:
        inicio = amostras_b[0]
        melhor: tuple[float, int] | None = None
        for a, amostras_a in trajetorias.items():
            fim = amostras_a[-1]
            lacuna = inicio.posicao - fim.posicao
            if a == b or a in usados or not 0 < lacuna <= LACUNA_DE_RETOMADA:
                continue
            if fim.role != inicio.role:
                continue
            distancia = distancia_em_diagonais(fim.box, inicio.box)
            if distancia <= 1.5 * lacuna + 1.5 and (melhor is None or distancia < melhor[0]):
                melhor = (distancia, a)
        if melhor is not None:
            usados.add(melhor[1])
            pares.append((melhor[1], b))
    return tuple(pares)


def distancia_em_diagonais(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    media = (_diagonal(a) + _diagonal(b)) / 2
    return math.dist(_centro(a), _centro(b)) / media if media else math.inf


def _contra_rotulos(clipe: Clipe, rotulos: Rotulos) -> dict[str, float | int]:
    """Troca e a trajetoria que junta duas cores entre times, goleiro e arbitragem.

    Costura e cada par de trajetorias originais que uma trajetoria do run medido uniu, julgado pelo
    par rotulado quando ele existe.
    """
    origem = rotulos.origem.get(clipe.clip_id, {})
    trocas = 0
    certas = erradas = sem_rotulo = 0
    for amostras in clipe.trajetorias.values():
        originais: list[int] = []
        for amostra in amostras:
            original = origem.get(amostra.chave)
            if original is not None and original not in originais:
                originais.append(original)
        cores: frozenset[str] = frozenset[str]().union(
            *(rotulos.times.get((clipe.clip_id, o), ()) for o in originais)
        )
        trocas += len(cores) > 1
        for a, b in zip(originais, originais[1:], strict=False):
            rotulo = rotulos.pares.get((clipe.clip_id, a, b))
            certas += rotulo == "S"
            erradas += rotulo == "D"
            sem_rotulo += rotulo not in {"S", "D"}
    return {
        "trocas": trocas,
        "costuras_S": certas,
        "costuras_D": erradas,
        "costuras_sem": sem_rotulo,
    }


def _clipe(caminho: Path) -> Clipe:
    documento = cast("dict[str, Any]", json.loads(caminho.read_text(encoding="utf-8")))
    quadros = cast("list[dict[str, Any]]", documento["frames"])
    trajetorias: dict[int, list[Amostra]] = defaultdict(list)
    deteccoes = sem_id = 0
    for posicao, quadro in enumerate(quadros):
        for jogador in quadro["players"]:
            deteccoes += 1
            if jogador["track_id"] is None:
                sem_id += 1
                continue
            box = cast("dict[str, float]", jogador["box"])
            chave = (int(quadro["frame_index"]), *(round(box[k]) for k in ("x1", "y1", "x2", "y2")))
            trajetorias[int(jogador["track_id"])].append(
                Amostra(posicao, cast("Chave", chave), box, jogador.get("role"))
            )
    return Clipe(
        clip_id=str(documento["clip"]["id"]),
        pico=max((len(q["players"]) for q in quadros), default=0),
        deteccoes=deteccoes,
        sem_id=sem_id,
        trajetorias={k: tuple(v) for k, v in trajetorias.items()},
    )


def _centro(box: Mapping[str, float]) -> tuple[float, float]:
    return ((box["x1"] + box["x2"]) / 2, (box["y1"] + box["y2"]) / 2)


def _diagonal(box: Mapping[str, float]) -> float:
    return math.hypot(box["x2"] - box["x1"], box["y2"] - box["y1"])
