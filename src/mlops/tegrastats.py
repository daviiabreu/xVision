"""Memoria, potencia e temperatura de uma execucao na Jetson, lidas do log do `tegrastats`.

A revisao da Sprint 3 cobrou memoria e energia, e nenhuma das duas tinha sido medida: o `run.json`
grava tempo por especie, mas nao quanto a placa gastou para produzi-lo. O `tegrastats` ja registra
tudo uma vez por intervalo, e este modulo so le o log e resume.

A energia sai da potencia instantanea de `VDD_IN`, a entrada da placa inteira, integrada no tempo.
A media que o proprio `tegrastats` imprime ao lado e acumulada desde que ele comecou, e por isso nao
e usada: ela depende de quando o log foi ligado, e nao so do que a execucao fez.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from pathlib import Path

LINHA = re.compile(
    r"^(?P<quando>\d\d-\d\d-\d{4} \d\d:\d\d:\d\d) RAM (?P<ram>\d+)/(?P<total>\d+)MB"
    r"(?:.*?SWAP (?P<swap>\d+)/\d+MB)?"
    r".*?GR3D_FREQ (?P<gpu>\d+)%"
    r".*?tj@(?P<tj>[\d.]+)C"
    r".*?VDD_IN (?P<vdd>\d+)mW/"
)
FORMATO = "%m-%d-%Y %H:%M:%S"


class TegrastatsError(ValueError): ...


@dataclass(frozen=True)
class Amostra:
    quando: datetime
    ram_mb: int
    ram_total_mb: int
    swap_mb: int
    gpu_pct: int
    tj_c: float
    vdd_in_mw: int


def le(caminho: Path) -> tuple[Amostra, ...]:
    if not caminho.is_file():
        raise TegrastatsError(f"{caminho} nao existe")
    amostras: list[Amostra] = []
    for line_number, linha in enumerate(caminho.read_text(encoding="utf-8").splitlines(), start=1):
        achado = LINHA.match(linha)
        if achado is None:
            continue
        try:
            amostras.append(
                Amostra(
                    quando=datetime.strptime(achado["quando"], FORMATO),
                    ram_mb=int(achado["ram"]),
                    ram_total_mb=int(achado["total"]),
                    swap_mb=int(achado["swap"] or 0),
                    gpu_pct=int(achado["gpu"]),
                    tj_c=float(achado["tj"]),
                    vdd_in_mw=int(achado["vdd"]),
                )
            )
        except ValueError as error:
            raise TegrastatsError(
                f"{caminho.name}:{line_number}: amostra invalida: {error}"
            ) from error
    if len(amostras) < 2:
        raise TegrastatsError(f"{caminho} tem menos de duas amostras reconheciveis")
    return tuple(amostras)


def resume(amostras: Sequence[Amostra], clipes: int | None = None) -> dict[str, object]:
    """O pico de memoria, a potencia e a energia integrada, com a energia por clipe quando dada.

    A integracao e por trapezios entre amostras consecutivas. Com o `tegrastats` a um segundo, o
    relogio do log tem resolucao de um segundo, entao a duracao sai com esse grao.
    """
    if len(amostras) < 2:
        raise TegrastatsError("o resumo precisa de pelo menos duas amostras")
    if clipes is not None and clipes <= 0:
        raise TegrastatsError("clipes precisa ser positivo")
    energia_j = 0.0
    for antes, depois in pairwise(amostras):
        intervalo = (depois.quando - antes.quando).total_seconds()
        if intervalo < 0:
            raise TegrastatsError("amostras fora de ordem cronologica")
        energia_j += intervalo * (antes.vdd_in_mw + depois.vdd_in_mw) / 2000
    duracao = (amostras[-1].quando - amostras[0].quando).total_seconds()
    if duracao <= 0:
        raise TegrastatsError("a duracao total precisa ser positiva")
    potencias = sorted(a.vdd_in_mw for a in amostras)
    resumo: dict[str, object] = {
        "amostras": len(amostras),
        "duracao_s": duracao,
        "ram_total_mb": amostras[0].ram_total_mb,
        "ram_pico_mb": max(a.ram_mb for a in amostras),
        "swap_pico_mb": max(a.swap_mb for a in amostras),
        "gpu_ocupada": round(sum(a.gpu_pct > 0 for a in amostras) / len(amostras), 3),
        "potencia_media_w": round(energia_j / duracao, 2),
        "potencia_mediana_w": round(potencias[len(potencias) // 2] / 1000, 2),
        "potencia_pico_w": round(potencias[-1] / 1000, 2),
        "energia_j": round(energia_j, 1),
        "temperatura_pico_c": max(a.tj_c for a in amostras),
    }
    if clipes is not None:
        resumo["clipes"] = clipes
        resumo["energia_por_clipe_j"] = round(energia_j / clipes, 1)
        resumo["segundos_por_clipe"] = round(duracao / clipes, 1)
    return resumo
