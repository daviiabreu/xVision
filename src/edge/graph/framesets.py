"""Conjuntos de quadros nomeados em configuracao, e nao em codigo.

Duas estrategias hoje. `uniform` amostra o clipe inteiro a uma taxa declarada, e `around` abre
uma janela em volta da saida de outro modulo. A segunda e a unica barreira do grafo: ela so
resolve depois que o produtor da ancora terminou.

A janela e assimetrica de proposito. Contexto, assistencia e tipo de jogada precisam do que
veio antes do chute, entao `before_s` e maior que `after_s` na configuracao padrao.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from contracts import Clip
from edge.modules.base import ModuleError

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - a borda suporta 3.10, onde tomllib ainda nao existe
    import tomli as tomllib

NATIVE = "native"


class FrameSetError(ModuleError): ...


@dataclass(frozen=True)
class UniformFrames:
    fps: float | None


@dataclass(frozen=True)
class AroundFrames:
    anchor: str
    before_s: float
    after_s: float
    fps: float | None


FrameSet = UniformFrames | AroundFrames


def read_framesets(source: str | Path) -> dict[str, FrameSet]:
    text = source.read_text(encoding="utf-8") if isinstance(source, Path) else source
    documento = cast("dict[str, object]", tomllib.loads(text))
    entradas = cast("dict[str, dict[str, object]]", documento.get("frames", {}))
    return {nome: _read_one(nome, entrada) for nome, entrada in entradas.items()}


def resolve(
    frameset: FrameSet, clip: Clip, anchors: dict[str, int] | None = None
) -> tuple[int, ...]:
    if isinstance(frameset, UniformFrames):
        return uniform_indices(clip, frameset.fps)
    ancora = (anchors or {}).get(frameset.anchor)
    if ancora is None:
        raise FrameSetError(
            f"a janela tem ancora em {frameset.anchor!r}, que nao produziu instante nesta execucao"
        )
    return around_indices(clip, ancora, frameset.before_s, frameset.after_s, frameset.fps)


def uniform_indices(clip: Clip, fps: float | None) -> tuple[int, ...]:
    return _sample(range(clip.frame_count), clip, fps)


def around_indices(
    clip: Clip, anchor_frame: int, before_s: float, after_s: float, fps: float | None
) -> tuple[int, ...]:
    inicio = max(0, anchor_frame - round(before_s * clip.fps))
    fim = min(clip.frame_count - 1, anchor_frame + round(after_s * clip.fps))
    return _sample(range(inicio, fim + 1), clip, fps)


def _sample(disponiveis: range, clip: Clip, fps: float | None) -> tuple[int, ...]:
    passo = 1 if fps is None else max(1, round(clip.fps / fps))
    return tuple(disponiveis[::passo])


CHAVES = {
    "uniform": frozenset({"strategy", "fps"}),
    "around": frozenset({"strategy", "anchor", "before_s", "after_s", "fps"}),
}


def _recusar_chave_desconhecida(nome: str, entrada: dict[str, object], estrategia: str) -> None:
    """Chave que ninguem le nao pode passar calada.

    O caso que dói: `fpss = 5` em vez de `fps` cai em `fps = None`, que significa taxa nativa, e
    a etapa passa a decodificar o clipe inteiro em vez de amostrar. Sai `ok`, o evento valida, e
    o custo por clipe dobra sem uma linha de aviso, num projeto cujo numero de manchete e
    milissegundos por clipe. Uma letra a mais.
    """
    sobrando = set(entrada) - CHAVES[estrategia]
    if sobrando:
        raise FrameSetError(
            f"frames.{nome} declara {', '.join(sorted(sobrando))}, que nao existe em "
            f"strategy {estrategia!r}. As chaves aceitas sao "
            f"{', '.join(sorted(CHAVES[estrategia]))}"
        )


def _read_one(nome: str, entrada: dict[str, object]) -> FrameSet:
    estrategia = entrada.get("strategy")
    if estrategia == "uniform":
        _recusar_chave_desconhecida(nome, entrada, "uniform")
        return UniformFrames(fps=_rate(nome, entrada.get("fps")))
    if estrategia == "around":
        _recusar_chave_desconhecida(nome, entrada, "around")
        return AroundFrames(
            anchor=_required(nome, entrada, "anchor"),
            before_s=float(_number(nome, entrada, "before_s")),
            after_s=float(_number(nome, entrada, "after_s")),
            fps=_rate(nome, entrada.get("fps")),
        )
    raise FrameSetError(
        f"frames.{nome} declara strategy {estrategia!r}, e as conhecidas sao uniform e around"
    )


def _rate(nome: str, valor: object) -> float | None:
    if valor is None or valor == NATIVE:
        return None
    if isinstance(valor, int | float) and not isinstance(valor, bool):
        return float(valor)
    raise FrameSetError(f"frames.{nome} declara fps {valor!r}, e o esperado e numero ou 'native'")


def _required(nome: str, entrada: dict[str, object], chave: str) -> str:
    valor = entrada.get(chave)
    if not isinstance(valor, str):
        raise FrameSetError(f"frames.{nome} com strategy around precisa declarar {chave}")
    return valor


def _number(nome: str, entrada: dict[str, object], chave: str) -> float:
    valor = entrada.get(chave)
    if isinstance(valor, int | float) and not isinstance(valor, bool):
        return float(valor)
    raise FrameSetError(f"frames.{nome} precisa declarar {chave} como numero")
