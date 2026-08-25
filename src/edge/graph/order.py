"""Ordenacao topologica sobre `needs`, com o ciclo falhando antes de qualquer execucao."""

from __future__ import annotations

from collections.abc import Mapping

from edge.modules.base import ModuleError


class GraphError(ModuleError): ...


def topological_order(needs: Mapping[str, tuple[str, ...]]) -> tuple[str, ...]:
    _refuse_unconfigured(needs)
    pendentes = {kind: set(dependencias) for kind, dependencias in needs.items()}
    ordem: list[str] = []
    while pendentes:
        prontos = sorted(kind for kind, faltando in pendentes.items() if not faltando)
        if not prontos:
            raise GraphError(
                f"o grafo tem ciclo entre {', '.join(sorted(pendentes))}, "
                "e nenhum deles pode rodar primeiro"
            )
        for kind in prontos:
            del pendentes[kind]
            ordem.append(kind)
        for faltando in pendentes.values():
            faltando.difference_update(prontos)
    return tuple(ordem)


def _refuse_unconfigured(needs: Mapping[str, tuple[str, ...]]) -> None:
    for kind, dependencias in needs.items():
        ausentes = tuple(sorted(set(dependencias) - set(needs)))
        if ausentes:
            raise GraphError(
                f"{kind} declara needs {ausentes}, e nenhum modulo desse tipo foi configurado"
            )
