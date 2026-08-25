"""Ordenacao topologica sobre `needs`, e o ciclo que precisa falhar cedo."""

from __future__ import annotations

import pytest

from edge.graph.order import GraphError, topological_order


def test_topological_order_respects_needs() -> None:
    ordem = topological_order({"field": ("players",), "players": (), "ball": ()})

    assert ordem.index("players") < ordem.index("field")


def test_a_chain_of_three_comes_out_in_order() -> None:
    ordem = topological_order({"identity": ("tracking",), "tracking": ("players",), "players": ()})

    assert ordem == ("players", "tracking", "identity")


def test_the_order_is_stable_for_modules_that_do_not_depend_on_each_other() -> None:
    grafo = {"ball": (), "players": (), "field": ()}

    assert topological_order(grafo) == topological_order(grafo) == ("ball", "field", "players")


def test_cycle_raises_with_both_module_names() -> None:
    with pytest.raises(GraphError) as erro:
        topological_order({"players": ("field",), "field": ("players",)})

    assert "players" in str(erro.value)
    assert "field" in str(erro.value)


def test_a_module_that_needs_something_nobody_configured_fails_before_execution() -> None:
    with pytest.raises(GraphError, match="tracking"):
        topological_order({"identity": ("tracking",)})
