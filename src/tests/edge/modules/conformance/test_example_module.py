"""Prova de que a suite se herda em tres linhas, que e o ponto dela."""

from __future__ import annotations

import pytest

from edge.modules.conformance import ModuleConformance
from edge.testing.example_module import ExampleModule


class TestExampleModule(ModuleConformance):
    @pytest.fixture
    def module(self) -> ExampleModule:
        return ExampleModule()
