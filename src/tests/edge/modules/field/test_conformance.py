"""A especie de campo cumpre o seam comum de percepcao."""

from __future__ import annotations

import pytest

from edge.modules.conformance import ModuleConformance
from edge.modules.field.module import YoloField


class TestYoloFieldConformance(ModuleConformance):
    @pytest.fixture
    def module(self) -> YoloField:
        return YoloField()
