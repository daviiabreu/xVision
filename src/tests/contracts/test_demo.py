"""A demonstracao precisa montar um evento valido, com os tres estados de feature."""

from __future__ import annotations

from contracts.demo import build_demo_event
from contracts.validation import validate_shot_event


def test_demo_event_covers_the_three_feature_states() -> None:
    demo = build_demo_event()

    validate_shot_event(demo.to_dict())
    assert {feature.status for feature in demo.features} == {"ok", "low_confidence", "unavailable"}
