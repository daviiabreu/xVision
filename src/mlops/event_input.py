"""Le as features de uma execucao sem escolher entre previsoes conflitantes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast


class EventInputError(ValueError): ...


def read_event_features(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    predictions: dict[str, dict[str, dict[str, Any]]] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        location = f"{path.name}:{number}"
        try:
            raw: object = json.loads(line)
        except json.JSONDecodeError as error:
            raise EventInputError(f"{location}: JSON invalido: {error}") from error
        if not isinstance(raw, dict):
            raise EventInputError(f"{location}: o evento precisa ser um objeto")
        event = cast("dict[str, Any]", raw)
        clip: object = event.get("clip_id")
        if not isinstance(clip, str) or not clip:
            raise EventInputError(f"{location}: clip_id precisa ser texto nao vazio")
        if clip in predictions:
            raise EventInputError(f"{location}: clip_id {clip!r} duplicado")
        predictions[clip] = _features(event.get("features"), location)
    return predictions


def _features(raw: object, location: str) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, list):
        raise EventInputError(f"{location}: features precisa ser uma lista")
    features: dict[str, dict[str, Any]] = {}
    for item in cast("list[object]", raw):
        if not isinstance(item, dict):
            raise EventInputError(f"{location}: cada feature precisa ser um objeto")
        feature = cast("dict[str, Any]", item)
        name: object = feature.get("name")
        if not isinstance(name, str) or not name:
            raise EventInputError(f"{location}: feature.name precisa ser texto nao vazio")
        if name in features:
            raise EventInputError(f"{location}: feature {name!r} duplicada")
        if "reason" not in feature or not isinstance(feature["reason"], str | type(None)):
            raise EventInputError(f"{location}: feature.reason precisa ser texto ou null")
        features[name] = feature
    return features
