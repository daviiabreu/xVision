"""Tipos congelados do evento de finalizacao e a serializacao explicita deles.

`from_dict` assume payload ja validado contra o JSON Schema. A validacao mora em `validation`
porque e ela que carrega o `jsonschema`: importar os tipos nao paga pelo validador.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from types import MappingProxyType
from typing import Any, Literal, cast

from contracts.schema import SCHEMA_FILENAME, SCHEMA_VERSION, load_schema


class ContractError(ValueError): ...


def _regras_do_schema() -> tuple[re.Pattern[str], frozenset[str]]:
    schema = load_schema()
    try:
        padrao = cast("str", schema["properties"]["shot_id"]["pattern"])
        estados = cast("list[str]", schema["$defs"]["featureValue"]["properties"]["status"]["enum"])
        return re.compile(padrao), frozenset(estados)
    except (KeyError, TypeError, re.error) as erro:
        raise ContractError(
            f"{SCHEMA_FILENAME} nao traz as regras que o tipo cobra: {erro}"
        ) from erro


_SHOT_ID_PATTERN, _FEATURE_STATUSES = _regras_do_schema()


def _e_numero(valor: object) -> bool:
    return isinstance(valor, int | float) and not isinstance(valor, bool)


def _e_escalar_json(valor: object) -> bool:
    return valor is None or isinstance(valor, str | int | float | bool)


def _freeze(campo: str, value: object) -> object:
    if isinstance(value, Mapping):
        congelado: dict[str, object] = {}
        for chave, item in cast("Mapping[object, object]", value).items():
            if not isinstance(chave, str):
                raise ContractError(
                    f"{campo} tem chave {chave!r}, e o json so aceita chave de texto: "
                    "int volta string do outro lado e tupla nao volta"
                )
            congelado[chave] = _freeze(campo, item)
        return MappingProxyType(congelado)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return tuple(_freeze(campo, v) for v in cast("Sequence[object]", value))
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise ContractError(
        f"{campo} carrega {type(value).__name__}, que o json nao serializa. "
        "Converta antes: int(), float(), .tolist(), .isoformat()"
    )


def _thaw(value: object) -> Any:
    if isinstance(value, Mapping):
        return {k: _thaw(v) for k, v in cast("Mapping[str, object]", value).items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in cast("tuple[object, ...]", value)]
    return value


_DATE_TIME_PROFILE = re.compile(
    r"^([0-9]{4}-[0-9]{2}-[0-9]{2})T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9](?:\.[0-9]+)?"
    r"(?:Z|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])\Z"
)


def _is_date_time(value: object) -> bool:
    if not isinstance(value, str):
        return False
    match = _DATE_TIME_PROFILE.match(value)
    if match is None:
        return False
    try:
        date.fromisoformat(match.group(1))
    except ValueError:
        return False
    return True


FeatureStatus = Literal["ok", "low_confidence", "unavailable"]


@dataclass(frozen=True)
class FeatureValue:
    name: str
    version: str
    status: FeatureStatus
    value: float | int | str | bool | None
    confidence: float | None
    reason: str | None
    evidence: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.status not in _FEATURE_STATUSES:
            raise ContractError(
                f"feature {self.name!r} com status {self.status!r} fora do enum do schema"
            )
        if self.status != "ok" and not self.reason:
            raise ContractError(
                f"feature {self.name!r} com status {self.status!r} exige reason preenchido"
            )
        for field, content in (("name", self.name), ("version", self.version)):
            if not content:
                raise ContractError(f"feature com {field} vazio")
        if not _e_escalar_json(self.value):
            raise ContractError(
                f"feature {self.name!r} com value do tipo {type(self.value).__name__}, "
                "que o json nao serializa"
            )
        if self.confidence is not None:
            if not _e_numero(self.confidence):
                raise ContractError(
                    f"feature {self.name!r} com confidence {self.confidence!r}, "
                    "e o schema espera number"
                )
            if not 0.0 <= self.confidence <= 1.0:
                raise ContractError(
                    f"feature {self.name!r} com confidence {self.confidence!r} fora de 0..1"
                )
        object.__setattr__(self, "evidence", _freeze("evidence", self.evidence))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "status": self.status,
            "value": self.value,
            "confidence": self.confidence,
            "reason": self.reason,
            "evidence": _thaw(self.evidence),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureValue:
        return cls(
            name=data["name"],
            version=data["version"],
            status=data["status"],
            value=data["value"],
            confidence=data["confidence"],
            reason=data["reason"],
            evidence=data["evidence"],
        )


@dataclass(frozen=True)
class PipelineMetadata:
    versions: Mapping[str, str]
    timings_ms: Mapping[str, float]
    host: str
    started_at: str

    def __post_init__(self) -> None:
        if not self.host:
            raise ContractError("pipeline com host vazio")
        if not _is_date_time(self.started_at):
            raise ContractError(
                f"pipeline com started_at {self.started_at!r} fora do formato do contrato, "
                "AAAA-MM-DDThh:mm:ss[.fracao] com Z ou +hh:mm"
            )
        for modulo, versao in cast("Mapping[str, object]", self.versions).items():
            if not isinstance(versao, str):
                raise ContractError(f"versions[{modulo!r}] = {versao!r}, e o schema espera string")
        for etapa, ms in self.timings_ms.items():
            if not _e_numero(ms):
                raise ContractError(f"timings_ms[{etapa!r}] = {ms!r}, e o schema espera number")
        object.__setattr__(self, "versions", _freeze("versions", self.versions))
        object.__setattr__(self, "timings_ms", _freeze("timings_ms", self.timings_ms))

    def to_dict(self) -> dict[str, Any]:
        return {
            "versions": _thaw(self.versions),
            "timings_ms": _thaw(self.timings_ms),
            "host": self.host,
            "started_at": self.started_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineMetadata:
        return cls(
            versions=dict(data["versions"]),
            timings_ms=dict(data["timings_ms"]),
            host=data["host"],
            started_at=data["started_at"],
        )


@dataclass(frozen=True)
class ShotEvent:
    schema_version: str
    shot_id: str
    clip_id: str
    match_id: str | None
    shot_timestamp_s: float
    features: tuple[FeatureValue, ...]
    pipeline: PipelineMetadata

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ContractError(
                f"schema_version {self.schema_version!r} desconhecida, esperada {SCHEMA_VERSION!r}"
            )
        if not self.clip_id:
            raise ContractError("clip_id vazio")
        prefix = f"{self.clip_id}:"
        if not self.shot_id.startswith(prefix):
            raise ContractError(f"shot_id {self.shot_id!r} nao deriva do clip_id {self.clip_id!r}")
        if not _SHOT_ID_PATTERN.fullmatch(self.shot_id):
            raise ContractError(f"shot_id {self.shot_id!r} nao termina em indice de quadro inteiro")
        if not _e_numero(self.shot_timestamp_s):
            raise ContractError(
                f"shot_timestamp_s {self.shot_timestamp_s!r}, e o schema espera number"
            )
        if self.shot_timestamp_s < 0:
            raise ContractError(f"shot_timestamp_s {self.shot_timestamp_s!r} negativo")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "shot_id": self.shot_id,
            "clip_id": self.clip_id,
            "match_id": self.match_id,
            "shot_timestamp_s": self.shot_timestamp_s,
            "features": [feature.to_dict() for feature in self.features],
            "pipeline": self.pipeline.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ShotEvent:
        return cls(
            schema_version=data["schema_version"],
            shot_id=data["shot_id"],
            clip_id=data["clip_id"],
            match_id=data["match_id"],
            shot_timestamp_s=data["shot_timestamp_s"],
            features=tuple(FeatureValue.from_dict(item) for item in data["features"]),
            pipeline=PipelineMetadata.from_dict(data["pipeline"]),
        )


def make_shot_id(clip_id: str, shot_frame_index: int) -> str:
    return f"{clip_id}:{shot_frame_index}"
