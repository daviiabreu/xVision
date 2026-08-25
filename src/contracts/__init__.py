"""Contrato do evento do xVision: tipos congelados e o JSON Schema que os dois lados validam."""

from contracts.events import (
    ContractError,
    FeatureStatus,
    FeatureValue,
    PipelineMetadata,
    ShotEvent,
    make_shot_id,
)
from contracts.perception import (
    BallDetection,
    Box,
    Clip,
    FieldGeometry,
    FramePerception,
    Perception,
    PlayerDetection,
)
from contracts.schema import SCHEMA_PATH, SCHEMA_VERSION, load_schema
from contracts.validation import SchemaValidationError, validate_shot_event

__all__ = [
    "SCHEMA_PATH",
    "SCHEMA_VERSION",
    "BallDetection",
    "Box",
    "Clip",
    "ContractError",
    "FeatureStatus",
    "FeatureValue",
    "FieldGeometry",
    "FramePerception",
    "Perception",
    "PipelineMetadata",
    "PlayerDetection",
    "SchemaValidationError",
    "ShotEvent",
    "load_schema",
    "make_shot_id",
    "validate_shot_event",
]
