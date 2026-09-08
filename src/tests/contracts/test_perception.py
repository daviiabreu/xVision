"""Invariantes dos contratos internos de percepcao."""

from __future__ import annotations

import pytest

from contracts import BallDetection, Box, ContractError


def test_interpolated_ball_requires_zero_confidence_and_positive_gap() -> None:
    with pytest.raises(ContractError, match="confidence"):
        BallDetection(
            box=Box(0.0, 0.0, 10.0, 10.0),
            confidence=0.8,
            source="interpolated",
            trajectory_segment=0,
            gap_frames=1,
        )

    with pytest.raises(ContractError, match="gap_frames"):
        BallDetection(
            box=Box(0.0, 0.0, 10.0, 10.0),
            confidence=0.0,
            source="interpolated",
            trajectory_segment=0,
            gap_frames=0,
        )


def test_ball_velocity_requires_provenance_and_finite_values() -> None:
    with pytest.raises(ContractError, match="procedencia"):
        BallDetection(
            box=Box(0.0, 0.0, 10.0, 10.0),
            confidence=0.8,
            velocity_xy_px_s=(10.0, 20.0),
        )

    with pytest.raises(ContractError, match="finita"):
        BallDetection(
            box=Box(0.0, 0.0, 10.0, 10.0),
            confidence=0.8,
            velocity_xy_px_s=(float("inf"), 20.0),
            velocity_source="finite-difference",
        )


def test_ball_rejects_unknown_origin_at_runtime() -> None:
    with pytest.raises(ContractError, match="source"):
        BallDetection(box=Box(0.0, 0.0, 10.0, 10.0), confidence=0.8, source="guessed")
