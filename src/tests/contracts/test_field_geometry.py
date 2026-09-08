"""Validacao da procedencia do refinamento de campo."""

import pytest

from contracts import ContractError, FieldGeometry


@pytest.mark.parametrize("status,count", [("guessed", 5), ("accepted", 3), ("disabled", -1)])
def test_field_refinement_requires_known_status_and_enough_lines(status: str, count: int) -> None:
    with pytest.raises(ContractError, match="field"):
        FieldGeometry(
            homography=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            reprojection_error_px=0.0,
            reprojection_error_p95_px=0.0,
            keypoints=(),
            observed_keypoint_count=6,
            inlier_count=6,
            source="detected",
            age_frames=0,
            template_id="synthetic",
            reason=None,
            refinement=status,  # type: ignore[arg-type]
            line_support_count=count,
        )
