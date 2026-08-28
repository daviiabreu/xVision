"""A especie `tracking`: o mesmo jogador entre dois quadros, sem peso e sem video."""

from edge.modules.tracking.greedy import (
    ALCANCE_EM_DIAGONAIS,
    MAX_GAP_FRAMES,
    MIN_AFINIDADE,
    PREPROCESS,
    GreedyTracking,
    afinidade,
    iou,
)

__all__ = [
    "ALCANCE_EM_DIAGONAIS",
    "MAX_GAP_FRAMES",
    "MIN_AFINIDADE",
    "PREPROCESS",
    "GreedyTracking",
    "afinidade",
    "iou",
]
