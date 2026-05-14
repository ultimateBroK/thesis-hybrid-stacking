"""Triple-barrier label generation package."""

from thesis.stage_3_labels._label_numba import (
    compute_average_uniqueness,
    compute_event_end,
)
from thesis.stage_3_labels.labeling import generate_labels

__all__ = [
    "generate_labels",
    "compute_event_end",
    "compute_average_uniqueness",
]
