"""Walk-forward training sub-package.

Re-exports the public dispatcher and architecture-specific entry points
so external callers can import from ``thesis.stage_4_training.walk_forward``.
"""

from thesis.stage_4_training.walk_forward.dispatcher import train_walk_forward
from thesis.stage_4_training.walk_forward.lgbm import (
    train_lgbm_fixed,
    train_lgbm_walk_forward,
)

__all__ = [
    "train_lgbm_fixed",
    "train_lgbm_walk_forward",
    "train_walk_forward",
]
