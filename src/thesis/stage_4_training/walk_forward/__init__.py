"""Walk-forward training sub-package.

Re-exports the public dispatcher and architecture-specific entry points
so external callers can import from ``thesis.stage_4_training.walk_forward``
or from the private-name aliases that tests and ``pipeline.py`` rely on.
"""

from thesis.stage_4_training.walk_forward.dispatcher import _run_walk_forward
from thesis.stage_4_training.walk_forward.hybrid import (
    _compute_regression_target,
    _run_walk_forward_hybrid,
)
from thesis.stage_4_training.walk_forward.static import (
    _run_static_train,
    _run_walk_forward_static,
)

__all__ = [
    "_compute_regression_target",
    "_run_static_train",
    "_run_walk_forward",
    "_run_walk_forward_hybrid",
    "_run_walk_forward_static",
]
