"""Model training package for tabular walk-forward workflows."""

from thesis.stage_4_training.baselines import (
    compute_baseline_metrics,
    majority_class_baseline,
    naive_direction,
    random_baseline,
    run_all_baselines,
)
from thesis.stage_4_training.lgbm import train_model
from thesis.stage_4_training.validation import (
    WalkForwardWindow,
    generate_windows,
)
from thesis.stage_4_training.walk_forward.stacking import train_stacking_walk_forward

__all__ = [
    "WalkForwardWindow",
    "compute_baseline_metrics",
    "generate_windows",
    "majority_class_baseline",
    "naive_direction",
    "random_baseline",
    "run_all_baselines",
    "train_model",
    "train_stacking_walk_forward",
]
