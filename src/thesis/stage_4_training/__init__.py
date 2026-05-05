"""Model training package for walk-forward GRU and LightGBM workflows."""

from thesis.stage_4_training._baselines import (
    compute_baseline_metrics,
    majority_class_baseline,
    naive_direction,
    random_baseline,
    run_all_baselines,
)
from thesis.stage_4_training._gru import (
    extract_hidden_states,
    prepare_sequences,
    save_gru_model,
    train_gru,
)
from thesis.stage_4_training._lgbm import train_model
from thesis.stage_4_training._validation import (
    WalkForwardWindow,
    generate_windows,
)
from thesis.stage_4_training._walk_forward import (
    _run_static_train,
    _run_walk_forward,
)

__all__ = [
    "train_model",
    "train_gru",
    "extract_hidden_states",
    "prepare_sequences",
    "save_gru_model",
    "WalkForwardWindow",
    "generate_windows",
    "_run_walk_forward",
    "_run_static_train",
    "run_all_baselines",
    "compute_baseline_metrics",
    "naive_direction",
    "majority_class_baseline",
    "random_baseline",
]
