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

__all__ = [
    "WalkForwardWindow",
    "compute_baseline_metrics",
    "extract_hidden_states",
    "generate_windows",
    "majority_class_baseline",
    "naive_direction",
    "prepare_sequences",
    "random_baseline",
    "run_all_baselines",
    "save_gru_model",
    "train_gru",
    "train_model",
]
