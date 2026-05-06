"""Model training package for walk-forward GRU and LightGBM workflows."""

from thesis.stage_4_training._baselines import (
    compute_baseline_metrics,
    majority_class_baseline,
    naive_direction,
    random_baseline,
    run_all_baselines,
)
from thesis.stage_4_training._gru import train_gru
from thesis.stage_4_training._gru_data import prepare_sequences
from thesis.stage_4_training._gru_inference import (
    extract_hidden_states,
    predict_gru_proba,
)
from thesis.stage_4_training._gru_persistence import (
    load_gru_classifier,
    load_gru_model,
    save_gru_model,
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
    "load_gru_classifier",
    "load_gru_model",
    "majority_class_baseline",
    "naive_direction",
    "predict_gru_proba",
    "prepare_sequences",
    "random_baseline",
    "run_all_baselines",
    "save_gru_model",
    "train_gru",
    "train_model",
]
