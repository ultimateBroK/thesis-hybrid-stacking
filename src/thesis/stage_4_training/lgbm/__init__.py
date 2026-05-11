"""LightGBM training sub-package."""

from thesis.stage_4_training.walk_forward.lgbm import (
    train_lgbm_walk_forward as train_model,
)

__all__ = ["train_model"]
