"""LightGBM training utilities for tabular walk-forward."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from thesis.shared.config import Config

logger = logging.getLogger("thesis")


def _wrap_np(X: np.ndarray, feature_cols: list[str]) -> Any:
    """Wrap NumPy matrix as pandas DataFrame. Preserves feature names for LightGBM."""
    import pandas as pd

    return pd.DataFrame(X, columns=feature_cols)


def _compute_class_weights(y: np.ndarray) -> dict[int, float]:
    """Balanced class weights for multiclass labels."""
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def _filter_unseen_classes(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    y_train: np.ndarray,
    feature_cols: list[str],
) -> tuple[Any, np.ndarray] | None:
    """Drop validation rows whose class is absent from training fold.

    LightGBM cannot evaluate on unseen class labels.
    Small folds can miss the rare Hold class.
    Returns None if no overlapping classes (skip early stopping).
    """
    seen = np.unique(y_train)
    mask = np.isin(y_val, seen)
    if not mask.any():
        logger.warning(
            "Validation has no overlapping classes with training "
            "— skipping early stopping",
        )
        return None
    if not mask.all():
        logger.warning(
            "Dropping %d unseen class rows from validation", int((~mask).sum())
        )
    return _wrap_np(X_val[mask], feature_cols), y_val[mask]


def _train_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weights: dict[int, float] | None,
    config: Config,
    feature_cols: list[str],
    sample_weight: np.ndarray | None = None,
) -> Any:
    """Train LightGBM with fixed hyperparameters. Optional early stopping."""
    import lightgbm as lgb

    m = config.model
    is_regression = m.objective == "regression"

    start = time.perf_counter()

    if is_regression:
        model = lgb.LGBMRegressor(
            num_leaves=m.num_leaves,
            max_depth=m.max_depth,
            learning_rate=m.learning_rate,
            n_estimators=m.n_estimators,
            min_child_samples=m.min_child_samples,
            subsample=m.subsample,
            subsample_freq=m.subsample_freq,
            colsample_bytree=m.feature_fraction,
            reg_alpha=m.reg_alpha,
            reg_lambda=m.reg_lambda,
            objective="regression",
            random_state=config.workflow.random_seed,
            n_jobs=config.workflow.n_jobs,
            verbose=-1,
        )
    else:
        model = lgb.LGBMClassifier(
            num_leaves=m.num_leaves,
            max_depth=m.max_depth,
            learning_rate=m.learning_rate,
            n_estimators=m.n_estimators,
            min_child_samples=m.min_child_samples,
            subsample=m.subsample,
            subsample_freq=m.subsample_freq,
            colsample_bytree=m.feature_fraction,
            reg_alpha=m.reg_alpha,
            reg_lambda=m.reg_lambda,
            interaction_constraints=[],
            class_weight=class_weights,
            objective="multiclass",
            num_class=3,
            random_state=config.workflow.random_seed,
            n_jobs=config.workflow.n_jobs,
            verbose=-1,
            use_missing=False,
            zero_as_missing=False,
        )

    # Filter validation to seen classes (regression skips this)
    filtered = (
        None
        if is_regression
        else _filter_unseen_classes(X_train, X_val, y_val, y_train, feature_cols)
    )

    def _progress(env: Any) -> None:
        """Log progress every 50 iterations."""
        if env.iteration % 50 == 0 or env.iteration == env.end_iteration - 1:
            loss = (
                env.evaluation_result_list[0][2] if env.evaluation_result_list else 0.0
            )
            logger.info("    LGBM iter=%d val_loss=%.5f", env.iteration, loss)

    if filtered is None:
        model.fit(_wrap_np(X_train, feature_cols), y_train, sample_weight=sample_weight)
    else:
        X_val_df, y_val_eval = filtered
        model.fit(
            _wrap_np(X_train, feature_cols),
            y_train,
            sample_weight=sample_weight,
            eval_set=[(X_val_df, y_val_eval)],
            callbacks=[
                lgb.early_stopping(m.early_stopping_rounds, verbose=False),
                _progress,
            ],
        )

    logger.info(
        "    LGBM done: best_iter=%d (%.1fs)",
        model.best_iteration_,
        time.perf_counter() - start,
    )
    return model
