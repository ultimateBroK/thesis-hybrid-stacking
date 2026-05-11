"""LightGBM utilities for tabular walk-forward training."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np

from thesis.shared.config import Config
from thesis.shared.constants import (
    DIST_SHIFT_CLIP_MAX,
    DIST_SHIFT_CLIP_MIN,
)

logger = logging.getLogger("thesis.model")


# ── LightGBM utilities ──────────────────────────────────────────────


def _wrap_np(X: np.ndarray, feature_cols: list[str]) -> Any:
    """Wrap a NumPy matrix as a pandas DataFrame.

    Args:
        X: Feature matrix of shape ``(n_samples, n_features)``.
        feature_cols: Feature names aligned to matrix columns.

    Returns:
        A pandas DataFrame preserving feature names for LightGBM.
    """
    import pandas as pd

    return pd.DataFrame(X, columns=feature_cols)


def _build_interaction_constraints(feature_cols: list[str]) -> list[list[int]]:
    """Interaction constraints for LightGBM feature groups.

    Currently disabled — returning an empty list allows full interaction
    between tabular price-action features.
    """
    return []


def _compute_class_weights(y: np.ndarray) -> dict[int, float]:
    """Compute balanced class weights for multiclass labels.

    Args:
        y: Target labels.

    Returns:
        Mapping from class label to balanced class weight.
    """
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def _compute_distribution_shift_weights(
    y_train: np.ndarray,
    y_val: np.ndarray,
    clip_range: tuple[float, float] = (DIST_SHIFT_CLIP_MIN, DIST_SHIFT_CLIP_MAX),
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute per-sample training weights to reduce stale-regime bias.

    Compares class frequencies between the training head and its internal
    validation tail.  Classes that have become *more* common in the recent
    tail relative to the full training window are up-weighted so the model
    pays more attention to emerging regimes.  Classes that are fading
    receive lower weight, reducing the influence of stale patterns.

    Time-safe: only training-window labels are used — no future/test labels
    are ever consulted.

    Args:
        y_train: Training-head labels in ``{-1, 0, 1}``.
        y_val: Training-tail (internal validation) labels in ``{-1, 0, 1}``.
        clip_range: Min/max bounds for per-class weight ratios.

    Returns:
        ``(sample_weights, ratio_dict)`` — Per-sample weight array aligned
        to ``y_train`` (mean ≈ 1.0) and per-class shift-weight ratio dict
        with string keys ``{"-1", "0", "1"}`` for JSON serialization.
    """
    classes = np.array([-1, 0, 1])
    train_counts = np.array([np.sum(y_train == c) for c in classes], dtype=np.float64)
    val_counts = np.array([np.sum(y_val == c) for c in classes], dtype=np.float64)

    train_freq = train_counts / train_counts.sum()
    val_freq = val_counts / val_counts.sum()

    # Ratio = val_freq / train_freq:
    #   > 1.0 → class is MORE common in recent data → up-weight
    #   < 1.0 → class is LESS common in recent data → down-weight
    # Avoid division by zero — classes absent from train get clip min.
    train_freq_safe = np.where(train_freq > 0, train_freq, 1e-8)
    ratios = val_freq / train_freq_safe
    ratios = np.clip(ratios, clip_range[0], clip_range[1])

    # Map per-class weight to per-sample (training head)
    weight_map = {int(c): float(r) for c, r in zip(classes, ratios)}
    sample_weights = np.array([weight_map[int(y)] for y in y_train], dtype=np.float64)

    # Build ratio dict with string keys for JSON-friendly diagnostics
    ratio_dict: dict[str, float] = {
        str(int(c)): float(r) for c, r in zip(classes, ratios)
    }

    logger.info(
        "Distribution-shift weights: SHORT=%d→%.2f HOLD=%d→%.2f LONG=%d→%.2f "
        "(train freq: [%.1f%%, %.1f%%, %.1f%%] val freq: [%.1f%%, %.1f%%, %.1f%%]) "
        "min=%.3f median=%.3f max=%.3f mean=%.3f",
        int(train_counts[0]),
        ratios[0],
        int(train_counts[1]),
        ratios[1],
        int(train_counts[2]),
        ratios[2],
        train_freq[0] * 100,
        train_freq[1] * 100,
        train_freq[2] * 100,
        val_freq[0] * 100,
        val_freq[1] * 100,
        val_freq[2] * 100,
        float(np.min(sample_weights)),
        float(np.median(sample_weights)),
        float(np.max(sample_weights)),
        float(np.mean(sample_weights)),
    )

    return sample_weights, ratio_dict


def _filter_validation_to_seen_classes(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    y_train: np.ndarray,
    feature_cols: list[str],
) -> tuple[Any, np.ndarray] | None:
    """Drop validation rows whose class is absent from the training fold.

    LightGBM's sklearn wrapper label-encodes classes from ``y_train`` and
    cannot transform an ``eval_set`` containing unseen labels. Small
    walk-forward folds can miss the rare Hold class, so validation is
    filtered to classes actually learnable in that fold.

    Returns ``None`` when the validation set has **no** overlapping classes
    with training — early stopping should be skipped for that fold.

    Returns:
        ``(X_val_filtered, y_val_filtered)`` or ``None``.
    """
    seen = np.unique(y_train)
    mask = np.isin(y_val, seen)
    if not mask.all():
        logger.warning(
            "LightGBM validation contains %d row(s) from unseen train classes %s; "
            "dropping them from early-stopping eval_set",
            int((~mask).sum()),
            sorted(set(map(int, y_val[~mask]))),
        )
    if not mask.any():
        logger.warning(
            "Validation set has no overlapping classes with training "
            "— skipping early stopping"
        )
        return None
    return _wrap_np(X_val[mask], feature_cols), y_val[mask]


# ── LightGBM training — fixed hyperparameters ───────────────────────


def _train_fixed(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weights: dict[int, float] | None,
    config: Config,
    feature_cols: list[str],
    sample_weight: np.ndarray | None = None,
) -> Any:
    """Train LightGBM with fixed hyperparameters.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels (multiclass) or continuous targets (regression).
        X_val: Validation feature matrix.
        y_val: Validation labels or targets.
        class_weights: Balanced class weights (None for regression).
        config: Resolved application configuration.
        feature_cols: Ordered feature names.
        sample_weight: Optional per-row training weights.

    Returns:
        Fitted LightGBM model (classifier or regressor).
    """
    import lightgbm as lgb

    m = config.model
    is_regression = m.objective == "regression"
    constraints = _build_interaction_constraints(feature_cols)
    logger.info(
        "LightGBM: %s leaves=%d depth=%d lr=%.4f n_est=%d features=%d",
        "regressor" if is_regression else "classifier",
        m.num_leaves,
        m.max_depth,
        m.learning_rate,
        m.n_estimators,
        len(feature_cols),
    )

    start_time = time.perf_counter()

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
            interaction_constraints=constraints,
            class_weight=class_weights,
            objective="multiclass",
            num_class=3,
            random_state=config.workflow.random_seed,
            n_jobs=config.workflow.n_jobs,
            verbose=-1,
            use_missing=False,
            zero_as_missing=False,
        )

    if is_regression:
        filtered = _wrap_np(X_val, feature_cols), y_val
    else:
        filtered = _filter_validation_to_seen_classes(
            X_train, X_val, y_val, y_train, feature_cols
        )

    def _progress_cb(env: Any) -> None:
        """Emit sparse boosting progress logs."""
        if env.iteration % 50 == 0 or env.iteration == env.end_iteration - 1:
            v_loss = (
                env.evaluation_result_list[0][2] if env.evaluation_result_list else 0.0
            )
            logger.info("LightGBM iter=%d val_loss=%.5f", env.iteration, v_loss)

    if filtered is None:
        logger.warning(
            "Validation set has no overlapping classes with training "
            "— skipping early stopping"
        )
        model.fit(
            _wrap_np(X_train, feature_cols),
            y_train,
            sample_weight=sample_weight,
        )
    else:
        X_val_df, y_val_eval = filtered
        model.fit(
            _wrap_np(X_train, feature_cols),
            y_train,
            sample_weight=sample_weight,
            eval_set=[(X_val_df, y_val_eval)],
            callbacks=[
                lgb.early_stopping(m.early_stopping_rounds, verbose=False),
                _progress_cb,
            ],
        )

    train_time = time.perf_counter() - start_time
    logger.info(
        "LightGBM done: best_iter=%d (%.1fs)",
        model.best_iteration_,
        train_time,
    )
    return model


def _save_feature_importance(
    model: Any, feature_cols: list[str], config: Config
) -> None:
    """Save sorted model feature importances to JSON.

    Args:
        model: Fitted model exposing ``feature_importances_``.
        feature_cols: Ordered feature names.
        config: Resolved application configuration.
    """
    try:
        imp = model.feature_importances_
        pairs = sorted(zip(feature_cols, imp), key=lambda x: x[1], reverse=True)
        if config.paths.session_dir:
            out_path = (
                Path(config.paths.session_dir) / "reports" / "feature_importance.json"
            )
        else:
            out_path = Path("results/feature_importance.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({name: float(val) for name, val in pairs}, f, indent=2)
        logger.info(
            "Feature importance saved (top 5: %s)",
            [p[0] for p in pairs[:5]],
        )
    except (OSError, ValueError) as e:
        logger.warning("Feature importance save failed: %s", e)

