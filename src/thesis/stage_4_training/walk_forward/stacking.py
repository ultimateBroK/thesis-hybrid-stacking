"""Classical time-safe stacking walk-forward trainer."""

from __future__ import annotations

from collections.abc import Callable
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from thesis.shared.config import Config
from thesis.stage_4_training.lgbm.utils import (
    _compute_class_weights,
    _save_feature_importance,
    _train_fixed,
    _wrap_np,
)
from thesis.stage_4_training.validation import WalkForwardWindow
from thesis.stage_4_training.walk_forward.artifacts import (
    _log_walk_forward_complete,
    _save_arch_copy,
    _save_oof_predictions,
    _save_training_history,
    _save_walk_forward_history,
)
from thesis.stage_4_training.walk_forward.diagnostics import (
    _add_prediction_diagnostics,
    _window_diagnostics,
)
from thesis.stage_4_training.walk_forward.feature_pipeline import (
    _select_static_feature_cols,
    fit_static_feature_pipeline,
)
from thesis.stage_4_training.walk_forward.lgbm import _prepare_static_wf_data
from thesis.stage_4_training.walk_forward.loop import run_walk_forward
from thesis.stage_4_training.walk_forward.predictions import (
    _CLASS_ORDER,
    _apply_confidence_threshold,
    _probability_columns,
)

logger = logging.getLogger("thesis.pipeline")

_BASE_MODEL_ALIASES = {
    "logistic_regression": "logreg",
    "random_forest": "rf",
    "lightgbm": "lgbm",
}
_MIN_SPLIT_ROWS = 4
_MIN_CALIBRATION_ROWS = 50

# ---------------------------------------------------------------------------
# Base model registry (Step 5)
# ---------------------------------------------------------------------------

_BASE_MODEL_REGISTRY: dict[str, Callable[..., Any]] = {}


def _register_base_model(name: str):
    def decorator(fn):
        _BASE_MODEL_REGISTRY[name] = fn
        return fn

    return decorator


@_register_base_model("logistic_regression")
def _build_logistic_regression(config: Config) -> Any:
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=config.workflow.random_seed,
    )


@_register_base_model("random_forest")
def _build_random_forest(config: Config) -> Any:
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=config.model.random_forest_n_estimators,
        max_depth=config.model.random_forest_max_depth,
        min_samples_leaf=config.model.random_forest_min_samples_leaf,
        class_weight="balanced_subsample",
        random_state=config.workflow.random_seed,
        n_jobs=config.workflow.n_jobs,
    )


def _build_base_model(name: str, config: Config) -> Any:
    if name not in _BASE_MODEL_REGISTRY:
        raise ValueError(f"Unsupported sklearn base model: {name!r}")
    return _BASE_MODEL_REGISTRY[name](config)


# ---------------------------------------------------------------------------
# Meta model registry (Step 5)
# ---------------------------------------------------------------------------

_META_MODEL_REGISTRY: dict[str, Callable[..., Any]] = {}


def _register_meta_model(name: str):
    def decorator(fn):
        _META_MODEL_REGISTRY[name] = fn
        return fn

    return decorator


def _fit_predictable_classifier(model: Any, X: np.ndarray, y: np.ndarray) -> Any:
    """Fit a classifier, falling back to DummyClassifier for one-class folds."""
    if len(np.unique(y)) < 2:
        from sklearn.dummy import DummyClassifier

        model = DummyClassifier(strategy="most_frequent")
    model.fit(X, y)
    return model


@_register_meta_model("logistic_regression")
def _build_meta_logreg(config: Config, X_meta: np.ndarray, y_meta: np.ndarray) -> Any:
    if len(np.unique(y_meta)) < 2:
        from sklearn.dummy import DummyClassifier

        return DummyClassifier(strategy="most_frequent").fit(X_meta, y_meta)
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=config.workflow.random_seed,
    ).fit(X_meta, y_meta)


@_register_meta_model("lightgbm")
def _build_meta_lgbm(config: Config, X_meta: np.ndarray, y_meta: np.ndarray) -> Any:
    if len(np.unique(y_meta)) < 2:
        from sklearn.dummy import DummyClassifier

        return DummyClassifier(strategy="most_frequent").fit(X_meta, y_meta)
    return _train_fixed(
        X_meta,
        y_meta,
        X_meta,
        y_meta,
        _compute_class_weights(y_meta),
        config,
        [f"meta_{i}" for i in range(X_meta.shape[1])],
    )


def _fit_meta_model(config: Config, X_meta: np.ndarray, y_meta: np.ndarray) -> Any:
    name = config.model.stacking_meta_model
    if name not in _META_MODEL_REGISTRY:
        raise ValueError(f"Unsupported stacking_meta_model: {name!r}")
    return _META_MODEL_REGISTRY[name](config, X_meta, y_meta)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _split_base_meta(
    train_df: pl.DataFrame, meta_fraction: float
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Chronologically split an outer train window into base and meta folds."""
    if not 0.0 < meta_fraction < 0.5:
        raise ValueError("stacking_meta_fraction must be between 0 and 0.5")
    if len(train_df) < _MIN_SPLIT_ROWS:
        raise ValueError("Training window too small for base/meta stacking split")
    meta_rows = max(1, int(round(len(train_df) * meta_fraction)))
    base_rows = len(train_df) - meta_rows
    if base_rows < 2 or meta_rows < 1:
        raise ValueError("Training window too small after base/meta split")
    return train_df.slice(0, base_rows), train_df.slice(base_rows, meta_rows)


def _split_base_cal_meta(
    train_df: pl.DataFrame,
    meta_fraction: float,
    cal_fraction: float,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Chronological 3-way split: base_train | calibration | meta."""
    total = len(train_df)
    if not 0.0 < meta_fraction < 0.5:
        raise ValueError("stacking_meta_fraction must be between 0 and 0.5")
    if not 0.0 < cal_fraction < 0.5:
        raise ValueError("stacking_calibration_fraction must be between 0 and 0.5")
    if meta_fraction + cal_fraction >= 0.9:
        raise ValueError("meta_fraction + calibration_fraction must be < 0.9")
    cal_rows = max(1, int(round(total * cal_fraction)))
    meta_rows = max(1, int(round(total * meta_fraction)))
    base_rows = total - cal_rows - meta_rows
    if base_rows < 2 or cal_rows < 1 or meta_rows < 1:
        raise ValueError("Training window too small after base/cal/meta split")
    return (
        train_df.slice(0, base_rows),
        train_df.slice(base_rows, cal_rows),
        train_df.slice(base_rows + cal_rows, meta_rows),
    )


def _split_base_cal(
    train_df: pl.DataFrame,
    cal_fraction: float,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Chronological split for OOF path: base_train | calibration."""
    total = len(train_df)
    if not 0.0 < cal_fraction < 0.5:
        raise ValueError("stacking_calibration_fraction must be between 0 and 0.5")
    cal_rows = max(1, int(round(total * cal_fraction)))
    base_rows = total - cal_rows
    if base_rows < 2 or cal_rows < 1:
        raise ValueError("Training window too small after base/cal split")
    return train_df.slice(0, base_rows), train_df.slice(base_rows, cal_rows)


def _compute_brier_scores(
    y_true: np.ndarray, proba: np.ndarray, label: str
) -> dict[str, float]:
    """Compute per-class Brier scores (one-vs-rest) for logging."""
    from sklearn.metrics import brier_score_loss

    scores = {}
    for i, cls in enumerate(_CLASS_ORDER):
        y_bin = (y_true == cls).astype(int)
        scores[f"{label}_class_{cls}"] = float(brier_score_loss(y_bin, proba[:, i]))
    scores[f"{label}_mean"] = float(np.mean(list(scores.values())))
    return scores


def _calibrate_base_models(
    base_models: dict[str, Any],
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    selected_cols: list[str] | None,
    n_base_models: int,
) -> dict[str, Any]:
    """Wrap fitted base models with Platt (sigmoid) scaling.

    Returns calibrated models dict. Falls back to raw model if calibration
    data is insufficient (< _MIN_CALIBRATION_ROWS per base model).
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator

    if len(y_cal) < _MIN_CALIBRATION_ROWS * n_base_models:
        logger.warning(
            ("Calibration set too small (%d rows, need ≥%d for %d models) — skipping"),
            len(y_cal),
            _MIN_CALIBRATION_ROWS * n_base_models,
            n_base_models,
        )
        return dict(base_models)

    calibrated = {}
    for name, model in base_models.items():
        frozen = FrozenEstimator(model)
        cal = CalibratedClassifierCV(estimator=frozen, method="sigmoid")
        cal.fit(X_cal, y_cal)
        calibrated[name] = cal
        logger.info("  %s: calibrated (Platt scaling on %d rows)", name, len(y_cal))
    return calibrated


def _generate_internal_folds(
    train_len: int,
    n_folds: int,
    purge_bars: int,
) -> list[WalkForwardWindow]:
    """Create expanding-origin sequential folds inside one outer train window.

    Fold boundaries are evenly spaced across *train_len* rows.  For fold *i*
    the model trains on rows 0..boundary[i] (expanding) and predicts rows
    boundary[i]..boundary[i+1].  A purge gap of *purge_bars* is applied
    between train-tail and predict-head to prevent label lookahead leakage.
    """
    fold_size = train_len // n_folds
    boundaries = [i * fold_size for i in range(n_folds)] + [train_len]
    folds: list[WalkForwardWindow] = []
    for i in range(1, n_folds):
        pred_start = boundaries[i]
        pred_end = boundaries[i + 1]
        train_start = 0
        raw_train_end = boundaries[i]
        window = WalkForwardWindow(
            train_start_idx=train_start,
            train_end_idx=max(train_start, raw_train_end - purge_bars),
            test_start_idx=min(pred_start + purge_bars, pred_end),
            test_end_idx=pred_end,
        )
        if (
            window.train_end_idx > window.train_start_idx
            and window.test_end_idx > window.test_start_idx
        ):
            folds.append(window)
    return folds


def _expanding_origin_oof(
    config: Config,
    train_df: pl.DataFrame,
    feature_cols: list[str],
    purge_bars: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, Any, list[str], list[str]]:
    """Generate OOF meta features via expanding-origin forward chain.

    Returns:
        (meta_train_outputs, y_meta, feature_pipeline, selected_cols, static_cols)
    """
    n_folds = config.model.stacking_internal_folds
    if n_folds < 2:
        raise ValueError("Need at least 2 folds for OOF stacking")

    internal_folds = _generate_internal_folds(len(train_df), n_folds, purge_bars)
    if not internal_folds:
        raise ValueError(
            f"Expanding-origin produced 0 usable folds from {len(train_df)} rows "
            f"with {n_folds} folds and {purge_bars} purge bars"
        )
    logger.info(
        "Expanding-origin OOF: %d internal folds, purge=%d, train_rows=%d",
        len(internal_folds),
        purge_bars,
        len(train_df),
    )

    # Fit feature pipeline on full train data so pipeline is consistent
    y_full = train_df["label"].to_numpy().astype(np.int32)
    static_cols = _select_static_feature_cols(config, train_df, feature_cols)
    feature_pipeline, selected_cols = fit_static_feature_pipeline(
        config, train_df, static_cols, y_full
    )
    X_full = feature_pipeline.transform(train_df.select(static_cols).to_pandas())

    oof_parts: dict[str, list[np.ndarray]] = {}
    y_oof_parts: list[np.ndarray] = []

    for fi, fold in enumerate(internal_folds):
        X_train = X_full[fold.train_start_idx : fold.train_end_idx]
        y_train = y_full[fold.train_start_idx : fold.train_end_idx]
        X_pred = X_full[fold.test_start_idx : fold.test_end_idx]
        y_pred = y_full[fold.test_start_idx : fold.test_end_idx]

        if len(np.unique(y_train)) < 2:
            logger.warning("Internal fold %d has single class — skipping", fi)
            continue

        for configured_name in config.model.stacking_base_models:
            short_name = _BASE_MODEL_ALIASES.get(configured_name, configured_name)
            if configured_name == "lightgbm":
                class_weights = _compute_class_weights(y_train)
                model = _train_fixed(
                    X_train,
                    y_train,
                    X_pred,
                    y_pred,
                    class_weights,
                    config,
                    selected_cols,
                )
            else:
                model = _fit_predictable_classifier(
                    _build_base_model(configured_name, config), X_train, y_train
                )
            proba = _aligned_predict_proba(model, X_pred, selected_cols)
            oof_parts.setdefault(short_name, []).append(proba)

        y_oof_parts.append(y_pred)

    if not oof_parts:
        raise ValueError("Expanding-origin OOF produced no predictions")

    meta_train_outputs = {
        name: np.concatenate(parts, axis=0) for name, parts in oof_parts.items()
    }
    y_meta = np.concatenate(y_oof_parts, axis=0).astype(np.int32)

    logger.info(
        "Expanding-origin OOF done: %d meta rows from %d folds",
        len(y_meta),
        len(internal_folds),
    )
    return meta_train_outputs, y_meta, feature_pipeline, selected_cols, static_cols


def _aligned_predict_proba(
    model: Any,
    X: np.ndarray,
    feature_names: list[str] | None = None,
) -> np.ndarray:
    """Predict probabilities aligned to the canonical class order [-1, 0, 1]."""
    from thesis.stage_4_training.walk_forward.predictions import (
        _align_probability_matrix,
    )

    X_pred: Any = X
    fitted_names = getattr(model, "feature_names_in_", None)
    if fitted_names is not None:
        names = [str(name) for name in fitted_names]
        if feature_names is not None and len(feature_names) == X.shape[1]:
            names = feature_names
        X_pred = _wrap_np(X, names)
    proba = model.predict_proba(X_pred)
    return _align_probability_matrix(proba, model.classes_)


def _stack_probability_features(
    base_outputs: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    """Concatenate base probability matrices into deterministic meta features."""
    if not base_outputs:
        raise ValueError("No base model outputs available for stacking")
    row_counts = {name: matrix.shape[0] for name, matrix in base_outputs.items()}
    if len(set(row_counts.values())) != 1:
        raise ValueError(f"Base probability row counts differ: {row_counts}")

    matrices: list[np.ndarray] = []
    names: list[str] = []
    suffixes = ["short", "hold", "long"]
    for name in sorted(base_outputs):
        matrix = np.asarray(base_outputs[name], dtype=np.float64)
        if matrix.shape[1] != len(_CLASS_ORDER):
            raise ValueError(f"{name} probability matrix must have 3 columns")
        matrices.append(matrix)
        names.extend(f"{name}_proba_{suffix}" for suffix in suffixes)
    return np.hstack(matrices), names


def _classification_summary(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    """Return accuracy, macro-F1 and per-class metrics for artifact JSON."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_recall_fscore_support,
    )

    if not y_true:
        return {"accuracy": None, "macro_f1": None, "per_class": {}}
    labels = [-1, 0, 1]
    p, r, f1, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "per_class": {
            str(label): {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f1[i]),
                "support": int(s[i]),
            }
            for i, label in enumerate(labels)
        },
    }


def _train_and_predict_stacking_window(
    config: Config,
    w_idx: int,
    window: Any,
    df: pl.DataFrame,
    feature_cols: list[str],
    *,
    is_regression_static: bool = False,
) -> dict[str, Any] | None:
    """Train base/meta learners for one outer walk-forward window."""
    train_df = df.slice(
        window.train_start_idx, window.train_end_idx - window.train_start_idx
    )
    test_df = df.slice(
        window.test_start_idx, window.test_end_idx - window.test_start_idx
    )
    if train_df.is_empty() or test_df.is_empty():
        return None

    y_test = test_df["label"].to_numpy().astype(np.int32)
    n_folds = config.model.stacking_internal_folds
    do_calibrate = (
        config.model.stacking_calibrate_base
        and config.model.stacking_calibration_fraction > 0.0
    )
    cal_frac = config.model.stacking_calibration_fraction

    if n_folds == 1:
        raise ValueError("Need at least 2 folds for OOF stacking")

    # ---- Choose meta-data generation strategy ----
    brier_log: dict[str, dict[str, float]] = {}
    calibration_info: dict[str, Any] = {}

    if n_folds >= 2:
        # Expanding-origin OOF path
        purge_bars = config.model.stacking_internal_purge
        if purge_bars <= 0:
            purge_bars = config.labels.horizon_bars

        if do_calibrate:
            # Carve out calibration set from tail of train_df
            base_cal_df, cal_df = _split_base_cal(train_df, cal_frac)
            y_cal = cal_df["label"].to_numpy().astype(np.int32)
            logger.info(
                (
                    "Window %d: OOF stacking (%d folds, purge=%d)"
                    " + calibration (%d rows, %.1f%%)"
                ),
                w_idx + 1,
                n_folds,
                purge_bars,
                len(y_cal),
                cal_frac * 100,
            )
            logger.info(
                "  Split: n_base_train=%d, n_calibration=%d",
                len(base_cal_df),
                len(cal_df),
            )
            meta_train_outputs, y_meta, feature_pipeline, selected_cols, static_cols = (
                _expanding_origin_oof(config, base_cal_df, feature_cols, purge_bars)
            )
            # Refit base models on base portion (not full train) for calibration
            y_base = base_cal_df["label"].to_numpy().astype(np.int32)
            X_base = feature_pipeline.transform(
                base_cal_df.select(static_cols).to_pandas()
            )
            X_cal = feature_pipeline.transform(cal_df.select(static_cols).to_pandas())
        else:
            logger.info(
                "Window %d: expanding-origin OOF stacking (%d folds, purge=%d)",
                w_idx + 1,
                n_folds,
                purge_bars,
            )
            meta_train_outputs, y_meta, feature_pipeline, selected_cols, static_cols = (
                _expanding_origin_oof(config, train_df, feature_cols, purge_bars)
            )
            # Refit all base models on FULL train for test predictions
            y_base = train_df["label"].to_numpy().astype(np.int32)
            X_base = feature_pipeline.transform(
                train_df.select(static_cols).to_pandas()
            )
            X_cal = None
            y_cal = None

        X_test = feature_pipeline.transform(test_df.select(static_cols).to_pandas())

        base_models: dict[str, Any] = {}
        test_outputs: dict[str, np.ndarray] = {}
        base_test_preds: dict[str, np.ndarray] = {}

        for configured_name in config.model.stacking_base_models:
            short_name = _BASE_MODEL_ALIASES.get(configured_name, configured_name)
            if configured_name == "lightgbm":
                class_weights = (
                    _compute_class_weights(y_base)
                    if len(np.unique(y_base)) > 1
                    else None
                )
                model = _train_fixed(
                    X_base,
                    y_base,
                    X_test,
                    y_test,
                    class_weights,
                    config,
                    selected_cols,
                )
            else:
                model = _fit_predictable_classifier(
                    _build_base_model(configured_name, config), X_base, y_base
                )
            base_models[short_name] = model

        # ---- Calibration ----
        if do_calibrate and X_cal is not None and y_cal is not None:
            for name, model in base_models.items():
                raw_proba_cal = _aligned_predict_proba(model, X_cal, selected_cols)
                brier_before = _compute_brier_scores(
                    y_cal, raw_proba_cal, f"{name}_before"
                )
                brier_log.update(brier_before)

            base_models = _calibrate_base_models(
                base_models,
                X_cal,
                y_cal,
                selected_cols,
                len(config.model.stacking_base_models),
            )
            calibration_info["calibrated"] = True
            calibration_info["n_calibration"] = len(y_cal)

            for name, model in base_models.items():
                cal_proba_cal = _aligned_predict_proba(model, X_cal, selected_cols)
                brier_after = _compute_brier_scores(
                    y_cal, cal_proba_cal, f"{name}_after"
                )
                brier_log.update(brier_after)

        # Generate test predictions from (possibly calibrated) models
        for name, model in base_models.items():
            test_outputs[name] = _aligned_predict_proba(model, X_test, selected_cols)
            base_test_preds[name] = _CLASS_ORDER[np.argmax(test_outputs[name], axis=1)]

        diag_extra = {
            "stacking_mode": "expanding_origin_oof",
            "internal_folds": n_folds,
            "internal_purge": purge_bars,
            "meta_train_rows": len(y_meta),
        }
    else:
        # Legacy 80/20 single-holdout path (stacking_internal_folds == 0)
        if do_calibrate:
            base_df, cal_df, meta_df = _split_base_cal_meta(
                train_df,
                config.model.stacking_meta_fraction,
                cal_frac,
            )
            y_base = base_df["label"].to_numpy().astype(np.int32)
            y_cal = cal_df["label"].to_numpy().astype(np.int32)
            y_meta = meta_df["label"].to_numpy().astype(np.int32)
            logger.info(
                "Window %d: single-holdout + calibration "
                "(n_base=%d, n_cal=%d, n_meta=%d)",
                w_idx + 1,
                len(base_df),
                len(cal_df),
                len(meta_df),
            )
        else:
            base_df, meta_df = _split_base_meta(
                train_df, config.model.stacking_meta_fraction
            )
            y_base = base_df["label"].to_numpy().astype(np.int32)
            y_meta = meta_df["label"].to_numpy().astype(np.int32)
            cal_df = None
            y_cal = None

        static_cols = _select_static_feature_cols(config, train_df, feature_cols)
        feature_pipeline, selected_cols = fit_static_feature_pipeline(
            config, base_df, static_cols, y_base
        )
        X_base = feature_pipeline.transform(base_df.select(static_cols).to_pandas())
        X_meta = feature_pipeline.transform(meta_df.select(static_cols).to_pandas())
        X_test = feature_pipeline.transform(test_df.select(static_cols).to_pandas())
        if do_calibrate and cal_df is not None:
            X_cal = feature_pipeline.transform(cal_df.select(static_cols).to_pandas())
        else:
            X_cal = None

        base_models = {}
        meta_train_outputs = {}
        test_outputs = {}
        base_test_preds = {}

        for configured_name in config.model.stacking_base_models:
            short_name = _BASE_MODEL_ALIASES.get(configured_name, configured_name)
            if configured_name == "lightgbm":
                class_weights = (
                    _compute_class_weights(y_base)
                    if len(np.unique(y_base)) > 1
                    else None
                )
                model = _train_fixed(
                    X_base,
                    y_base,
                    X_meta,
                    y_meta,
                    class_weights,
                    config,
                    selected_cols,
                )
            else:
                model = _fit_predictable_classifier(
                    _build_base_model(configured_name, config), X_base, y_base
                )
            base_models[short_name] = model

        # ---- Calibration ----
        if do_calibrate and X_cal is not None and y_cal is not None:
            for name, model in base_models.items():
                raw_proba_cal = _aligned_predict_proba(model, X_cal, selected_cols)
                brier_before = _compute_brier_scores(
                    y_cal, raw_proba_cal, f"{name}_before"
                )
                brier_log.update(brier_before)

            base_models = _calibrate_base_models(
                base_models,
                X_cal,
                y_cal,
                selected_cols,
                len(config.model.stacking_base_models),
            )
            calibration_info["calibrated"] = True
            calibration_info["n_calibration"] = len(y_cal)

            for name, model in base_models.items():
                cal_proba_cal = _aligned_predict_proba(model, X_cal, selected_cols)
                brier_after = _compute_brier_scores(
                    y_cal, cal_proba_cal, f"{name}_after"
                )
                brier_log.update(brier_after)

        # Generate meta and test predictions from (possibly calibrated) models
        for name, model in base_models.items():
            meta_train_outputs[name] = _aligned_predict_proba(
                model, X_meta, selected_cols
            )
            test_outputs[name] = _aligned_predict_proba(model, X_test, selected_cols)
            base_test_preds[name] = _CLASS_ORDER[np.argmax(test_outputs[name], axis=1)]

        diag_extra = {
            "stacking_mode": "single_holdout",
            "meta_train_rows": len(y_meta),
        }

    # ---- Soft-voting baseline (average base probabilities, argmax) ----
    soft_vote_proba = np.mean(list(test_outputs.values()), axis=0)
    soft_vote_preds = _CLASS_ORDER[np.argmax(soft_vote_proba, axis=1)]
    base_test_preds["soft_vote"] = soft_vote_preds

    # ---- Log Brier score comparison ----
    if brier_log:
        for configured_name in config.model.stacking_base_models:
            short_name = _BASE_MODEL_ALIASES.get(configured_name, configured_name)
            before_mean = brier_log.get(f"{short_name}_before_mean")
            after_mean = brier_log.get(f"{short_name}_after_mean")
            if before_mean is not None and after_mean is not None:
                delta = after_mean - before_mean
                improved = "✓" if delta < 0 else "✗"
                logger.info(
                    "  %s Brier: before=%.4f after=%.4f Δ=%.4f %s",
                    short_name,
                    before_mean,
                    after_mean,
                    delta,
                    improved,
                )

    # ---- Fit meta model and produce final predictions ----
    X_meta_stack, meta_feature_names = _stack_probability_features(meta_train_outputs)
    X_test_stack, _ = _stack_probability_features(test_outputs)
    meta_model = _fit_meta_model(config, X_meta_stack, y_meta)
    final_proba = _aligned_predict_proba(meta_model, X_test_stack, meta_feature_names)
    threshold = config.model.prediction_confidence_threshold
    final_preds = _apply_confidence_threshold(final_proba, threshold)

    # Check probability calibration: sum per row
    prob_sums = test_outputs[next(iter(test_outputs))].sum(axis=1)
    mean_prob_sum = float(np.mean(prob_sums))

    diag = _window_diagnostics(
        w_idx + 1, train_df, test_df, train_df["label"].to_numpy(), y_test
    )
    diag["base_train_rows"] = len(train_df)
    diag["meta_train_rows"] = diag_extra["meta_train_rows"]
    diag["base_models"] = list(base_models)
    diag["meta_model"] = config.model.stacking_meta_model
    diag["meta_feature_names"] = meta_feature_names
    diag["base_model_accuracy"] = {
        name: float((preds == y_test).mean()) for name, preds in base_test_preds.items()
    }
    diag["mean_base_prob_sum"] = mean_prob_sum
    diag.update(diag_extra)
    if calibration_info:
        diag["calibration"] = calibration_info
    if brier_log:
        diag["brier_scores"] = brier_log
    _add_prediction_diagnostics(
        diag,
        final_preds,
        y_test,
        final_proba,
        confidence_threshold=threshold,
    )

    oof_chunk = pl.DataFrame(
        {
            "timestamp": test_df["timestamp"],
            "true_label": y_test,
            "pred_label": final_preds.astype(np.int32),
            **_probability_columns(final_proba, _CLASS_ORDER),
        }
    )
    return {
        "oof_chunk": oof_chunk,
        "bundle": {
            "architecture": "stacking",
            "feature_pipeline": feature_pipeline,
            "feature_cols": selected_cols,
            "static_cols": static_cols,
            "base_models": base_models,
            "meta_model": meta_model,
            "meta_feature_names": meta_feature_names,
            "class_order": _CLASS_ORDER.tolist(),
        },
        "lgbm_model": base_models.get("lgbm"),
        "feature_cols": selected_cols,
        "accuracy": diag["accuracy"],
        "diag": diag,
        "base_preds": {name: preds.tolist() for name, preds in base_test_preds.items()},
        "final_preds": final_preds.tolist(),
        "y_true": y_test.tolist(),
    }


def _save_stacking_wf_results(
    config: Config,
    results: list[dict[str, Any]],
    windows: list[Any],
    stage_start: float,
) -> None:
    """Persist stacking walk-forward artifacts."""
    import joblib

    if not results:
        raise RuntimeError("No stacking OOF predictions generated")

    all_oof_preds = [r["oof_chunk"] for r in results]
    window_diagnostics = [r["diag"] for r in results]
    last_result = results[-1]
    last_bundle = last_result["bundle"]
    last_lgbm_model = last_result["lgbm_model"]
    last_feature_cols = last_result["feature_cols"]
    last_window_accuracy = last_result["accuracy"]
    last_window_index = len(results)

    # Model comparison accumulation
    comparison_inputs: dict[str, dict[str, list[int]]] = {
        "hybrid_stacking": {"true": [], "pred": []}
    }
    for r in results:
        y_true = [int(x) for x in r["y_true"]]
        comparison_inputs["hybrid_stacking"]["true"].extend(y_true)
        comparison_inputs["hybrid_stacking"]["pred"].extend(
            [int(x) for x in r["final_preds"]]
        )
        for name, preds in r["base_preds"].items():
            comparison_inputs.setdefault(name, {"true": [], "pred": []})
            comparison_inputs[name]["true"].extend(y_true)
            comparison_inputs[name]["pred"].extend([int(x) for x in preds])

    oof_df = _save_oof_predictions(
        config,
        all_oof_preds=all_oof_preds,
        window_diagnostics=window_diagnostics,
    )
    model_path = Path(config.paths.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(last_bundle, model_path)
    if last_lgbm_model is not None:
        _save_feature_importance(last_lgbm_model, last_feature_cols, config)
    _save_model_comparison(config, comparison_inputs)

    per_window_accuracies = {
        str(d.get("window")): d.get("accuracy") for d in window_diagnostics
    }
    n_folds = config.model.stacking_internal_folds
    stacking_mode = "expanding_origin_oof" if n_folds >= 2 else "single_holdout"
    purge_bars = (
        config.model.stacking_internal_purge
        if config.model.stacking_internal_purge > 0
        else config.labels.horizon_bars
    )

    validation_protocol: dict[str, Any] = {
        "outer_windows": "bar_based_walk_forward_with_purge_embargo",
        "stacking_mode": stacking_mode,
    }
    if stacking_mode == "expanding_origin_oof":
        validation_protocol["internal_folds"] = n_folds
        validation_protocol["internal_purge"] = purge_bars
    else:
        validation_protocol["base_meta_split"] = "chronological_train_head_meta_tail"
        validation_protocol["meta_fraction"] = config.model.stacking_meta_fraction

    _save_training_history(
        config,
        {
            "architecture": "stacking",
            "stacking": {
                "artifact_strategy": "last_walk_forward_window",
                "validation_protocol": validation_protocol,
                "base_models": config.model.stacking_base_models,
                "meta_model": config.model.stacking_meta_model,
                "last_window_accuracy": last_window_accuracy,
                "n_features": len(last_feature_cols),
            },
            "deployment_note": (
                f"Stacking bundle saved from window {last_window_index}/{len(windows)} "
                "(last chronological walk-forward window). It has not seen future data."
            ),
            "per_window_accuracies": per_window_accuracies,
        },
    )
    _save_walk_forward_history(
        config,
        windows=windows,
        window_diagnostics=window_diagnostics,
        oof_len=len(oof_df),
        architecture="stacking",
    )
    _save_arch_copy(oof_df, "stacking", config)
    _log_walk_forward_complete(
        arch_name="stacking",
        windows_count=len(windows),
        oof_len=len(oof_df),
        stage_start=stage_start,
        prefix="Stacking walk-forward complete",
    )


def _save_model_comparison(
    config: Config,
    comparison_inputs: dict[str, dict[str, list[int]]],
) -> None:
    """Persist aggregate base-vs-stacking classification metrics."""
    out = {
        name: _classification_summary(values["true"], values["pred"])
        for name, values in comparison_inputs.items()
    }
    if config.paths.session_dir:
        path = Path(config.paths.session_dir) / "reports" / "model_comparison.json"
    else:
        path = Path("results/model_comparison.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))


def train_stacking_walk_forward(config: Config) -> None:
    """Train leakage-safe classical stacking with outer walk-forward validation."""
    if config.model.objective != "multiclass":
        raise ValueError(
            "Stacking architecture currently supports objective='multiclass' only"
        )
    run_walk_forward(
        config,
        prepare_fn=_prepare_static_wf_data,
        window_fn=_train_and_predict_stacking_window,
        save_fn=_save_stacking_wf_results,
    )
