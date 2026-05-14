"""LightGBM walk-forward training. Tabular only, no sequences."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from thesis.shared.config import Config
from thesis.shared.constants import EXCLUDE_COLS
from thesis.stage_4_training.lgbm.utils import (
    _compute_class_weights,
    _train_lgbm,
    _wrap_np,
)
from thesis.stage_4_training.validation import WalkForwardWindow, generate_windows
from thesis.stage_4_training.walk_forward.artifacts import (
    _save_arch_copy,
    _save_feature_importance,
    _save_oof_predictions,
    _save_training_history,
    _save_walk_forward_history,
)
from thesis.stage_4_training.walk_forward.diagnostics import (
    _add_prediction_diagnostics,
    _window_diagnostics,
)
from thesis.stage_4_training.walk_forward.feature_pipeline import (
    _add_label_prior_features,
    fit_static_feature_pipeline,
    select_static_cols,
)
from thesis.stage_4_training.walk_forward.predictions import (
    _align_proba,
    _apply_confidence_threshold,
    proba_columns,
)

logger = logging.getLogger("thesis")


# ── Data loading ────────────────────────────────────────────────────


def _load_labeled_data(config: Config) -> tuple[pl.DataFrame, bool]:
    """Load labels parquet. Pre-compute regression target if needed."""
    path = Path(config.paths.labels)
    if not path.exists():
        raise FileNotFoundError(f"Labels not found: {path}")

    df = pl.read_parquet(path)
    logger.info("Loaded labels: %d rows", len(df))

    # Regression: sign of forward return as target
    is_regression = config.model.objective == "regression"
    if is_regression:
        if "close" not in df.columns:
            raise ValueError("Regression objective requires 'close' column")
        h = config.labels.horizon_bars
        close = df["close"].to_numpy()
        n = len(close)
        reg = np.full(n, np.nan, dtype=np.float64)
        future = np.roll(close, -h)[: n - h]
        reg[: n - h] = (future - close[: n - h]) / close[: n - h]
        df = df.with_columns(pl.Series("regression_target", reg))
        df = df.filter(pl.col("regression_target").is_not_nan())
        logger.info("Regression target: horizon=%d", h)

    # Regime label-prior features
    if getattr(config.features, "enable_regime_features", False):
        df = _add_label_prior_features(df, config)

    return df, is_regression


# ── Prepare ─────────────────────────────────────────────────────────


def _prepare(
    config: Config,
) -> tuple[pl.DataFrame, list[WalkForwardWindow], list[str], dict[str, Any]]:
    """Load data and generate walk-forward windows."""
    df, is_regression = _load_labeled_data(config)

    event_end = df["event_end"].to_numpy() if "event_end" in df.columns else None

    windows = generate_windows(
        total_bars=len(df),
        train_window_bars=config.validation.train_window_bars,
        test_window_bars=config.validation.test_window_bars,
        step_bars=config.validation.step_bars,
        purge_bars=config.validation.purge_bars,
        embargo_bars=config.validation.embargo_bars,
        min_train_bars=config.validation.min_train_bars,
        event_end=event_end,
    )
    if not windows:
        raise RuntimeError("No valid walk-forward windows")

    feature_cols = sorted(c for c in df.columns if c not in EXCLUDE_COLS)
    return df, windows, feature_cols, {"is_regression": is_regression}


# ── Train one window ────────────────────────────────────────────────


def _train_lgbm_window(
    config: Config,
    w_idx: int,
    window: WalkForwardWindow,
    df: pl.DataFrame,
    feature_cols: list[str],
    *,
    is_regression: bool,
    expanded_features: bool = False,
) -> dict[str, Any] | None:
    """Train LightGBM and predict for one window."""
    train_df = df.slice(window.train_start_idx, window.train_len)
    test_df = df.slice(window.test_start_idx, window.test_len)

    if train_df.is_empty() or test_df.is_empty():
        logger.warning("Window %d: empty split, skipping", w_idx + 1)
        return None

    y_train_cls = train_df["label"].to_numpy().astype(np.int32)
    y_test_cls = test_df["label"].to_numpy().astype(np.int32)

    diag = _window_diagnostics(w_idx + 1, train_df, test_df, y_train_cls, y_test_cls)

    # Feature columns
    if expanded_features:
        static_cols = [
            c
            for c in feature_cols
            if c in train_df.columns and c != "regression_target"
        ]
    else:
        static_cols = select_static_cols(config, train_df, feature_cols)

    # Tail 20% of training for LightGBM's internal validation
    val_split = max(1, int(len(train_df) * 0.2))
    train_head_df = train_df.slice(0, len(train_df) - val_split)
    pipeline, selected = fit_static_feature_pipeline(
        config, train_head_df, static_cols, y_train_cls[:-val_split]
    )

    X_train = pipeline.transform(train_df.select(static_cols).to_pandas())
    X_test = pipeline.transform(test_df.select(static_cols).to_pandas())

    X_tr, y_tr = X_train[:-val_split], y_train_cls[:-val_split]
    X_val, y_val = X_train[-val_split:], y_train_cls[-val_split:]

    # Sample weights
    sw = (
        train_df["sample_weight"].to_numpy().astype(np.float64)
        if "sample_weight" in train_df.columns
        else None
    )
    w_tr = sw[:-val_split] if sw is not None else None

    # Class weights (regression has no class weights)
    class_weights = None if is_regression else _compute_class_weights(y_tr)
    if class_weights:
        diag["class_weights"] = {str(k): v for k, v in class_weights.items()}

    # Train
    model = _train_lgbm(
        X_tr, y_tr, X_val, y_val, class_weights, config, selected, sample_weight=w_tr
    )

    # Predict
    if is_regression:
        raw = model.predict(_wrap_np(X_test, selected))
        preds = np.sign(raw).astype(np.int32)
        proba = np.zeros((len(preds), 3), dtype=np.float64)
        proba[np.arange(len(preds)), preds + 1] = 1.0
    else:
        proba = model.predict_proba(_wrap_np(X_test, selected))
        proba = _align_proba(proba, model.classes_)
        threshold = config.model.prediction_confidence_threshold
        preds = _apply_confidence_threshold(proba, threshold)

    _add_prediction_diagnostics(
        diag,
        preds,
        y_test_cls,
        proba,
        confidence_threshold=config.model.prediction_confidence_threshold,
    )

    acc = float((preds == y_test_cls).mean())
    logger.info("  LGBM window %d: acc=%.4f, test=%d", w_idx + 1, acc, len(y_test_cls))

    oof_chunk = pl.DataFrame(
        {
            "timestamp": test_df["timestamp"],
            "true_label": y_test_cls,
            "pred_label": preds.astype(np.int32),
            **proba_columns(proba, np.array([-1, 0, 1])),
        }
    )

    return {
        "oof_chunk": oof_chunk,
        "model": model,
        "static_cols": selected,
        "accuracy": acc,
        "diag": diag,
    }


# ── Save ────────────────────────────────────────────────────────────


def _save_results(
    config: Config,
    results: list[dict[str, Any]],
    windows: list[WalkForwardWindow],
    _elapsed: float,
) -> None:
    """Validate OOF, persist artifacts."""
    import joblib

    if not results:
        raise RuntimeError("No LGBM OOF predictions generated")

    last = results[-1]

    all_oof = [r["oof_chunk"] for r in results]
    diags = [r["diag"] for r in results]

    oof_df = _save_oof_predictions(
        config, all_oof_preds=all_oof, window_diagnostics=diags
    )

    model_path = Path(config.paths.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(last["model"], model_path)
    _save_feature_importance(last["model"], last["static_cols"], config)

    if config.paths.session_dir:
        per_window_acc = {str(d.get("window")): d.get("accuracy") for d in diags}

        _save_training_history(
            config,
            {
                "architecture": "lgbm",
                "lightgbm": {
                    "artifact_strategy": "last_walk_forward_window",
                    "validation_protocol": {
                        "outer_windows": "bar_based_walk_forward_with_purge_embargo",
                        "lgbm_validation": "tail_20_percent_of_outer_train",
                    },
                    "last_window_accuracy": last["accuracy"],
                    "best_iteration": int(last["model"].best_iteration_)
                    if hasattr(last["model"], "best_iteration_")
                    else None,
                    "n_features": len(last["static_cols"]),
                    "n_classes": len(last["model"].classes_)
                    if hasattr(last["model"], "classes_")
                    else None,
                },
                "deployment_note": (
                    f"Model from window {len(results)}/{len(windows)} "
                    "(last chronological window, no future data seen)"
                ),
                "per_window_accuracies": per_window_acc,
            },
        )

        _save_walk_forward_history(
            config,
            windows=windows,
            window_diagnostics=diags,
            oof_len=len(oof_df),
            architecture="lgbm",
        )

    _save_arch_copy(oof_df, "lgbm", config)
    logger.info(
        "LGBM walk-forward complete: %d windows, %d OOF rows",
        len(windows),
        len(oof_df),
    )


# ── Entry point ─────────────────────────────────────────────────────


def train_lgbm_walk_forward(config: Config, *, expanded_features: bool = False) -> None:
    """Train LightGBM with walk-forward validation."""
    from thesis.stage_4_training.walk_forward.loop import run_walk_forward

    run_walk_forward(
        config,
        prepare_fn=_prepare,
        window_fn=lambda c, w_i, w, df, fc, **kw: _train_lgbm_window(
            c, w_i, w, df, fc, expanded_features=expanded_features, **kw
        ),
        save_fn=_save_results,
    )
