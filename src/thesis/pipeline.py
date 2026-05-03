"""Pipeline orchestration — sequential stage runner with walk-forward validation.

Stages:
    0. Data preparation (tick → OHLCV)
    1. Feature engineering
    2. Triple-barrier labeling
    3. Walk-forward training (sliding window: GRU + LightGBM per window)
    4. Backtest (on concatenated OOF predictions)
    5. Report generation
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from thesis.config import Config
from thesis.constants import EXCLUDE_COLS
from thesis.ui import console, stage_header, stage_skip
from thesis.data import prepare_data
from thesis.features import generate_features
from thesis.labels import generate_labels
from thesis.validation import generate_windows, log_windows
from thesis.backtest import run_backtest
from thesis.report import generate_report

logger = logging.getLogger("thesis.pipeline")
_CLASS_ORDER = np.array([-1, 0, 1], dtype=np.int32)


def _select_static_feature_cols(
    config: Config,
    df: pl.DataFrame,
    candidate_cols: list[str],
) -> list[str]:
    """Return compact, interpretable static features for LightGBM.

    Args:
        config: Runtime configuration containing the static feature whitelist.
        df: DataFrame slice used for model training or inference.
        candidate_cols: Fallback feature columns discovered from the dataset.

    Returns:
        Ordered feature names present in ``df``. Uses the centralized whitelist
        first and falls back to discovered candidates for tests or partial data.
    """
    available = [c for c in config.features.static_feature_cols if c in df.columns]
    if available:
        return available
    # Fallback keeps tests and partial feature sets usable.
    return [c for c in candidate_cols if c in df.columns]


def _counts_dict(values: np.ndarray) -> dict[str, int]:
    """Return compact class/count diagnostics with string keys for JSON."""
    if len(values) == 0:
        return {}
    labels, counts = np.unique(values.astype(np.int32), return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(labels, counts)}


def _pct_dict(counts: dict[str, int]) -> dict[str, float]:
    """Convert count dict to rounded percentages."""
    total = sum(counts.values())
    if total == 0:
        return {}
    return {label: round(count / total * 100.0, 2) for label, count in counts.items()}


def _window_dates(df: pl.DataFrame) -> dict[str, str]:
    """Return start/end timestamps for a window slice."""
    if len(df) == 0 or "timestamp" not in df.columns:
        return {"start": "", "end": ""}
    return {"start": str(df["timestamp"][0]), "end": str(df["timestamp"][-1])}


def _window_diagnostics(
    window_idx: int,
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Build per-window label diagnostics for logs and JSON artifacts."""
    train_counts = _counts_dict(y_train)
    test_counts = _counts_dict(y_test)
    diag: dict[str, Any] = {
        "window": window_idx,
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "train_dates": _window_dates(train_df),
        "test_dates": _window_dates(test_df),
        "train_label_counts": train_counts,
        "train_label_pct": _pct_dict(train_counts),
        "test_label_counts": test_counts,
        "test_label_pct": _pct_dict(test_counts),
    }
    logger.info(
        "Window %d labels | train=%s test=%s",
        window_idx,
        diag["train_label_pct"],
        diag["test_label_pct"],
    )
    return diag


def _add_prediction_diagnostics(
    diag: dict[str, Any],
    preds: np.ndarray,
    y_test: np.ndarray,
    proba: np.ndarray,
) -> None:
    """Attach prediction distribution and confidence diagnostics in-place."""
    pred_counts = _counts_dict(preds)
    confidence = np.max(proba, axis=1) if len(proba) else np.array([], dtype=float)

    # Compute LONG/SHORT prediction ratio
    long_count = pred_counts.get("1", 0)
    short_count = pred_counts.get("-1", 0)
    ls_ratio = long_count / short_count if short_count > 0 else float("inf")

    diag.update(
        {
            "prediction_counts": pred_counts,
            "prediction_pct": _pct_dict(pred_counts),
            "accuracy": float((preds == y_test).mean()) if len(y_test) else None,
            "mean_confidence": float(confidence.mean()) if len(confidence) else None,
            "high_conf_70_pct": float((confidence >= 0.70).mean() * 100.0)
            if len(confidence)
            else None,
            "ls_ratio": round(ls_ratio, 4) if short_count > 0 else None,
        }
    )
    logger.info(
        "Window %d preds | pred=%s acc=%.4f mean_conf=%.3f L/S=%.3f",
        diag["window"],
        diag["prediction_pct"],
        diag["accuracy"] or 0.0,
        diag["mean_confidence"] or 0.0,
        ls_ratio if short_count > 0 else float("nan"),
    )
    if short_count > 0 and long_count / short_count < 0.5:
        logger.warning(
            "Window %d: SHORT bias — LONG/SHORT ratio = %.2f",
            diag["window"],
            long_count / short_count,
        )


def _log_gru_signal_quality(
    hidden_states: np.ndarray,
    labels: np.ndarray,
    config: Config,
) -> None:
    """Log GRU hidden-state signal-to-noise diagnostic using ANOVA F-statistic.

    For each walk-forward window's GRU hidden states, compute the ANOVA
    F-statistic between each hidden dimension and the label via
    ``sklearn.feature_selection.f_classif``.  Logs the top-5 and bottom-5
    dimensions with their F-scores.  If all dimensions have near-zero
    predictive power, logs a warning that the GRU is contributing noise.

    Args:
        hidden_states: (n_samples, n_features) GRU hidden-state matrix.
        labels: (n_samples,) multiclass integer labels.
        config: Runtime configuration (unused; kept for interface consistency).
    """
    try:
        from sklearn.feature_selection import f_classif  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("sklearn not available — skipping GRU signal quality check")
        return

    if hidden_states is None or len(hidden_states) == 0:
        logger.warning("GRU signal quality: empty hidden states, skipping")
        return

    if labels is None or len(labels) == 0:
        logger.warning("GRU signal quality: empty labels, skipping")
        return

    if len(hidden_states) != len(labels):
        logger.warning(
            "GRU signal quality: shape mismatch hidden=%s vs labels=%s, skipping",
            hidden_states.shape,
            labels.shape,
        )
        return

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        logger.warning(
            "GRU signal quality: only %d class(es) present, "
            "cannot compute F-statistic (need ≥2)",
            len(unique_labels),
        )
        return

    min_samples_per_class = 2
    for cls in unique_labels:
        if np.sum(labels == cls) < min_samples_per_class:
            logger.warning(
                "GRU signal quality: class %s has < %d samples, skipping",
                cls,
                min_samples_per_class,
            )
            return

    try:
        f_scores, _p_values = f_classif(hidden_states, labels)
    except Exception as exc:
        logger.warning("GRU signal quality: f_classif failed — %s", exc)
        return

    n_features = len(f_scores)
    sorted_indices = np.argsort(f_scores)[::-1]  # descending

    top_n = min(5, n_features)
    bottom_n = min(5, n_features)

    top_indices = sorted_indices[:top_n]
    bottom_indices = sorted_indices[-bottom_n:][::-1]  # ascending for bottom display

    mean_f = float(np.mean(f_scores))

    logger.info(
        "GRU hidden signal quality: mean F=%.4f | top-5: %s | bottom-5: %s",
        mean_f,
        ", ".join(f"dim{i}={f_scores[i]:.3f}" for i in top_indices),
        ", ".join(f"dim{i}={f_scores[i]:.3f}" for i in bottom_indices),
    )

    if mean_f < 0.5:
        logger.warning(
            "GRU hidden states show no detectable signal — GRU contributes noise "
            "(mean F=%.4f across %d dimensions)",
            mean_f,
            n_features,
        )


# ---------------------------------------------------------------------------
# Stage runner with cache checking
# ---------------------------------------------------------------------------


def _run_stage(
    stage_num: int,
    config: Config,
    flag_name: str,
    cache_path: str | Path | None,
    work_fn: callable,
) -> None:
    """Execute a pipeline stage with cache checking."""
    flag = getattr(config.workflow, flag_name, False)
    if not flag:
        stage_skip(stage_num, "disabled")
        return

    if cache_path is not None:
        cache_path = Path(cache_path)
        if not config.workflow.force_rerun and cache_path.exists():
            stage_skip(stage_num, f"cached ({cache_path.name})")
            return

    stage_header(stage_num)
    work_fn(config)


# ---------------------------------------------------------------------------
# Walk-forward training loop
# ---------------------------------------------------------------------------


def _run_walk_forward_hybrid(config: Config) -> None:
    """Execute walk-forward sliding window training across all windows.

    For each window:
        1. Slice labeled data into train/test
        2. Apply purge & embargo
        3. Train GRU feature extractor on train
        4. Extract hidden states for train and test
        5. Build hybrid feature matrix (GRU hidden + static features)
        6. Train LightGBM on hybrid features
        7. Generate predictions on test slice
        8. Collect OOF predictions

    After all windows: concatenate OOF predictions and save for backtest.
    """
    import joblib
    import torch

    from thesis.gru import (
        train_gru,
        extract_hidden_states,
        prepare_sequences,
        save_gru_model,
    )
    from thesis.model import (
        _compute_class_weights,
        _train_fixed,
        _wrap_np,
    )

    labels_path = Path(config.paths.labels)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels not found: {labels_path}")

    df = pl.read_parquet(labels_path)
    logger.info("Loaded labeled data: %d rows", len(df))

    # Pre-compute regression target when objective is "regression"
    is_regression = config.model.objective == "regression"
    if is_regression:
        if "close" not in df.columns:
            raise ValueError(
                "Regression objective requires 'close' column in labeled data. "
                "Ensure feature engineering includes OHLCV data."
            )
        horizon = config.labels.horizon_bars
        close = df["close"].to_numpy()
        close_future = np.roll(close, -horizon)
        close_future[-horizon:] = close[-horizon:]  # fill trailing with current
        reg_target = (close_future - close) / close
        df = df.with_columns(pl.Series("regression_target", reg_target))
        logger.info(
            "Regression target computed: horizon=%d bars, mean=%.6f, std=%.6f",
            horizon,
            float(np.mean(reg_target)),
            float(np.std(reg_target)),
        )

    event_end = df["event_end"].to_numpy() if "event_end" in df.columns else None
    if event_end is None:
        logger.warning(
            "Labels lack event_end column — falling back to fixed-bar purge. "
            "Regenerate labels to enable event-time purging."
        )

    # Generate walk-forward windows
    v = config.validation
    windows = generate_windows(
        total_bars=len(df),
        train_window_bars=v.train_window_bars,
        test_window_bars=v.test_window_bars,
        step_bars=v.step_bars,
        purge_bars=v.purge_bars,
        embargo_bars=v.embargo_bars,
        min_train_bars=v.min_train_bars,
        event_end=event_end,
    )

    if not windows:
        raise RuntimeError(
            "No valid walk-forward windows generated — check data size and window parameters"
        )

    # P0-1: Guard against sequence leakage — ensure gap >= sequence_length
    gap_bars = (
        v.embargo_bars if event_end is not None else v.purge_bars + v.embargo_bars
    )
    seq_len = config.gru.sequence_length
    if gap_bars < seq_len:
        raise ValueError(
            f"Leakage risk: purge/embargo gap ({gap_bars} bars) < GRU sequence_length "
            f"({seq_len} bars). Test sequences would overlap with training data. "
            f"Increase embargo_bars to at least {seq_len}."
        )

    log_windows(windows, df, "timestamp")
    logger.info("Walk-forward: %d bar-based windows", len(windows))

    # Identify feature columns (exclude non-features)
    feature_cols = sorted(c for c in df.columns if c not in EXCLUDE_COLS)

    all_oof_preds: list[pl.DataFrame] = []
    gru_model = None
    gru_mean = None
    gru_std = None
    last_lgbm_model = None
    last_feature_cols: list[str] = []
    last_window_accuracy: float | None = None
    last_gru_history: list[dict] = []
    window_diagnostics: list[dict[str, Any]] = []

    stage_start = time.perf_counter()

    for w_idx, window in enumerate(windows):
        window_start = time.perf_counter()
        logger.info(
            "=== Window %d/%d: train=[%d:%d] test=[%d:%d] ===",
            w_idx + 1,
            len(windows),
            window.train_start_idx,
            window.train_end_idx,
            window.test_start_idx,
            window.test_end_idx,
        )

        # Slice data
        train_df = df.slice(
            window.train_start_idx, window.train_end_idx - window.train_start_idx
        )
        test_df = df.slice(
            window.test_start_idx, window.test_end_idx - window.test_start_idx
        )

        if len(train_df) < config.gru.sequence_length:
            logger.warning(
                "Window %d: train too small (%d), skipping", w_idx + 1, len(train_df)
            )
            continue

        # --- Train GRU ---
        # Validation protocol: GRU gets the last 20% of the outer training
        # window for neural early stopping. This slice never includes the
        # outer test window.
        val_split = max(1, int(len(train_df) * 0.2))
        gru_train_df = train_df.head(len(train_df) - val_split)
        gru_val_df = train_df.tail(val_split)

        (
            gru_model,
            _classifier,
            _,  # train_hidden from train_gru – overwritten below after full-train extraction
            val_hidden,
            gru_history,
            gru_mean,
            gru_std,
            dynamic_gru_cols,
        ) = train_gru(config, gru_train_df, gru_val_df, window_index=w_idx)

        # Extract hidden states for full train and test
        seq_len = config.gru.sequence_length
        train_seq, _, _ = prepare_sequences(train_df, dynamic_gru_cols, seq_len)
        test_seq, _, _ = prepare_sequences(test_df, dynamic_gru_cols, seq_len)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_hidden = extract_hidden_states(
            gru_model,
            train_seq,
            config.gru.batch_size,
            device=device,
            mean=gru_mean,
            std=gru_std,
        )
        test_hidden = extract_hidden_states(
            gru_model,
            test_seq,
            config.gru.batch_size,
            device=device,
            mean=gru_mean,
            std=gru_std,
        )

        # Align DataFrames with sequence outputs
        train_aligned = train_df.slice(seq_len - 1, len(train_hidden))
        test_aligned = test_df.slice(seq_len - 1, len(test_hidden))

        if len(train_aligned) == 0 or len(test_aligned) == 0:
            logger.warning("Window %d: aligned data empty, skipping", w_idx + 1)
            continue

        # --- PCA dimensionality reduction on GRU hidden states ---
        pca_k = config.gru.pca_components
        if pca_k > 0:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=pca_k, random_state=config.workflow.random_seed)
            pca.fit(train_hidden)
            train_hidden = pca.transform(train_hidden)
            test_hidden = pca.transform(test_hidden)

            explained = float(pca.explained_variance_ratio_.sum())
            logger.info(
                "GRU hidden states: %d→%d PCs, explained variance=%.1f%%",
                config.gru.hidden_size,
                pca_k,
                explained * 100,
            )
            if explained < 0.50:
                logger.warning(
                    "GRU hidden state space appears mostly noise "
                    "(%.1f%% explained by %d PCs)",
                    explained * 100,
                    pca_k,
                )

        # --- Build hybrid feature matrix ---
        static_cols = _select_static_feature_cols(config, train_aligned, feature_cols)
        hidden_components = pca_k if pca_k > 0 else config.gru.hidden_size
        gru_feat_names = [
            f"gru_pc_{i}" if pca_k > 0 else f"gru_h{i}"
            for i in range(hidden_components)
        ]
        all_feature_cols = gru_feat_names + static_cols

        X_train = np.concatenate(
            [train_hidden, train_aligned.select(static_cols).to_numpy()], axis=1
        )
        X_test = np.concatenate(
            [test_hidden, test_aligned.select(static_cols).to_numpy()], axis=1
        )

        y_train = train_aligned["label"].to_numpy().astype(np.int32)
        y_test = test_aligned["label"].to_numpy().astype(np.int32)
        if is_regression:
            reg_y_train = (
                train_aligned["regression_target"].to_numpy().astype(np.float64)
            )

        # Diagnose whether GRU hidden states carry any signal to labels
        _log_gru_signal_quality(train_hidden, y_train, config)

        diag = _window_diagnostics(
            w_idx + 1,
            train_aligned,
            test_aligned,
            y_train,
            y_test,
        )
        train_weights = (
            train_aligned["sample_weight"].to_numpy().astype(np.float64)
            if "sample_weight" in train_aligned.columns
            else None
        )

        # Validation protocol: LightGBM uses the last 20% of sequence-aligned
        # hybrid training rows for early stopping/Optuna. This is a separate
        # validation split after GRU hidden-state extraction, still inside the
        # outer training window.
        val_split_idx = max(1, int(len(X_train) * 0.2))
        X_tr = X_train[:-val_split_idx]
        w_tr = train_weights[:-val_split_idx] if train_weights is not None else None
        X_val = X_train[-val_split_idx:]

        if is_regression:
            y_tr = reg_y_train[:-val_split_idx]
            y_val = reg_y_train[-val_split_idx:]
            class_weights = None
        else:
            y_tr = y_train[:-val_split_idx]
            y_val = y_train[-val_split_idx:]
            class_weights = _compute_class_weights(y_tr)

        model = _train_fixed(
            X_tr,
            y_tr,
            X_val,
            y_val,
            class_weights,
            config,
            all_feature_cols,
            sample_weight=w_tr,
        )

        # --- Predict on test ---
        if is_regression:
            raw_preds = model.predict(_wrap_np(X_test, all_feature_cols))
            # Threshold at 0 for direction: pred > 0 → Long(1), pred < 0 → Short(-1)
            preds = np.where(raw_preds > 0, 1, np.where(raw_preds < 0, -1, 0)).astype(
                np.int32
            )
            # Build a synthetic 3-column proba for diagnostics compatibility
            aligned_proba = np.zeros((len(raw_preds), 3), dtype=np.float64)
            for i in range(len(raw_preds)):
                if preds[i] == -1:
                    aligned_proba[i, 0] = 1.0
                elif preds[i] == 0:
                    aligned_proba[i, 1] = 1.0
                else:
                    aligned_proba[i, 2] = 1.0
        else:
            proba = model.predict_proba(_wrap_np(X_test, all_feature_cols))
            aligned_proba = _align_probability_matrix(proba, model.classes_)
            preds = _CLASS_ORDER[np.argmax(aligned_proba, axis=1)]
        _add_prediction_diagnostics(diag, preds, y_test, aligned_proba)
        window_diagnostics.append(diag)

        acc = (preds == y_test).mean()
        logger.info(
            "Window %d: accuracy=%.4f, test_samples=%d",
            w_idx + 1,
            acc,
            len(y_test),
        )

        # Save deployable artifacts from the latest walk-forward window. Do not
        # select a "final" model by test-fold accuracy; OOF predictions carry
        # evaluation, while the last model is a chronological deployment proxy.
        last_lgbm_model = model
        last_feature_cols = all_feature_cols
        last_window_accuracy = float(acc)
        last_gru_history = gru_history

        # Collect OOF predictions
        if is_regression:
            oof_chunk = pl.DataFrame(
                {
                    "timestamp": test_aligned["timestamp"],
                    "true_label": y_test,
                    "pred_label": preds.astype(np.int32),
                    "pred_raw": raw_preds.astype(np.float64),
                }
            )
        else:
            oof_chunk = pl.DataFrame(
                {
                    "timestamp": test_aligned["timestamp"],
                    "true_label": y_test,
                    "pred_label": preds.astype(np.int32),
                    **_probability_columns(proba, model.classes_),
                }
            )
        all_oof_preds.append(oof_chunk)

        window_time = time.perf_counter() - window_start
        logger.info("Window %d done (%.1fs)", w_idx + 1, window_time)

    # --- Validate OOF predictions before saving ---
    if not all_oof_preds or gru_model is None:
        raise RuntimeError(
            "No OOF predictions generated — all walk-forward windows were skipped"
        )

    # --- Save final GRU model (last window only) ---
    if config.paths.session_dir:
        gru_path = Path(config.paths.session_dir) / "models" / "gru_model.pt"
        save_gru_model(gru_model, config, gru_path, mean=gru_mean, std=gru_std)

    # --- Concatenate OOF predictions ---

    oof_df = pl.concat(all_oof_preds)

    # P1-2: Verify OOF predictions have unique timestamps
    ts_col = oof_df["timestamp"]
    if ts_col.n_unique() < len(ts_col):
        dup_count = len(ts_col) - ts_col.n_unique()
        raise ValueError(
            f"OOF predictions contain {dup_count} duplicate timestamps — "
            f"walk-forward test windows should be non-overlapping. "
            f"Check step_bars vs test_window_bars."
        )

    preds_path = Path(config.paths.predictions)
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    oof_df.write_parquet(preds_path)
    oof_df.write_csv(preds_path.with_suffix(".csv"))

    # Save latest chronological LightGBM model, not a best-by-test artifact.
    if last_lgbm_model is not None:
        model_path = Path(config.paths.model)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(last_lgbm_model, model_path)

    # Save feature importance from the same latest chronological model.
    if last_lgbm_model is not None and last_feature_cols:
        from thesis.model import _save_feature_importance

        _save_feature_importance(last_lgbm_model, last_feature_cols, config)

    # Save training history (GRU + LightGBM info)
    if config.paths.session_dir:
        models_dir = Path(config.paths.session_dir) / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        history_path = models_dir / "training_history.json"

        lgbm_info: dict[str, Any] = {}
        if last_lgbm_model is not None:
            lgbm_info = {
                "artifact_strategy": "last_walk_forward_window",
                "validation_protocol": {
                    "outer_windows": "bar_based_walk_forward_with_purge_embargo",
                    "gru_validation": "tail_20_percent_of_outer_train",
                    "lgbm_validation": "tail_20_percent_of_sequence_aligned_outer_train",
                },
                "last_window_accuracy": last_window_accuracy,
                "best_iteration": int(last_lgbm_model.best_iteration_)
                if hasattr(last_lgbm_model, "best_iteration_")
                else None,
                "n_features": len(last_feature_cols),
                "n_classes": len(last_lgbm_model.classes_)
                if hasattr(last_lgbm_model, "classes_")
                else None,
            }

        with open(history_path, "w") as f:
            json.dump({"gru": last_gru_history, "lightgbm": lgbm_info}, f, indent=2)
        logger.info("Training history saved to %s", history_path)

    # Save walk-forward history
    if config.paths.session_dir:
        wf_path = (
            Path(config.paths.session_dir) / "reports" / "walk_forward_history.json"
        )
        wf_path.parent.mkdir(parents=True, exist_ok=True)
        history = {
            "num_windows": len(windows),
            "total_oof_predictions": len(oof_df),
            "window_details": [
                {
                    "window": i + 1,
                    "train_start_idx": w.train_start_idx,
                    "train_end_idx": w.train_end_idx,
                    "test_start_idx": w.test_start_idx,
                    "test_end_idx": w.test_end_idx,
                    **next(
                        (
                            item
                            for item in window_diagnostics
                            if item["window"] == i + 1
                        ),
                        {},
                    ),
                }
                for i, w in enumerate(windows)
            ],
        }
        with open(wf_path, "w") as f:
            json.dump(history, f, indent=2)

    total_time = time.perf_counter() - stage_start
    logger.info(
        "Walk-forward complete: %d windows, %d OOF predictions (%.1fs)",
        len(windows),
        len(oof_df),
        total_time,
    )


def _run_walk_forward_static(
    config: Config, *, expanded_features: bool = False
) -> None:
    """Execute a static-feature-only walk-forward baseline.

    This isolates whether the GRU hidden states add value. It uses the same
    event-time purged windows, LightGBM training path, sample weights, OOF
    prediction output, and report/backtest stages as the hybrid architecture.

    Args:
        config: Runtime configuration.
        expanded_features: When True, use ALL available feature columns
            (except EXCLUDE_COLS, gru_* columns, and regression_target)
            instead of the curated whitelist. Enables a fair feature-space
            comparison against the hybrid architecture.
    """
    import joblib

    from thesis.model import _save_feature_importance, _wrap_np

    labels_path = Path(config.paths.labels)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels not found: {labels_path}")

    df = pl.read_parquet(labels_path)
    logger.info("Loaded labeled data for static baseline: %d rows", len(df))

    # Pre-compute regression target when objective is "regression"
    is_regression_static = config.model.objective == "regression"
    if is_regression_static:
        if "close" not in df.columns:
            raise ValueError(
                "Regression objective requires 'close' column in labeled data."
            )
        horizon = config.labels.horizon_bars
        close = df["close"].to_numpy()
        close_future = np.roll(close, -horizon)
        close_future[-horizon:] = close[-horizon:]
        reg_target = (close_future - close) / close
        df = df.with_columns(pl.Series("regression_target", reg_target))
        logger.info("Static regression target computed: horizon=%d bars", horizon)

    event_end = df["event_end"].to_numpy() if "event_end" in df.columns else None
    if event_end is None:
        logger.warning(
            "Labels lack event_end column — falling back to fixed-bar purge. "
            "Regenerate labels to enable event-time purging."
        )

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
        raise RuntimeError("No valid walk-forward windows generated")

    log_windows(windows, df, "timestamp")
    feature_cols = sorted(c for c in df.columns if c not in EXCLUDE_COLS)
    all_oof_preds: list[pl.DataFrame] = []
    last_lgbm_model = None
    last_feature_cols: list[str] = []
    last_window_accuracy: float | None = None
    window_diagnostics: list[dict[str, Any]] = []
    stage_start = time.perf_counter()

    for w_idx, window in enumerate(windows):
        window_start = time.perf_counter()
        logger.info(
            "=== Static window %d/%d: train=[%d:%d] test=[%d:%d] ===",
            w_idx + 1,
            len(windows),
            window.train_start_idx,
            window.train_end_idx,
            window.test_start_idx,
            window.test_end_idx,
        )

        train_df = df.slice(
            window.train_start_idx, window.train_end_idx - window.train_start_idx
        )
        test_df = df.slice(
            window.test_start_idx, window.test_end_idx - window.test_start_idx
        )
        if len(train_df) < 2 or len(test_df) == 0:
            logger.warning("Static window %d too small; skipping", w_idx + 1)
            continue

        if expanded_features:
            static_cols = [
                c
                for c in feature_cols
                if c in train_df.columns
                and not c.startswith("gru_")
                and c != "regression_target"
            ]
            mode_tag = "expanded"
        else:
            static_cols = _select_static_feature_cols(config, train_df, feature_cols)
            mode_tag = "whitelist"
        logger.info(
            "Static baseline using %d features (%s mode)",
            len(static_cols),
            mode_tag,
        )
        X_train = train_df.select(static_cols).to_numpy()
        X_test = test_df.select(static_cols).to_numpy()
        if is_regression_static:
            y_train = train_df["regression_target"].to_numpy().astype(np.float64)
            y_test = test_df["regression_target"].to_numpy().astype(np.float64)
            # For diagnostics, use original classification labels
            y_train_cls = train_df["label"].to_numpy().astype(np.int32)
            y_test_cls = test_df["label"].to_numpy().astype(np.int32)
        else:
            y_train = train_df["label"].to_numpy().astype(np.int32)
            y_test = test_df["label"].to_numpy().astype(np.int32)
            y_train_cls = y_train
            y_test_cls = y_test
        sample_weight = (
            train_df["sample_weight"].to_numpy().astype(np.float64)
            if "sample_weight" in train_df.columns
            else None
        )
        diag = _window_diagnostics(
            w_idx + 1, train_df, test_df, y_train_cls, y_test_cls
        )

        val_split_idx = max(1, int(len(X_train) * 0.2))
        X_tr = X_train[:-val_split_idx]
        y_tr = y_train[:-val_split_idx]
        X_val = X_train[-val_split_idx:]
        y_val = y_train[-val_split_idx:]
        w_tr = sample_weight[:-val_split_idx] if sample_weight is not None else None
        val_reference_price = float(
            train_df.slice(len(train_df) - val_split_idx, val_split_idx)[
                "close"
            ].median()
        )

        model = _fit_lgbm_model(
            X_tr,
            y_tr,
            X_val,
            y_val,
            val_reference_price,
            config,
            static_cols,
            trials_override=config.validation.wf_optuna_trials
            if config.validation.wf_optuna_trials > 0
            else None,
            sample_weight=w_tr,
        )

        if is_regression_static:
            raw_preds = model.predict(_wrap_np(X_test, static_cols))
            preds = np.where(raw_preds > 0, 1, np.where(raw_preds < 0, -1, 0)).astype(
                np.int32
            )
            aligned_proba = np.zeros((len(raw_preds), 3), dtype=np.float64)
            for i in range(len(raw_preds)):
                if preds[i] == -1:
                    aligned_proba[i, 0] = 1.0
                elif preds[i] == 0:
                    aligned_proba[i, 1] = 1.0
                else:
                    aligned_proba[i, 2] = 1.0
        else:
            proba = model.predict_proba(_wrap_np(X_test, static_cols))
            aligned_proba = _align_probability_matrix(proba, model.classes_)
            preds = _CLASS_ORDER[np.argmax(aligned_proba, axis=1)]
        _add_prediction_diagnostics(diag, preds, y_test_cls, aligned_proba)
        window_diagnostics.append(diag)

        acc = float((preds == y_test_cls).mean())
        logger.info(
            "Static window %d: accuracy=%.4f, test_samples=%d",
            w_idx + 1,
            acc,
            len(y_test_cls),
        )

        last_lgbm_model = model
        last_feature_cols = static_cols
        last_window_accuracy = acc

        if is_regression_static:
            chunk_data = {
                "timestamp": test_df["timestamp"],
                "true_label": y_test_cls,
                "pred_label": preds.astype(np.int32),
                "pred_raw": raw_preds.astype(np.float64),
            }
        else:
            chunk_data = {
                "timestamp": test_df["timestamp"],
                "true_label": y_test_cls,
                "pred_label": preds.astype(np.int32),
                **_probability_columns(proba, model.classes_),
            }
        all_oof_preds.append(pl.DataFrame(chunk_data))
        logger.info(
            "Static window %d done (%.1fs)",
            w_idx + 1,
            time.perf_counter() - window_start,
        )

    if not all_oof_preds or last_lgbm_model is None:
        raise RuntimeError("No static OOF predictions generated")

    oof_df = pl.concat(all_oof_preds)
    ts_col = oof_df["timestamp"]
    if ts_col.n_unique() < len(ts_col):
        dup_count = len(ts_col) - ts_col.n_unique()
        raise ValueError(
            f"OOF predictions contain {dup_count} duplicate timestamps — "
            "walk-forward test windows should be non-overlapping."
        )

    preds_path = Path(config.paths.predictions)
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    oof_df.write_parquet(preds_path)
    oof_df.write_csv(preds_path.with_suffix(".csv"))

    model_path = Path(config.paths.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(last_lgbm_model, model_path)
    _save_feature_importance(last_lgbm_model, last_feature_cols, config)

    if config.paths.session_dir:
        models_dir = Path(config.paths.session_dir) / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        history_path = models_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(
                {
                    "architecture": "static",
                    "lightgbm": {
                        "artifact_strategy": "last_walk_forward_window",
                        "validation_protocol": {
                            "outer_windows": "bar_based_walk_forward_with_purge_embargo",
                            "lgbm_validation": "tail_20_percent_of_outer_train",
                        },
                        "last_window_accuracy": last_window_accuracy,
                        "best_iteration": int(last_lgbm_model.best_iteration_)
                        if hasattr(last_lgbm_model, "best_iteration_")
                        else None,
                        "n_features": len(last_feature_cols),
                        "n_classes": len(last_lgbm_model.classes_)
                        if hasattr(last_lgbm_model, "classes_")
                        else None,
                    },
                },
                f,
                indent=2,
            )

        wf_path = (
            Path(config.paths.session_dir) / "reports" / "walk_forward_history.json"
        )
        wf_path.parent.mkdir(parents=True, exist_ok=True)
        with open(wf_path, "w") as f:
            json.dump(
                {
                    "architecture": "static",
                    "num_windows": len(windows),
                    "total_oof_predictions": len(oof_df),
                    "window_details": [
                        {
                            "window": i + 1,
                            "train_start_idx": w.train_start_idx,
                            "train_end_idx": w.train_end_idx,
                            "test_start_idx": w.test_start_idx,
                            "test_end_idx": w.test_end_idx,
                            **next(
                                (
                                    item
                                    for item in window_diagnostics
                                    if item["window"] == i + 1
                                ),
                                {},
                            ),
                        }
                        for i, w in enumerate(windows)
                    ],
                },
                f,
                indent=2,
            )

    logger.info(
        "Static walk-forward complete: %d windows, %d OOF predictions (%.1fs)",
        len(windows),
        len(oof_df),
        time.perf_counter() - stage_start,
    )


def _label_suffix(class_label: int) -> str:
    """Return the canonical probability-column suffix for a class label."""
    return f"minus{abs(class_label)}" if class_label < 0 else str(class_label)


def _align_probability_matrix(
    proba: np.ndarray,
    class_order: list[int] | np.ndarray,
) -> np.ndarray:
    """Align class probabilities to the canonical ``[-1, 0, 1]`` order."""
    aligned = np.zeros((len(proba), len(_CLASS_ORDER)), dtype=np.float64)
    index_by_class = {int(cls): idx for idx, cls in enumerate(class_order)}
    for target_idx, cls in enumerate(_CLASS_ORDER):
        source_idx = index_by_class.get(int(cls))
        if source_idx is not None:
            aligned[:, target_idx] = proba[:, source_idx]
    return aligned


def _probability_columns(
    proba: np.ndarray,
    class_order: list[int] | np.ndarray,
    *,
    prefix: str = "pred_proba_class_",
) -> dict[str, np.ndarray]:
    """Build canonical probability columns for ``{-1, 0, 1}``."""
    aligned = _align_probability_matrix(proba, class_order)
    return {
        f"{prefix}{_label_suffix(int(cls))}": aligned[:, idx]
        for idx, cls in enumerate(_CLASS_ORDER)
    }


def _split_tail_frame(
    df: pl.DataFrame,
    *,
    fraction: float = 0.2,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split a DataFrame into head train and tail validation slices."""
    if len(df) < 2:
        raise ValueError("Need at least 2 rows to create a train/validation split")
    val_size = min(max(1, int(len(df) * fraction)), len(df) - 1)
    return df.head(len(df) - val_size), df.tail(val_size)


def _split_tail_arrays(
    X: np.ndarray,
    y: np.ndarray,
    reference_prices: np.ndarray,
    *,
    fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Split matrices into train/validation tails and return val price anchor."""
    if len(X) < 2:
        raise ValueError("Need at least 2 rows to create a train/validation split")
    val_size = min(max(1, int(len(X) * fraction)), len(X) - 1)
    X_tr = X[:-val_size]
    y_tr = y[:-val_size]
    X_val = X[-val_size:]
    y_val = y[-val_size:]
    reference_price = float(np.median(reference_prices[-val_size:]))
    return X_tr, y_tr, X_val, y_val, reference_price


def _fit_lgbm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    reference_price: float,
    config: Config,
    feature_cols: list[str],
    *,
    trials_override: int | None = None,
    sample_weight: np.ndarray | None = None,
) -> Any:
    """Train a LightGBM model using fixed hyperparameters."""
    from thesis.model import (
        _compute_class_weights,
        _train_fixed,
    )

    if config.model.objective == "regression":
        class_weights = None
    else:
        class_weights = _compute_class_weights(y_train)

    return _train_fixed(
        X_train,
        y_train,
        X_val,
        y_val,
        class_weights,
        config,
        feature_cols,
        sample_weight=sample_weight,
    )


def _run_walk_forward(config: Config) -> None:
    """Dispatch walk-forward training to the configured architecture."""
    architecture = config.model.architecture

    if architecture == "static":
        logger.info("Using static-feature-only walk-forward baseline")
        _run_walk_forward_static(config, expanded_features=config.model.static_expanded)
        return

    if architecture != "hybrid":
        raise ValueError(f"Unsupported model.architecture: {architecture!r}")

    logger.info("Using hybrid walk-forward pipeline")
    _run_walk_forward_hybrid(config)


def _run_static_train(config: Config) -> None:
    """Run traditional static train/val/test split training.

    WARNING: Static split does not apply purge/embargo at the split boundary.
    With triple-barrier labels (horizon_bars > 0), labels near the boundary
    may use future information from the adjacent split. For thesis evaluation,
    use validation.method = "sliding" instead.
    """
    from thesis.model import train_model

    logger.warning(
        "Static split mode does not apply purge/embargo — potential label leakage "
        "at split boundaries. Recommended: validation.method = 'sliding'."
    )
    train_model(config)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(config: Config) -> None:
    """Execute the full thesis pipeline.

    Stages:
        0. Data preparation (tick → OHLCV)
        1. Feature engineering
        2. Triple-barrier labeling
        3. Walk-forward model training (GRU + LightGBM per window)
        4. Backtest (on concatenated OOF predictions)
        5. Report generation

    Args:
        config: Loaded application configuration.
    """
    # Stage 0: Prepare OHLCV from raw ticks
    _run_stage(0, config, "run_data_pipeline", config.paths.ohlcv, prepare_data)

    # Stage 1: Features
    _run_stage(
        1,
        config,
        "run_feature_engineering",
        config.paths.features,
        generate_features,
    )

    # Stage 2: Labels
    _run_stage(2, config, "run_label_generation", config.paths.labels, generate_labels)

    # Stage 3: Training (walk-forward or static)
    if config.validation.method == "sliding":
        stage_header(3)
        logger.info(
            "Using walk-forward sliding window validation (%s architecture)",
            config.model.architecture,
        )
        if config.workflow.run_model_training:
            _run_walk_forward(config)
        else:
            stage_skip(3, "disabled")
    else:
        logger.info("Using static train/val/test split")
        _run_stage(3, config, "run_model_training", None, _run_static_train)

    # Stage 4: Backtest
    if config.workflow.run_backtest:
        tp_l = config.labels.atr_tp_multiplier
        sl_l = config.labels.atr_sl_multiplier
        tp_b = config.backtest.atr_tp_multiplier
        sl_b = config.backtest.atr_stop_multiplier
        if tp_l != tp_b or sl_l != sl_b:
            logger.warning(
                "Label and backtest barrier multipliers differ: "
                "labels (%.1f/%.1f) vs backtest (%.1f/%.1f)",
                tp_l,
                sl_l,
                tp_b,
                sl_b,
            )

    _run_stage(
        4,
        config,
        "run_backtest",
        None,
        run_backtest,
    )

    # Stage 5: Report
    _run_stage(
        5,
        config,
        "run_reporting",
        None,
        generate_report,
    )

    console.print()
    console.rule("[bold green]Pipeline Complete[/]")
    console.print()
