"""Walk-forward training orchestration — hybrid (GRU+LGBM) and static (LGBM-only)."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from thesis._shared.config import Config
from thesis._shared.constants import CENSORED_LABEL, EXCLUDE_COLS
from thesis._shared.ui import console
from thesis.stage_4_training._validation import generate_windows, log_windows

logger = logging.getLogger("thesis.pipeline")

# Module-level constants

_CLASS_ORDER = np.array([-1, 0, 1], dtype=np.int32)

# --- Confidence & Signal Quality Thresholds ---
_HIGH_CONFIDENCE_THRESHOLD = 0.70  # High-confidence prediction floor
_SHORT_BIAS_RATIO_THRESHOLD = 0.5  # LONG/SHORT ratio warning trigger
_GRU_SIGNAL_F_SCORE_THRESHOLD = 0.5  # Mean F-score below → no detectable signal
_PCA_VARIANCE_THRESHOLD = 0.50  # Explained variance below → mostly noise

# --- Validation Split ---
_VALIDATION_SPLIT_FRACTION = 0.2  # Tail validation split for GRU/LGBM/static

# --- Minimum Sample Thresholds ---
_ANOVA_MIN_SAMPLES_PER_CLASS = 2  # Minimum samples per class for ANOVA F-statistic
_STATIC_MIN_TRAIN_ROWS = 2  # Minimum training rows for static walk-forward

# --- Display / Logging ---
_SIGNAL_QUALITY_TOP_N = 5  # Top-N for GRU signal quality logging

# --- Regression Threshold ---
_REGRESSION_DIRECTION_THRESHOLD = (
    0  # Zero threshold for regression-to-direction mapping
)


# Utility helpers


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


def _validate_predictions(df: pl.DataFrame, path: Path) -> None:
    """Validate final OOF predictions before writing the parquet artifact."""
    required = {"timestamp", "pred_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Predictions missing columns {sorted(missing)}: file={path}")
    if len(df) == 0:
        raise ValueError(f"Predictions are empty: file={path}")

    ts_col = df["timestamp"]
    if ts_col.null_count() > 0:
        raise ValueError(
            f"Predictions timestamp has nulls: actual={ts_col.null_count()}, file={path}"
        )
    if ts_col.n_unique() < len(ts_col):
        dup_count = len(ts_col) - ts_col.n_unique()
        raise ValueError(
            f"OOF predictions contain {dup_count} duplicate timestamps — "
            "walk-forward test windows should be non-overlapping. "
            f"Check step_bars vs test_window_bars. file={path}"
        )
    if ts_col.to_list() != sorted(ts_col.to_list()):
        raise ValueError(f"OOF predictions must be sorted by timestamp: file={path}")

    pred_col = df["pred_label"]
    if pred_col.null_count() > 0:
        raise ValueError(
            f"pred_label has nulls: actual={pred_col.null_count()}, file={path}"
        )
    invalid = sorted(set(pred_col.unique().to_list()) - {-1, 0, 1})
    if invalid:
        raise ValueError(
            f"Invalid pred_label values: expected={{-1,0,1}}, actual={invalid}, file={path}"
        )

    null_cols = {
        col: df[col].null_count() for col in df.columns if df[col].null_count()
    }
    if null_cols:
        raise ValueError(f"Predictions contain nulls: actual={null_cols}, file={path}")


def _write_prediction_manifest(
    df: pl.DataFrame,
    path: Path,
    *,
    windows_count: int,
) -> None:
    """Write compact diagnostics beside final_predictions.parquet."""
    mean_confidence = (
        float(df["max_confidence"].mean()) if "max_confidence" in df.columns else None
    )
    manifest = {
        "row_count": len(df),
        "start": str(df["timestamp"][0]),
        "end": str(df["timestamp"][-1]),
        "label_distribution": _counts_dict(df["true_label"].to_numpy())
        if "true_label" in df.columns
        else {},
        "prediction_distribution": _counts_dict(df["pred_label"].to_numpy()),
        "mean_confidence": mean_confidence,
        "windows_count": windows_count,
    }
    manifest_path = path.with_name("prediction_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Prediction manifest saved: %s", manifest_path)


def _window_diagnostics(
    window_idx: int,
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Build per-window label diagnostics for logs and JSON artifacts.

    Args:
        window_idx: Zero-based window index.
        train_df: Training split Polars DataFrame.
        test_df: Test split Polars DataFrame.
        y_train: Training label array.
        y_test: Test label array.

    Returns:
        Dictionary with window index, row counts, date ranges, label
        counts (raw and percentage) for both train and test splits.
    """
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


def _compute_per_class_metrics(
    preds: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Compute per-class precision, recall, F1, and support from predictions.

    Uses ``sklearn.metrics.precision_recall_fscore_support`` with
    ``zero_division=0`` so missing classes return 0.0 rather than raising.

    Args:
        preds: Predicted class labels as a NumPy array.
        y_test: Ground-truth class labels.

    Returns:
        Mapping ``{class_label_str: {"precision", "recall", "f1", "support"}}``
        with string keys (``"-1"``, ``"0"``, ``"1"``) for JSON serialization.
    """
    from sklearn.metrics import precision_recall_fscore_support

    classes = np.array([-1, 0, 1], dtype=np.int32)
    p, r, f1, s = precision_recall_fscore_support(
        y_test, preds, labels=classes, zero_division=0
    )
    return {
        str(int(cls)): {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(s[i]),
        }
        for i, cls in enumerate(classes)
    }


def _add_prediction_diagnostics(
    diag: dict[str, Any],
    preds: np.ndarray,
    y_test: np.ndarray,
    proba: np.ndarray,
) -> None:
    """Attach prediction distribution, confidence, and per-class metrics.

    Mutates ``diag`` by adding prediction counts, accuracy, mean
    confidence, high-confidence fraction, long/short ratio, and per-class
    precision / recall / F1.

    Args:
        diag: Per-window diagnostics dict (mutated in-place).
        preds: Predicted class labels as a NumPy array.
        y_test: Ground-truth class labels.
        proba: Probability matrix (N x 3).
    """
    pred_counts = _counts_dict(preds)
    confidence = np.max(proba, axis=1) if len(proba) else np.array([], dtype=float)

    # Compute LONG/SHORT prediction ratio
    long_count = pred_counts.get("1", 0)
    short_count = pred_counts.get("-1", 0)
    ls_ratio = long_count / short_count if short_count > 0 else float("inf")

    per_class = _compute_per_class_metrics(preds, y_test) if len(y_test) else {}
    diag.update(
        {
            "prediction_counts": pred_counts,
            "prediction_pct": _pct_dict(pred_counts),
            "accuracy": float((preds == y_test).mean()) if len(y_test) else None,
            "mean_confidence": float(confidence.mean()) if len(confidence) else None,
            "high_conf_70_pct": float(
                (confidence >= _HIGH_CONFIDENCE_THRESHOLD).mean() * 100.0
            )
            if len(confidence)
            else None,
            "ls_ratio": round(ls_ratio, 4) if short_count > 0 else None,
            "per_class": per_class,
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
    if per_class:
        logger.info(
            "Window %d per-class | SHORT: P=%.3f R=%.3f F1=%.3f | "
            "HOLD: P=%.3f R=%.3f F1=%.3f | "
            "LONG: P=%.3f R=%.3f F1=%.3f",
            diag["window"],
            per_class["-1"]["precision"],
            per_class["-1"]["recall"],
            per_class["-1"]["f1"],
            per_class["0"]["precision"],
            per_class["0"]["recall"],
            per_class["0"]["f1"],
            per_class["1"]["precision"],
            per_class["1"]["recall"],
            per_class["1"]["f1"],
        )
    if short_count > 0 and long_count / short_count < _SHORT_BIAS_RATIO_THRESHOLD:
        logger.warning(
            "Window %d: SHORT bias — LONG/SHORT ratio = %.2f",
            diag["window"],
            long_count / short_count,
        )
    elif long_count > 0 and short_count / long_count < _SHORT_BIAS_RATIO_THRESHOLD:
        logger.warning(
            "Window %d: LONG bias — SHORT/LONG ratio = %.2f",
            diag["window"],
            short_count / long_count,
        )
    else:
        logger.info(
            "Window %d: L/S balanced — ratio %.2f",
            diag["window"],
            ls_ratio if short_count > 0 else float("inf"),
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

    min_samples_per_class = _ANOVA_MIN_SAMPLES_PER_CLASS
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

    top_n = min(_SIGNAL_QUALITY_TOP_N, n_features)
    bottom_n = min(_SIGNAL_QUALITY_TOP_N, n_features)

    top_indices = sorted_indices[:top_n]
    bottom_indices = sorted_indices[-bottom_n:][::-1]  # ascending for bottom display

    mean_f = float(np.mean(f_scores))

    logger.info(
        "GRU hidden signal quality: mean F=%.4f | top-5: %s | bottom-5: %s",
        mean_f,
        ", ".join(f"dim{i}={f_scores[i]:.3f}" for i in top_indices),
        ", ".join(f"dim{i}={f_scores[i]:.3f}" for i in bottom_indices),
    )

    if mean_f < _GRU_SIGNAL_F_SCORE_THRESHOLD:
        logger.warning(
            "GRU hidden states show no detectable signal — GRU contributes noise "
            "(mean F=%.4f across %d dimensions)",
            mean_f,
            n_features,
        )


# Walk-forward training loop — hybrid


def _compute_regression_target(
    df: pl.DataFrame, config: Config
) -> tuple[pl.DataFrame, bool]:
    """Pre-compute regression target column when objective is 'regression'.

    The last ``horizon_bars`` rows are set to NaN (insufficient forward
    data) and their ``label`` is set to ``CENSORED_LABEL`` so downstream
    filters exclude them.  The rows are dropped before returning.

    Args:
        df: Labeled Polars DataFrame containing a ``close`` column.
        config: Application configuration.

    Returns:
        ``(df_maybe_augmented, is_regression)`` — the DataFrame is
        augmented with a ``regression_target`` column when the
        objective is ``"regression"``; otherwise returned unchanged
        with ``is_regression=False``.
    """
    is_regression = config.model.objective == "regression"
    gru_needs_regression = config.gru.objective == "regression"
    if not is_regression and not gru_needs_regression:
        return df, False

    if "close" not in df.columns:
        raise ValueError(
            "Regression objective requires 'close' column in labeled data. "
            "Ensure feature engineering includes OHLCV data."
        )
    horizon = config.labels.horizon_bars
    close = df["close"].to_numpy()
    n = len(close)

    # Compute forward returns, leaving the last ``horizon`` rows as NaN.
    reg_target = np.full(n, np.nan, dtype=np.float64)
    close_future = np.roll(close, -horizon)[: n - horizon]
    reg_target[: n - horizon] = (close_future - close[: n - horizon]) / close[
        : n - horizon
    ]

    # Mark tail rows as censored so downstream _filter_censored excludes them.
    label_arr = df["label"].to_numpy().copy()
    tail_start = max(0, n - horizon)
    label_arr[tail_start:] = CENSORED_LABEL

    df = df.with_columns(
        [
            pl.Series("regression_target", reg_target),
            pl.Series("label", label_arr),
        ]
    )

    # Drop rows where the regression target is NaN (tail censored rows).
    n_before = len(df)
    df = df.filter(pl.col("regression_target").is_not_nan())
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.info(
            "Dropped %d regression tail rows (%d horizon bars) — "
            "insufficient forward horizon",
            n_dropped,
            horizon,
        )

    logger.info(
        "Regression target computed: horizon=%d bars, mean=%.6f, std=%.6f",
        horizon,
        float(np.nanmean(reg_target)),
        float(np.nanstd(reg_target)),
    )
    return df, is_regression


def _prepare_wf_data(
    config: Config,
) -> tuple[pl.DataFrame, list, list[str], bool]:
    """Load labeled data, generate walk-forward windows, and return prepared state.

    Args:
        config: Application configuration.

    Returns:
        ``(df, windows, feature_cols, is_regression)`` tuple containing the
        full labeled DataFrame, list of walk-forward window objects, sorted
        feature column names, and whether regression objective is active.

    Raises:
        FileNotFoundError: If the labels parquet file does not exist.
        RuntimeError: If no valid walk-forward windows were generated.
        ValueError: If the purge/embargo gap is smaller than the GRU
            sequence length (sequence leakage risk).
    """
    labels_path = Path(config.paths.labels)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels not found: {labels_path}")

    with console.status(f"[cyan]Loading labels[/] {labels_path}"):
        df = pl.read_parquet(labels_path)
    logger.info("Loaded labeled data: %d rows", len(df))
    df, is_regression = _compute_regression_target(df, config)

    event_end = df["event_end"].to_numpy() if "event_end" in df.columns else None
    if event_end is None:
        logger.warning(
            "Labels lack event_end column — falling back to fixed-bar purge. "
            "Regenerate labels to enable event-time purging."
        )

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

    # P0-1: Guard against sequence leakage
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

    feature_cols = sorted(c for c in df.columns if c not in EXCLUDE_COLS)
    return df, windows, feature_cols, is_regression


def _wf_gru_phase(
    config: Config, w_idx: int, window: Any, df: pl.DataFrame
) -> dict[str, Any] | None:
    """GRU phase of a hybrid window: slice, train, extract hidden, align, PCA.

    Args:
        config: Application configuration.
        w_idx: Zero-based window index for logging.
        window: Walk-forward window object with ``train_start_idx``,
            ``train_end_idx``, ``test_start_idx``, ``test_end_idx``.
        df: Full labeled Polars DataFrame.

    Returns:
        State dictionary with ``gru_model``, normalization
        parameters, training history, aligned DataFrames, and
        hidden-state arrays, or ``None`` if the window is too small.
    """
    import torch

    from thesis.stage_4_training._gru import (
        train_gru,
        extract_hidden_states,
        prepare_sequences,
    )

    # Slice
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
        return None

    # Train GRU
    val_split = max(1, int(len(train_df) * _VALIDATION_SPLIT_FRACTION))
    gru_train_df = train_df.head(len(train_df) - val_split)
    gru_val_df = train_df.tail(val_split)
    (gru_model, _, _, _, gru_history, gru_mean, gru_std, dynamic_gru_cols) = train_gru(
        config, gru_train_df, gru_val_df, window_index=w_idx
    )

    # Extract hidden states
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

    # Align DataFrames
    train_aligned = train_df.slice(seq_len - 1, len(train_hidden))
    test_aligned = test_df.slice(seq_len - 1, len(test_hidden))
    if len(train_aligned) == 0 or len(test_aligned) == 0:
        logger.warning("Window %d: aligned data empty, skipping", w_idx + 1)
        return None
    train_dates = _window_dates(train_df)
    test_dates = _window_dates(test_df)
    pred_dates = _window_dates(test_aligned)
    logger.info(
        "Window %d alignment: train_start=%s train_end=%s test_start=%s "
        "test_end=%s raw_test_rows=%d aligned_test_rows=%d "
        "dropped_by_sequence=%d pred_start=%s pred_end=%s",
        w_idx + 1,
        train_dates["start"],
        train_dates["end"],
        test_dates["start"],
        test_dates["end"],
        len(test_df),
        len(test_aligned),
        len(test_df) - len(test_aligned),
        pred_dates["start"],
        pred_dates["end"],
    )

    # PCA on GRU hidden states
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
        if explained < _PCA_VARIANCE_THRESHOLD:
            logger.warning(
                "GRU hidden state space appears mostly noise (%.1f%% explained by %d PCs)",
                explained * 100,
                pca_k,
            )

    return {
        "gru_model": gru_model,
        "gru_mean": gru_mean,
        "gru_std": gru_std,
        "gru_history": gru_history,
        "train_aligned": train_aligned,
        "test_aligned": test_aligned,
        "train_hidden": train_hidden,
        "test_hidden": test_hidden,
    }


def _wf_format_predictions(
    model: Any, X_test: np.ndarray, all_feature_cols: list[str], is_regression: bool
) -> tuple[np.ndarray, np.ndarray, Any, Any]:
    """Generate predictions and aligned probability matrix.

    Args:
        model: Trained LightGBM model (classifier or regressor).
        X_test: Test feature matrix.
        all_feature_cols: Feature column names for wrapping.
        is_regression: Whether the model is a regressor.

    Returns:
        ``(preds, aligned_proba, proba, raw_preds)`` — predicted class
        labels, aligned 3-column probability matrix, raw LightGBM
        probability output, and raw regression predictions (``None``
        for classification).
    """
    from thesis.stage_4_training._lgbm import _wrap_np

    if is_regression:
        raw_preds = model.predict(_wrap_np(X_test, all_feature_cols))
        preds = np.where(
            raw_preds > _REGRESSION_DIRECTION_THRESHOLD,
            1,
            np.where(raw_preds < _REGRESSION_DIRECTION_THRESHOLD, -1, 0),
        ).astype(np.int32)
        aligned_proba = np.zeros((len(raw_preds), 3), dtype=np.float64)
        for i, p in enumerate(preds):
            aligned_proba[i, {-1: 0, 0: 1, 1: 2}[int(p)]] = 1.0
        return preds, aligned_proba, None, raw_preds
    else:
        proba = model.predict_proba(_wrap_np(X_test, all_feature_cols))
        aligned_proba = _align_probability_matrix(proba, model.classes_)
        preds = _CLASS_ORDER[np.argmax(aligned_proba, axis=1)]
        return preds, aligned_proba, proba, None


def _wf_build_predict_phase(
    config: Config,
    w_idx: int,
    gru_state: dict[str, Any],
    feature_cols: list[str],
    is_regression: bool,
) -> dict[str, Any]:
    """Build hybrid matrix, train LGBM, predict, return full window result.

    Args:
        config: Application configuration.
        w_idx: Zero-based window index.
        gru_state: GRU phase state dict from ``_wf_gru_phase``.
        feature_cols: Candidate feature column names.
        is_regression: Whether the model objective is regression.

    Returns:
        Full window result dict containing the GRU state, test labels,
        predictions, probabilities, trained LGBM model, feature
        columns, accuracy, diagnostics, and class ordering.
    """
    from thesis.stage_4_training._lgbm import (
        _compute_class_weights,
        _compute_distribution_shift_weights,
        _train_fixed,
    )

    train_aligned = gru_state["train_aligned"]
    test_aligned = gru_state["test_aligned"]
    train_hidden = gru_state["train_hidden"]
    test_hidden = gru_state["test_hidden"]

    # ── Build hybrid feature matrix ──
    static_cols = _select_static_feature_cols(config, train_aligned, feature_cols)
    pca_k = config.gru.pca_components
    hidden_components = pca_k if pca_k > 0 else config.gru.hidden_size
    gru_feat_names = [
        f"gru_pc_{i}" if pca_k > 0 else f"gru_h{i}" for i in range(hidden_components)
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
    reg_y_train: np.ndarray | None = None
    if is_regression:
        reg_y_train = train_aligned["regression_target"].to_numpy().astype(np.float64)

    # ── Diagnostics & weights ──
    _log_gru_signal_quality(train_hidden, y_train, config)
    diag = _window_diagnostics(w_idx + 1, train_aligned, test_aligned, y_train, y_test)
    train_weights = (
        train_aligned["sample_weight"].to_numpy().astype(np.float64)
        if "sample_weight" in train_aligned.columns
        else None
    )

    # ── Train LightGBM ──
    val_split_idx = max(1, int(len(X_train) * _VALIDATION_SPLIT_FRACTION))
    X_tr = X_train[:-val_split_idx]
    w_tr = train_weights[:-val_split_idx] if train_weights is not None else None
    X_val = X_train[-val_split_idx:]
    shift_ratios: dict[str, float] | None = None
    if is_regression:
        y_tr = reg_y_train[:-val_split_idx]  # type: ignore[index]
        y_val = reg_y_train[-val_split_idx:]  # type: ignore[index]
        class_weights = None
        combined_weights = w_tr
    else:
        y_tr = y_train[:-val_split_idx]
        y_val = y_train[-val_split_idx:]
        class_weights = _compute_class_weights(y_tr)
        shift_weights, shift_ratios = _compute_distribution_shift_weights(y_tr, y_val)
        # Combine with existing sample weights (average-uniqueness) if present
        if w_tr is not None:
            combined_weights = w_tr * shift_weights
        else:
            combined_weights = shift_weights

    # ── Attach weight diagnostics to window diag ──
    diag["class_weights"] = (
        {str(k): v for k, v in class_weights.items()} if class_weights else None
    )
    diag["shift_weights_per_class"] = shift_ratios

    model = _train_fixed(
        X_tr,
        y_tr,
        X_val,
        y_val,
        class_weights,
        config,
        all_feature_cols,
        sample_weight=combined_weights,
    )

    # ── Predict ──
    preds, aligned_proba, proba, raw_preds = _wf_format_predictions(
        model, X_test, all_feature_cols, is_regression
    )
    _add_prediction_diagnostics(diag, preds, y_test, aligned_proba)
    acc = (preds == y_test).mean()
    logger.info(
        "Window %d: accuracy=%.4f, test_samples=%d", w_idx + 1, acc, len(y_test)
    )

    return {
        **gru_state,
        "test_aligned": test_aligned,
        "y_test": y_test,
        "preds": preds,
        "raw_preds": raw_preds,
        "proba": proba,
        "model": model,
        "all_feature_cols": all_feature_cols,
        "accuracy": float(acc),
        "diag": diag,
        "classes": model.classes_,
    }


def _run_hybrid_window(
    config: Config,
    w_idx: int,
    window: Any,
    df: pl.DataFrame,
    feature_cols: list[str],
    is_regression: bool,
) -> dict[str, Any] | None:
    """Run a single hybrid walk-forward window: GRU → PCA → LGBM → predict.

    Delegates to ``_wf_gru_phase`` and ``_wf_build_predict_phase``.
    """
    gru_state = _wf_gru_phase(config, w_idx, window, df)
    if gru_state is None:
        return None
    return _wf_build_predict_phase(
        config, w_idx, gru_state, feature_cols, is_regression
    )


def _collect_oof_predictions(
    result: dict[str, Any], is_regression: bool
) -> pl.DataFrame:
    """Build OOF prediction chunk from a single window result.

    Args:
        result: Window result dict containing ``test_aligned`` DataFrame,
            ``y_test`` array, ``preds`` array, and (for classification)
            ``proba`` matrix with ``classes`` ordering.
        is_regression: Whether the model is a regressor.

    Returns:
        Polars DataFrame with ``timestamp``, ``true_label``,
        ``pred_label``, and probability columns (classification) or
        ``pred_raw`` (regression).
    """
    test_aligned: pl.DataFrame = result["test_aligned"]
    y_test: np.ndarray = result["y_test"]
    preds: np.ndarray = result["preds"]
    if is_regression:
        preds_int = preds.astype(np.int32)
        return pl.DataFrame(
            {
                "timestamp": test_aligned["timestamp"],
                "true_label": y_test,
                "pred_label": preds_int,
                "pred_raw": result["raw_preds"].astype(np.float64),
                **_one_hot_proba_columns(preds_int),
            }
        )
    return pl.DataFrame(
        {
            "timestamp": test_aligned["timestamp"],
            "true_label": y_test,
            "pred_label": preds.astype(np.int32),
            **_probability_columns(result["proba"], result["classes"]),
        }
    )


def _build_lgbm_info(
    last_lgbm_model: Any,
    last_feature_cols: list[str],
    last_window_accuracy: float | None,
    window_index: int | None = None,
    total_windows: int = 0,
    window_train_dates: dict[str, str] | None = None,
    window_test_dates: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build LightGBM metadata dict for training history JSON.

    Args:
        last_lgbm_model: Trained LightGBM model from the last window.
        last_feature_cols: Feature column names used by the model.
        last_window_accuracy: OOF accuracy of the last window, or ``None``.
        window_index: 1-based index of the window that produced the model.
        total_windows: Total number of walk-forward windows.
        window_train_dates: ``{"start": ..., "end": ...}`` date range for
            the last window's training data.
        window_test_dates: ``{"start": ..., "end": ...}`` date range for
            the last window's test data.

    Returns:
        Dictionary with validation protocol, window provenance, best
        iteration, feature count, and class count.
    """
    info: dict[str, Any] = {
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
    if window_index is not None:
        info["window_index"] = window_index
        info["total_windows"] = total_windows
        info["window_train_date_range"] = window_train_dates or {}
        info["window_test_date_range"] = window_test_dates or {}
        info["window_oof_accuracy"] = last_window_accuracy
    return info


def _build_wf_history(
    windows: list,
    window_diagnostics: list[dict[str, Any]],
    oof_len: int,
) -> dict[str, Any]:
    """Build walk-forward history dict with per-window details.

    Args:
        windows: List of walk-forward window objects with index attrs.
        window_diagnostics: Per-window diagnostic dictionaries.
        oof_len: Total number of OOF predictions.

    Returns:
        Dictionary with ``num_windows``, ``total_oof_predictions``,
        and ``window_details`` list.
    """
    return {
        "num_windows": len(windows),
        "total_oof_predictions": oof_len,
        "window_details": [
            {
                "window": i + 1,
                "train_start_idx": w.train_start_idx,
                "train_end_idx": w.train_end_idx,
                "test_start_idx": w.test_start_idx,
                "test_end_idx": w.test_end_idx,
                **next(
                    (item for item in window_diagnostics if item["window"] == i + 1), {}
                ),
            }
            for i, w in enumerate(windows)
        ],
    }


def _save_wf_artifacts(
    config: Config,
    all_oof_preds: list[pl.DataFrame],
    gru_model: Any,
    gru_mean: Any,
    gru_std: Any,
    last_lgbm_model: Any,
    last_feature_cols: list[str],
    last_window_accuracy: float | None,
    last_window_index: int,
    last_gru_history: list[dict],
    windows: list,
    window_diagnostics: list[dict[str, Any]],
    stage_start: float,
    is_regression: bool,
) -> None:
    """Validate OOF predictions and persist all walk-forward artifacts.

    Args:
        config: Application configuration.
        all_oof_preds: List of per-window OOF Polars DataFrames.
        gru_model: GRU model from the last window.
        gru_mean: Normalization mean for the GRU model.
        gru_std: Normalization std for the GRU model.
        last_lgbm_model: LightGBM model from the last window.
        last_feature_cols: Feature column names.
        last_window_accuracy: Accuracy of the last window.
        last_window_index: 1-based index of the last window.
        last_gru_history: GRU training history list.
        windows: Walk-forward window objects.
        window_diagnostics: Per-window diagnostic dictionaries.
        stage_start: ``time.perf_counter()`` start timestamp.
        is_regression: Whether the objective is regression.

    Raises:
        RuntimeError: If no predictions were generated.
        ValueError: If duplicate timestamps are found in OOF data.
    """
    import joblib

    from thesis.stage_4_training._gru import save_gru_model
    from thesis.stage_4_training._lgbm import _save_feature_importance

    # ── Guard ──
    if not all_oof_preds or gru_model is None:
        raise RuntimeError(
            "No OOF predictions generated — all walk-forward windows were skipped"
        )

    # ── Save GRU model (last window) ──
    if config.paths.session_dir:
        gru_path = Path(config.paths.session_dir) / "models" / "gru_model.pt"
        save_gru_model(gru_model, config, gru_path, mean=gru_mean, std=gru_std)

    # ── Concatenate & validate OOF ──
    oof_df = pl.concat(all_oof_preds)
    oof_df = _add_confidence_columns(oof_df)

    preds_path = Path(config.paths.predictions)
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    _validate_predictions(oof_df, preds_path)
    oof_df.write_parquet(preds_path)
    oof_df.write_csv(preds_path.with_suffix(".csv"))
    _write_prediction_manifest(
        oof_df,
        preds_path,
        windows_count=len(window_diagnostics),
    )

    # ── Save LGBM model + feature importance ──
    if last_lgbm_model is not None:
        model_path = Path(config.paths.model)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(last_lgbm_model, model_path)
    if last_lgbm_model is not None and last_feature_cols:
        _save_feature_importance(last_lgbm_model, last_feature_cols, config)

    # ── Save training history ──
    if config.paths.session_dir:
        models_dir = Path(config.paths.session_dir) / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        history_path = models_dir / "training_history.json"

        # Build per-window accuracy map from diagnostics
        per_window_accuracies: dict[str, float | None] = {}
        for d in window_diagnostics:
            key = str(d.get("window", ""))
            if key:
                per_window_accuracies[key] = d.get("accuracy")

        # Extract last-window date ranges from diagnostics
        last_train_dates: dict[str, str] = {}
        last_test_dates: dict[str, str] = {}
        last_win_key = str(last_window_index)
        for d in window_diagnostics:
            if str(d.get("window", "")) == last_win_key:
                last_train_dates = d.get("train_dates", {})
                last_test_dates = d.get("test_dates", {})
                break

        lgbm_info = (
            _build_lgbm_info(
                last_lgbm_model,
                last_feature_cols,
                last_window_accuracy,
                window_index=last_window_index,
                total_windows=len(windows),
                window_train_dates=last_train_dates,
                window_test_dates=last_test_dates,
            )
            if last_lgbm_model is not None
            else {}
        )

        deployment_note = (
            f"Model saved from window {last_window_index}/{len(windows)} "
            "(the last chronological walk-forward window). "
            "This model has NOT seen any future data beyond its training window."
        )

        with open(history_path, "w") as f:
            json.dump(
                {
                    "gru": last_gru_history,
                    "lightgbm": lgbm_info,
                    "deployment_note": deployment_note,
                    "per_window_accuracies": per_window_accuracies,
                },
                f,
                indent=2,
            )
        logger.info("Training history saved to %s", history_path)

    # ── Save walk-forward history ──
    if config.paths.session_dir:
        wf_path = (
            Path(config.paths.session_dir) / "reports" / "walk_forward_history.json"
        )
        wf_path.parent.mkdir(parents=True, exist_ok=True)
        with open(wf_path, "w") as f:
            json.dump(
                _build_wf_history(windows, window_diagnostics, len(oof_df)), f, indent=2
            )

    total_time = time.perf_counter() - stage_start
    logger.info(
        "Walk-forward complete: %d windows, %d OOF predictions (%.1fs)",
        len(windows),
        len(oof_df),
        total_time,
    )


def _run_walk_forward_hybrid(config: Config) -> None:
    """Execute walk-forward hybrid training across all windows.

    Orchestration: load → windows → loop(GPU→PCA→LGBM→collect) → save.
    Each step delegates to a focused helper ≤ 80 lines.
    """
    # 1. Load labeled data, compute regression target, generate windows
    df, windows, feature_cols, is_regression = _prepare_wf_data(config)

    # 2. Initialize loop state
    all_oof_preds: list[pl.DataFrame] = []
    gru_model = None
    gru_mean = None
    gru_std = None
    last_lgbm_model = None
    last_feature_cols: list[str] = []
    last_window_accuracy: float | None = None
    last_window_index = 0
    last_gru_history: list[dict] = []
    window_diagnostics: list[dict[str, Any]] = []
    stage_start = time.perf_counter()

    # 3. Process each walk-forward window
    for w_idx, window in enumerate(windows):
        window_start = time.perf_counter()
        console.rule(
            f"[bold cyan]Walk-forward window {w_idx + 1}/{len(windows)}[/]",
            style="cyan",
        )
        logger.info(
            "=== Window %d/%d: train=[%d:%d] test=[%d:%d] ===",
            w_idx + 1,
            len(windows),
            window.train_start_idx,
            window.train_end_idx,
            window.test_start_idx,
            window.test_end_idx,
        )

        result = _run_hybrid_window(
            config, w_idx, window, df, feature_cols, is_regression
        )
        if result is None:
            continue

        # Collect OOF predictions
        all_oof_preds.append(_collect_oof_predictions(result, is_regression))
        window_diagnostics.append(result["diag"])

        # Update deployable state (latest chronological window)
        gru_model = result["gru_model"]
        gru_mean = result["gru_mean"]
        gru_std = result["gru_std"]
        last_lgbm_model = result["model"]
        last_feature_cols = result["all_feature_cols"]
        last_window_accuracy = result["accuracy"]
        last_window_index = w_idx + 1
        last_gru_history = result["gru_history"]

        logger.info(
            "Window %d done (%.1fs)", w_idx + 1, time.perf_counter() - window_start
        )

    # 4. Validate and persist all artifacts
    _save_wf_artifacts(
        config,
        all_oof_preds,
        gru_model,
        gru_mean,
        gru_std,
        last_lgbm_model,
        last_feature_cols,
        last_window_accuracy,
        last_window_index,
        last_gru_history,
        windows,
        window_diagnostics,
        stage_start,
        is_regression,
    )


# Walk-forward training loop — static baseline


def _prepare_static_wf_data(
    config: Config,
) -> tuple[pl.DataFrame, list[Any], list[str], bool]:
    """Load labeled data, pre-compute regression target, and generate windows.

    Args:
        config: Application configuration.

    Returns:
        ``(df, windows, feature_cols, is_regression)`` — the full labeled
        DataFrame, walk-forward window objects, sorted feature column
        names, and a boolean indicating regression objective.
    """
    labels_path = Path(config.paths.labels)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels not found: {labels_path}")

    with console.status(f"[cyan]Loading labels[/] {labels_path}"):
        df = pl.read_parquet(labels_path)
    logger.info("Loaded labeled data for static baseline: %d rows", len(df))
    df, is_regression_static = _compute_regression_target(df, config)

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
    return df, windows, feature_cols, is_regression_static


def _train_and_predict_static_window(
    config: Config,
    w_idx: int,
    window: Any,
    df: pl.DataFrame,
    feature_cols: list[str],
    is_regression_static: bool,
    expanded_features: bool,
) -> dict[str, Any] | None:
    """Train LightGBM and generate predictions for a single static window.

    Returns a dict with ``oof_chunk``, ``model``, ``static_cols``,
    ``accuracy``, and ``diag``, or ``None`` if the window is too small.
    """
    from thesis.stage_4_training._lgbm import (
        _compute_class_weights,
        _train_fixed,
        _wrap_np,
    )

    train_df = df.slice(
        window.train_start_idx, window.train_end_idx - window.train_start_idx
    )
    test_df = df.slice(
        window.test_start_idx, window.test_end_idx - window.test_start_idx
    )
    if len(train_df) < _STATIC_MIN_TRAIN_ROWS or len(test_df) == 0:
        logger.warning("Static window %d too small; skipping", w_idx + 1)
        return None
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
        "Static baseline using %d features (%s mode)", len(static_cols), mode_tag
    )
    X_train = train_df.select(static_cols).to_numpy()
    X_test = test_df.select(static_cols).to_numpy()
    if is_regression_static:
        y_train = train_df["regression_target"].to_numpy().astype(np.float64)
        y_test = test_df["regression_target"].to_numpy().astype(np.float64)
        y_train_cls = train_df["label"].to_numpy().astype(np.int32)
        y_test_cls = test_df["label"].to_numpy().astype(np.int32)
    else:
        y_train = train_df["label"].to_numpy().astype(np.int32)
        y_test = test_df["label"].to_numpy().astype(np.int32)
        y_train_cls, y_test_cls = y_train, y_test
    sw = (
        train_df["sample_weight"].to_numpy().astype(np.float64)
        if "sample_weight" in train_df.columns
        else None
    )
    diag = _window_diagnostics(w_idx + 1, train_df, test_df, y_train_cls, y_test_cls)
    val_split_idx = max(1, int(len(X_train) * _VALIDATION_SPLIT_FRACTION))
    X_tr, y_tr = X_train[:-val_split_idx], y_train[:-val_split_idx]
    X_val, y_val = X_train[-val_split_idx:], y_train[-val_split_idx:]
    w_tr = sw[:-val_split_idx] if sw is not None else None
    class_weights = None if is_regression_static else _compute_class_weights(y_tr)
    diag["class_weights"] = (
        {str(k): v for k, v in class_weights.items()} if class_weights else None
    )
    diag["shift_weights_per_class"] = None  # static baseline: no shift weights
    model = _train_fixed(
        X_tr,
        y_tr,
        X_val,
        y_val,
        class_weights,
        config,
        static_cols,
        sample_weight=w_tr,
    )
    if is_regression_static:
        raw_preds = model.predict(_wrap_np(X_test, static_cols))
        preds = np.sign(raw_preds).astype(np.int32)  # threshold=0
        aligned_proba = np.zeros((len(raw_preds), 3), dtype=np.float64)
        aligned_proba[np.arange(len(preds)), preds + 1] = 1.0
        oof_chunk = pl.DataFrame(
            {
                "timestamp": test_df["timestamp"],
                "true_label": y_test_cls,
                "pred_label": preds,
                "pred_raw": raw_preds.astype(np.float64),
                **_one_hot_proba_columns(preds),
            }
        )
    else:
        proba = model.predict_proba(_wrap_np(X_test, static_cols))
        aligned_proba = _align_probability_matrix(proba, model.classes_)
        preds = _CLASS_ORDER[np.argmax(aligned_proba, axis=1)]
        oof_chunk = pl.DataFrame(
            {
                "timestamp": test_df["timestamp"],
                "true_label": y_test_cls,
                "pred_label": preds.astype(np.int32),
                **_probability_columns(proba, model.classes_),
            }
        )
    _add_prediction_diagnostics(diag, preds, y_test_cls, aligned_proba)
    acc = float((preds == y_test_cls).mean())
    logger.info(
        "Static window %d: accuracy=%.4f, test_samples=%d",
        w_idx + 1,
        acc,
        len(y_test_cls),
    )
    return {
        "oof_chunk": oof_chunk,
        "model": model,
        "static_cols": static_cols,
        "accuracy": acc,
        "diag": diag,
    }


def _save_static_wf_artifacts(
    config: Config,
    all_oof_preds: list[pl.DataFrame],
    last_lgbm_model: Any,
    last_feature_cols: list[str],
    last_window_accuracy: float | None,
    last_window_index: int,
    windows: list[Any],
    window_diagnostics: list[dict[str, Any]],
    stage_start: float,
) -> None:
    """Validate OOF predictions and persist static walk-forward artifacts.

    Args:
        config: Application configuration.
        all_oof_preds: List of per-window OOF Polars DataFrames.
        last_lgbm_model: LightGBM model from the last window.
        last_feature_cols: Feature column names.
        last_window_accuracy: Accuracy of the last window.
        last_window_index: 1-based index of the last window.
        windows: Walk-forward window objects.
        window_diagnostics: Per-window diagnostic dictionaries.
        stage_start: ``time.perf_counter()`` start timestamp.

    Raises:
        RuntimeError: If no predictions were generated.
        ValueError: If duplicate timestamps are found in OOF data.
    """
    import joblib

    from thesis.stage_4_training._lgbm import _save_feature_importance

    if not all_oof_preds or last_lgbm_model is None:
        raise RuntimeError("No static OOF predictions generated")

    oof_df = pl.concat(all_oof_preds)
    oof_df = _add_confidence_columns(oof_df)

    preds_path = Path(config.paths.predictions)
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    _validate_predictions(oof_df, preds_path)
    oof_df.write_parquet(preds_path)
    oof_df.write_csv(preds_path.with_suffix(".csv"))
    _write_prediction_manifest(
        oof_df,
        preds_path,
        windows_count=len(window_diagnostics),
    )

    model_path = Path(config.paths.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(last_lgbm_model, model_path)
    _save_feature_importance(last_lgbm_model, last_feature_cols, config)

    if config.paths.session_dir:
        models_dir = Path(config.paths.session_dir) / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        history_path = models_dir / "training_history.json"

        # Build per-window accuracy map from diagnostics
        per_window_accuracies: dict[str, float | None] = {}
        for d in window_diagnostics:
            key = str(d.get("window", ""))
            if key:
                per_window_accuracies[key] = d.get("accuracy")

        deployment_note = (
            f"Model saved from window {last_window_index}/{len(windows)} "
            "(the last chronological walk-forward window). "
            "This model has NOT seen any future data beyond its training window."
        )

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
                    "deployment_note": deployment_note,
                    "per_window_accuracies": per_window_accuracies,
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


def _run_walk_forward_static(
    config: Config, *, expanded_features: bool = False
) -> None:
    """Execute a static-feature-only walk-forward baseline.

    Isolates whether GRU hidden states add value. Uses event-time purged
    windows, LightGBM, sample weights, and OOF prediction output. When
    ``expanded_features`` is True, uses all available feature columns.

    Args:
        config: Application configuration.
        expanded_features: If True, use all available features rather
            than the whitelist.
    """
    # 1. Prepare data and windows
    df, windows, feature_cols, is_regression_static = _prepare_static_wf_data(config)

    # 2. Walk-forward loop
    all_oof_preds: list[pl.DataFrame] = []
    last_lgbm_model = None
    last_feature_cols: list[str] = []
    last_window_accuracy: float | None = None
    last_window_index = 0
    window_diagnostics: list[dict[str, Any]] = []
    stage_start = time.perf_counter()
    for w_idx, window in enumerate(windows):
        window_start = time.perf_counter()
        console.rule(
            f"[bold cyan]Static window {w_idx + 1}/{len(windows)}[/]",
            style="cyan",
        )
        logger.info(
            "=== Static window %d/%d: train=[%d:%d] test=[%d:%d] ===",
            w_idx + 1,
            len(windows),
            window.train_start_idx,
            window.train_end_idx,
            window.test_start_idx,
            window.test_end_idx,
        )
        result = _train_and_predict_static_window(
            config,
            w_idx,
            window,
            df,
            feature_cols,
            is_regression_static,
            expanded_features,
        )
        if result is None:
            continue
        all_oof_preds.append(result["oof_chunk"])
        window_diagnostics.append(result["diag"])
        last_lgbm_model = result["model"]
        last_feature_cols = result["static_cols"]
        last_window_accuracy = result["accuracy"]
        last_window_index = w_idx + 1
        logger.info(
            "Static window %d done (%.1fs)",
            w_idx + 1,
            time.perf_counter() - window_start,
        )

    # 3. Validate and persist
    _save_static_wf_artifacts(
        config,
        all_oof_preds,
        last_lgbm_model,
        last_feature_cols,
        last_window_accuracy,
        last_window_index,
        windows,
        window_diagnostics,
        stage_start,
    )


# Probability / label helpers


def _label_suffix(class_label: int) -> str:
    """Return the canonical probability-column suffix for a class label.

    Args:
        class_label: An integer from ``{-1, 0, 1}``.

    Returns:
        String suffix such as ``"minus1"`` or ``"0"``.
    """
    return f"minus{abs(class_label)}" if class_label < 0 else str(class_label)


def _one_hot_proba_columns(
    preds: np.ndarray,
    *,
    prefix: str = "pred_proba_class_",
) -> dict[str, np.ndarray]:
    """Build one-hot probability columns from predicted class labels.

    Used for regression mode where the model outputs a scalar rather than
    class probabilities.  Each class column is 1.0 where the prediction
    matches, 0.0 otherwise.

    Args:
        preds: Array of predicted class labels (``-1``, ``0``, or ``1``).
        prefix: Column name prefix (default ``"pred_proba_class_"``).

    Returns:
        Dictionary mapping canonical column names to 1-D one-hot arrays.
    """
    preds = np.asarray(preds, dtype=np.int32)
    return {
        f"{prefix}{_label_suffix(int(cls))}": (preds == cls).astype(np.float64)
        for cls in _CLASS_ORDER
    }


def _align_probability_matrix(
    proba: np.ndarray,
    class_order: list[int] | np.ndarray,
) -> np.ndarray:
    """Align class probabilities to the canonical ``[-1, 0, 1]`` order.

    Some LightGBM models produce probabilities in a different class order
    (e.g. ``[0, 1]`` for binary).  This function maps them to the fixed
    ``[-1, 0, 1]`` column order expected by downstream stages.

    Args:
        proba: Raw probability matrix ``(N, C)`` from the model.
        class_order: Model's ``classes_`` attribute (list or array).

    Returns:
        Probability matrix aligned to ``_CLASS_ORDER`` (``[-1, 0, 1]``).
    """
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
    """Build canonical probability columns for ``{-1, 0, 1}``.

    Args:
        proba: Raw probability matrix from the model.
        class_order: Model's ``classes_`` attribute.
        prefix: Column name prefix (default ``"pred_proba_class_"``).

    Returns:
        Dictionary mapping canonical column names to 1-D probability
        arrays.
    """
    aligned = _align_probability_matrix(proba, class_order)
    return {
        f"{prefix}{_label_suffix(int(cls))}": aligned[:, idx]
        for idx, cls in enumerate(_CLASS_ORDER)
    }


_PROBA_COLS = ("pred_proba_class_minus1", "pred_proba_class_0", "pred_proba_class_1")
"""Canonical probability column names in ``[-1, 0, 1]`` order."""


def _add_confidence_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Attach ``max_confidence`` and ``confidence_bin`` to an OOF DataFrame.

    Requires the three canonical probability columns to exist.  If they
    are absent (e.g. legacy outputs), returns the DataFrame unchanged.

    Args:
        df: OOF predictions DataFrame with ``pred_proba_class_*`` columns.

    Returns:
        DataFrame augmented with ``max_confidence`` (float64) and
        ``confidence_bin`` (string: ``"high"`` / ``"medium"`` / ``"low"``).
    """
    if not all(c in df.columns for c in _PROBA_COLS):
        return df
    return df.with_columns(
        pl.max_horizontal([pl.col(c) for c in _PROBA_COLS]).alias("max_confidence"),
    ).with_columns(
        pl.when(pl.col("max_confidence") >= 0.6)
        .then(pl.lit("high"))
        .when(pl.col("max_confidence") >= 0.4)
        .then(pl.lit("medium"))
        .otherwise(pl.lit("low"))
        .alias("confidence_bin"),
    )


# Public walk-forward entry points


def _run_walk_forward(config: Config) -> None:
    """Dispatch walk-forward training to the configured architecture.

    Args:
        config: Application configuration. Reads ``model.architecture``
            to route to ``_run_walk_forward_static`` or
            ``_run_walk_forward_hybrid``.

    Raises:
        ValueError: If ``model.architecture`` is unsupported.
    """
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

    Args:
        config: Application configuration.

    Static split does not apply purge or embargo at the split boundary. With
    triple-barrier labels, boundary labels may use future information from the
    adjacent split. For thesis evaluation, use sliding validation instead.
    """
    from thesis.stage_4_training._lgbm import train_model

    logger.warning(
        "Static split mode does not apply purge/embargo — potential label leakage "
        "at split boundaries. Recommended: validation.method = 'sliding'."
    )
    train_model(config)
