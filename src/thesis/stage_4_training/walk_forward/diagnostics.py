"""Window diagnostic helpers for walk-forward training."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import polars as pl

logger = logging.getLogger("thesis.pipeline")

_HIGH_CONFIDENCE_THRESHOLD = 0.70
_SHORT_BIAS_RATIO_THRESHOLD = 0.5


def _counts_dict(values: np.ndarray) -> dict[str, int]:
    """Return class/count dict with string keys for JSON."""
    if values.size == 0:
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
    if df.is_empty() or "timestamp" not in df.columns:
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


def _compute_per_class_metrics(
    preds: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Compute per-class precision, recall, F1, and support from predictions."""
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
    """Attach prediction distribution, confidence, and per-class metrics to *diag*."""
    pred_counts = _counts_dict(preds)
    confidence = np.max(proba, axis=1) if len(proba) else np.array([], dtype=float)

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
