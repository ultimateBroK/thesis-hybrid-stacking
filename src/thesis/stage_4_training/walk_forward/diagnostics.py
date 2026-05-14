"""Per-window diagnostics: label distributions, prediction metrics, entropy."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import precision_recall_fscore_support

logger = logging.getLogger("thesis")


def _shannon_entropy(p: np.ndarray) -> float:
    """Shannon entropy (base-2) of a probability distribution."""
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


def _counts(values: np.ndarray) -> dict[str, int]:
    """Class → count dict with string keys. For JSON serialization."""
    if values.size == 0:
        return {}
    labels, counts = np.unique(values.astype(np.int32), return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(labels, counts)}


def _pct(counts: dict[str, int]) -> dict[str, float]:
    """Convert counts to percentages."""
    total = sum(counts.values())
    if total == 0:
        return {}
    return {k: round(v / total * 100.0, 2) for k, v in counts.items()}


def _dates(df: pl.DataFrame) -> dict[str, str]:
    """Start/end timestamps for a window slice."""
    if df.is_empty() or "timestamp" not in df.columns:
        return {"start": "", "end": ""}
    return {"start": str(df["timestamp"][0]), "end": str(df["timestamp"][-1])}


def _window_diagnostics(
    w_idx: int,
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Build label distribution diagnostics for one window."""
    train_c = _counts(y_train)
    test_c = _counts(y_test)
    diag: dict[str, Any] = {
        "window": w_idx,
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "train_dates": _dates(train_df),
        "test_dates": _dates(test_df),
        "train_label_counts": train_c,
        "train_label_pct": _pct(train_c),
        "test_label_counts": test_c,
        "test_label_pct": _pct(test_c),
    }
    logger.info(
        "  Window %d labels | train=%s | test=%s",
        w_idx,
        diag["train_label_pct"],
        diag["test_label_pct"],
    )
    return diag


def _per_class_metrics(
    preds: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Per-class precision, recall, F1, support."""
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
    *,
    confidence_threshold: float = 0.0,
) -> None:
    """Attach prediction distribution, confidence, L/S ratio, entropy to diag."""
    pred_c = _counts(preds)
    confidence = np.max(proba, axis=1) if len(proba) else np.array([])
    n_total = sum(pred_c.values())
    n_long = pred_c.get("1", 0)
    n_short = pred_c.get("-1", 0)
    n_hold = pred_c.get("0", 0)

    # Prediction distribution entropy
    if n_total > 0:
        pred_probs = np.array([pred_c[k] / n_total for k in sorted(pred_c)])
        pred_entropy = round(_shannon_entropy(pred_probs), 4)
    else:
        pred_entropy = None

    # Mean per-sample entropy (model uncertainty across classes)
    if len(proba):
        sample_entropies = np.array(
            [_shannon_entropy(proba[i]) for i in range(len(proba))]
        )
        mean_sample_entropy = round(float(sample_entropies.mean()), 4)
    else:
        mean_sample_entropy = None

    # L/S ratio guardrail: flag if outside [0.2, 5.0]
    ls_ratio = n_long / n_short if n_short > 0 else float("inf")
    ls_flagged = False
    if n_short > 0 and n_long > 0:
        ls_flagged = ls_ratio < 0.2 or ls_ratio > 5.0

    diag.update(
        {
            "prediction_counts": pred_c,
            "prediction_pct": _pct(pred_c),
            "accuracy": float((preds == y_test).mean()) if len(y_test) else None,
            "mean_confidence": float(confidence.mean()) if len(confidence) else None,
            "high_conf_70_pct": (
                float((confidence >= 0.70).mean() * 100.0) if len(confidence) else None
            ),
            "ls_ratio": round(ls_ratio, 4) if n_short > 0 else None,
            "ls_ratio_flagged": ls_flagged,
            "prediction_entropy": pred_entropy,
            "mean_sample_entropy": mean_sample_entropy,
            "confidence_threshold": confidence_threshold,
            "hold_count": n_hold,
            "hold_pct": round(n_hold / n_total * 100.0, 2) if n_total else None,
            "per_class": _per_class_metrics(preds, y_test) if len(y_test) else {},
        }
    )

    logger.info(
        "  Window %d preds | %s | acc=%.4f conf=%.3f L/S=%.3f hold=%d/%d (%.1f%%)",
        diag["window"],
        diag["prediction_pct"],
        diag["accuracy"] or 0.0,
        diag["mean_confidence"] or 0.0,
        ls_ratio if n_short > 0 else float("nan"),
        n_hold,
        n_total,
        diag["hold_pct"] or 0.0,
    )

    # Guardrail warnings
    if n_short == 0 and n_long > 0:
        logger.warning("  Window %d: No SHORT predictions", diag["window"])
    elif n_long == 0 and n_short > 0:
        logger.warning("  Window %d: No LONG predictions", diag["window"])
    elif ls_flagged:
        logger.warning(
            "  Window %d: L/S ratio %.2f outside [0.2, 5.0]",
            diag["window"],
            ls_ratio,
        )
