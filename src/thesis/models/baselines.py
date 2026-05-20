"""Majority-class baseline for model skill checks."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from thesis.reporting.metrics import accuracy, directional_accuracy, macro_f1


def majority_label(y: npt.NDArray) -> int:
    """Return the most frequent class label."""
    vals, counts = np.unique(y, return_counts=True)
    return int(vals[np.argmax(counts)])


def predict_majority_label(
    y_train: npt.NDArray, n_rows: int
) -> tuple[npt.NDArray, int]:
    """Predict majority class from train labels for n_rows test rows."""
    label = majority_label(y_train)
    return np.full(n_rows, label, dtype=np.int32), label


def _per_class_f1(y_true: npt.NDArray, y_pred: npt.NDArray) -> dict[str, float]:
    """Per-class F1 for label -1 (Short) and +1 (Long)."""
    result: dict[str, float] = {}
    for cls, key in [(-1, "short_f1"), (1, "long_f1")]:
        tp = int(((y_pred == cls) & (y_true == cls)).sum())
        fp = int(((y_pred == cls) & (y_true != cls)).sum())
        fn = int(((y_pred != cls) & (y_true == cls)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        result[key] = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return result


def compute_metrics(y_true: npt.NDArray, y_pred: npt.NDArray) -> dict[str, float]:
    """Core classification metrics."""
    m: dict[str, float] = {
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
    }
    m.update(_per_class_f1(y_true, y_pred))
    return m


def run_all(
    y_true: npt.NDArray,
    y_returns: npt.NDArray | None = None,
    seed: int | None = None,
) -> dict[str, dict]:
    """Run thesis baseline: majority class only."""
    maj_pred, maj_cls = predict_majority_label(y_true, len(y_true))
    metrics = compute_metrics(y_true, maj_pred)
    metrics["majority_class_label"] = maj_cls
    return {"majority_class": metrics}
