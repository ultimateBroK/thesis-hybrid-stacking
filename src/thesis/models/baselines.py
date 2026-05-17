"""Naive baselines. Sanity floor for model skill."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from thesis.reporting.metrics import accuracy, directional_accuracy, macro_f1


def naive_direction(y_returns: npt.NDArray) -> npt.NDArray:
    """Predict previous bar's direction. Persistence."""
    preds = np.zeros(len(y_returns), dtype=np.int8)
    preds[1:] = np.sign(y_returns[:-1]).astype(np.int8)
    return preds


def always_class(y_true: npt.NDArray, cls: int) -> npt.NDArray:
    """Predict one class. Bias check."""
    return np.full(len(y_true), cls, dtype=np.int8)


def majority_class(y_true: npt.NDArray) -> tuple[npt.NDArray, int]:
    """Predict majority class. Class imbalance floor."""
    vals, counts = np.unique(y_true, return_counts=True)
    maj = int(vals[np.argmax(counts)])
    return np.full(len(y_true), maj, dtype=np.int8), maj


def random_baseline(
    n: int,
    classes: list[int] | None = None,
    seed: int = 42,
) -> npt.NDArray:
    """Random labels. Seeded noise floor."""
    if classes is None:
        classes = [-1, 0, 1]
    rng = np.random.default_rng(seed)
    return rng.choice(classes, size=n).astype(np.int8)


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
    y_returns: npt.NDArray,
    seed: int = 42,
) -> dict[str, dict]:
    """Run all baselines. Compare model lift."""
    results: dict[str, dict] = {}

    results["naive_direction"] = compute_metrics(y_true, naive_direction(y_returns))
    results["always_long"] = compute_metrics(y_true, always_class(y_true, 1))
    results["always_short"] = compute_metrics(y_true, always_class(y_true, -1))
    results["always_hold"] = compute_metrics(y_true, always_class(y_true, 0))

    maj_pred, maj_cls = majority_class(y_true)
    results["majority_class"] = compute_metrics(y_true, maj_pred)
    results["majority_class"]["majority_class_label"] = maj_cls

    results["random"] = compute_metrics(y_true, random_baseline(len(y_true), seed=seed))

    return results
