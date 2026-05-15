"""Probability calibration metrics: ECE, Brier, log-loss, confidence bins."""

from __future__ import annotations

import numpy as np

from thesis.stage_6_reporting.model_metrics import high_confidence_accuracy


def to_onehot(y_true: np.ndarray, classes: list[int]) -> np.ndarray:
    """Convert labels to one-hot encoding."""
    k = len(classes)
    idx_map = {c: i for i, c in enumerate(classes)}
    indices = np.array([idx_map[int(y)] for y in y_true], dtype=int)
    return np.eye(k)[indices]


def expected_calibration_error(
    y_true_onehot: np.ndarray, y_proba: np.ndarray, n_bins: int = 10
) -> float:
    """Expected Calibration Error (ECE)."""
    confidences = np.max(y_proba, axis=1)
    correct = (np.argmax(y_proba, axis=1) == np.argmax(y_true_onehot, axis=1)).astype(
        float
    )
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        count = mask.sum()
        if count == 0:
            continue
        ece += count * np.abs(confidences[mask].mean() - correct[mask].mean())
    return float(ece / len(y_true_onehot))


def brier_score(y_true_onehot: np.ndarray, y_proba: np.ndarray) -> float:
    """Brier score for probability calibration."""
    return float(np.mean((y_true_onehot - y_proba) ** 2))


def log_loss(
    y_true: np.ndarray, y_proba: np.ndarray, classes: list[int] | None = None
) -> float:
    """Log loss (cross-entropy) for multi-class classification."""
    if classes is None:
        classes = [-1, 0, 1]
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    eps = 1e-15
    clipped = np.clip(y_proba, eps, 1.0 - eps)
    normalized = clipped / clipped.sum(axis=1, keepdims=True)
    indices = np.array([class_to_idx[int(y)] for y in y_true], dtype=int)
    return float(-np.mean(np.log(normalized[np.arange(len(y_true)), indices])))


def confidence_bins_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    bins: list[float] | None = None,
) -> list[dict]:
    """Accuracy per confidence bin."""
    if bins is None:
        bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    confidences = np.max(y_proba, axis=1)
    results: list[dict] = []
    lo = 0.0
    for hi in bins:
        mask = (confidences > lo) & (confidences <= hi)
        count = int(mask.sum())
        acc = float((y_true[mask] == y_pred[mask]).mean()) if count > 0 else 0.0
        results.append(
            {
                "lo": round(lo, 2),
                "hi": round(hi, 2),
                "count": count,
                "accuracy": round(acc, 4),
            }
        )
        lo = hi
    return results


def calibration_reliability_data(
    y_true_onehot: np.ndarray, y_proba: np.ndarray, n_bins: int = 10
) -> dict:
    """Reliability data for calibration curve."""
    confidences = np.max(y_proba, axis=1)
    correct_mask = (
        np.argmax(y_proba, axis=1) == np.argmax(y_true_onehot, axis=1)
    ).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers, accuracies, counts = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        c = int(mask.sum())
        counts.append(c)
        centers.append(round((lo + hi) / 2, 3))
        accuracies.append(round(float(correct_mask[mask].mean()), 4) if c > 0 else 0.0)
    return {"bin_centers": centers, "accuracies": accuracies, "counts": counts}


def compute_all_calibration_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: list[int] | None = None,
) -> dict:
    """Compute all calibration metrics."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_proba = np.asarray(y_proba, dtype=np.float64)
    if classes is None:
        classes = [-1, 0, 1]

    y_true_onehot = to_onehot(y_true, classes)

    return {
        "ece": round(expected_calibration_error(y_true_onehot, y_proba), 6),
        "brier_score": round(brier_score(y_true_onehot, y_proba), 6),
        "log_loss": round(log_loss(y_true, y_proba, classes=classes), 6),
        "high_confidence_accuracy": high_confidence_accuracy(
            y_true, y_pred, y_proba, threshold=0.70
        ),
        "confidence_bins": confidence_bins_accuracy(y_true, y_pred, y_proba),
        "reliability_data": calibration_reliability_data(y_true_onehot, y_proba),
    }
