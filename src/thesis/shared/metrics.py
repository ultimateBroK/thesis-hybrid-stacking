"""Shared classification metric functions.

Canonical source for accuracy, macro_f1, directional_accuracy.
Avoids circular imports between stage_4 baselines and stage_6 reporting.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def accuracy(y_true: npt.NDArray, y_pred: npt.NDArray) -> float:
    """Fraction of correct predictions."""
    return float((y_true == y_pred).mean())


def macro_f1(
    y_true: npt.NDArray, y_pred: npt.NDArray, classes: list[int] | None = None
) -> float:
    """Macro-averaged F1 score."""

    def _prf(yt, yp, cls):
        true_mask = yt == cls
        pred_mask = yp == cls
        rec = float((yp[true_mask] == cls).mean()) if true_mask.sum() > 0 else 0.0
        prec = float((yt[pred_mask] == cls).mean()) if pred_mask.sum() > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return prec, rec, f1

    if classes is None:
        classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    f1s = [_prf(y_true, y_pred, c)[2] for c in classes]
    return float(np.mean(f1s))


def directional_accuracy(y_true: npt.NDArray, y_pred: npt.NDArray) -> float:
    """Accuracy on bars where both true and predicted labels are non-zero.

    Hold-vs-direction mismatches excluded rather than counted as wrong.
    """
    mask = (y_true != 0) & (y_pred != 0)
    if mask.sum() == 0:
        return 0.0
    return float((y_true[mask] == y_pred[mask]).mean())
