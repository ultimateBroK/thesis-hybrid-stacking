"""Core classification metrics for Stage 3 model experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
from sklearn.metrics import precision_recall_fscore_support

from thesis.models.artifacts import proba_columns
from thesis.models.estimators import CLASS_ORDER, align_proba
from thesis.reporting.metrics import accuracy, directional_accuracy, macro_f1

__all__ = [
    "classification_metrics",
    "confusion_matrix",
    "model_comparison_table",
    "one_hot_proba",
    "per_class_metrics",
    "proba_columns",
]


def _label_suffix(cls: int) -> str:
    return f"minus{abs(cls)}" if cls < 0 else str(cls)


def one_hot_proba(
    preds: npt.NDArray,
    *,
    prefix: str = "pred_proba_class_",
) -> dict[str, npt.NDArray]:
    """Return one-hot probability columns for hard labels."""
    preds = np.asarray(preds, dtype=np.int32)
    return {
        f"{prefix}{_label_suffix(int(c))}": (preds == c).astype(np.float64)
        for c in CLASS_ORDER
    }


def confusion_matrix(
    y_true: npt.NDArray,
    y_pred: npt.NDArray,
    labels: npt.NDArray = CLASS_ORDER,
) -> dict[str, dict[str, int]]:
    """Return label-indexed confusion matrix."""
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    return {
        str(int(actual)): {
            str(int(pred)): int(((y_true == actual) & (y_pred == pred)).sum())
            for pred in labels
        }
        for actual in labels
    }


def per_class_metrics(
    y_true: npt.NDArray,
    y_pred: npt.NDArray,
    labels: npt.NDArray = CLASS_ORDER,
) -> dict[str, dict[str, float]]:
    """Return precision, recall, F1, and support for each class."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    return {
        str(int(cls)): {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, cls in enumerate(labels)
    }


def classification_metrics(
    y_true: npt.NDArray,
    y_pred: npt.NDArray,
) -> dict[str, Any]:
    """Return Stage 3 core classification metrics."""
    return {
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
        "per_class": per_class_metrics(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def model_comparison_table(metrics: dict[str, dict[str, Any]]) -> pl.DataFrame:
    """Build report-ready model comparison table."""
    rows = [
        {
            "model": name,
            "accuracy": values.get("accuracy"),
            "macro_f1": values.get("macro_f1"),
            "directional_accuracy": values.get("directional_accuracy"),
        }
        for name, values in metrics.items()
    ]
    return pl.DataFrame(rows).sort("model") if rows else pl.DataFrame()


_align_proba = align_proba
