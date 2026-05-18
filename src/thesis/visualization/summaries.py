"""Prediction summary computation shared by dashboard and reporting."""

from __future__ import annotations

import numpy as np
import polars as pl

from thesis.reporting.metrics import (
    DEFAULT_CLASSES,
    accuracy,
    directional_accuracy,
    macro_f1,
    precision_recall_f1_per_class,
)


def _extract_prediction_arrays(
    preds: pl.DataFrame | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    required = {"true_label", "pred_label"}
    if preds is None or preds.is_empty() or not required.issubset(set(preds.columns)):
        return None
    return preds["true_label"].to_numpy(), preds["pred_label"].to_numpy()


def _find_best_base_model(rows: list[dict]) -> str:
    valid_rows = [r for r in rows if r.get("macro_f1") is not None]
    if not valid_rows:
        return "N/A"
    best = max(valid_rows, key=lambda r: float(r.get("macro_f1") or 0))
    return str(best.get("model", "N/A"))


_CLASS_ID: dict[str, int] = {"Short": -1, "Hold": 0, "Long": 1}


def compute_prediction_summary(data: dict) -> dict[str, object]:
    """Derive ML summary metrics from final predictions."""
    arrays = _extract_prediction_arrays(data.get("predictions"))
    if arrays is None:
        return {}

    y_true, y_pred = arrays
    per_class_raw = precision_recall_f1_per_class(y_true, y_pred)
    per_class: dict[str, dict[str, float | int]] = {}
    for name, metrics in per_class_raw.items():
        cls = _CLASS_ID[name]
        per_class[name] = {
            "true_count": int((y_true == cls).sum()),
            "pred_count": int((y_pred == cls).sum()),
            **metrics,
        }

    return {
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred, DEFAULT_CLASSES),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
        "total_predictions": len(y_true),
        "per_class": per_class,
        "y_true": y_true,
        "y_pred": y_pred,
        "best_base_model": _find_best_base_model(data.get("model_comparison", [])),
    }


__all__ = ["compute_prediction_summary"]
