"""Prediction validation and probability column helpers for walk-forward training."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import polars as pl

logger = logging.getLogger("thesis.pipeline")

_CLASS_ORDER = np.array([-1, 0, 1], dtype=np.int32)


def _validate_predictions(df: pl.DataFrame, path: Path) -> None:
    """Validate final OOF predictions before writing the parquet artifact."""
    required = {"timestamp", "pred_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Predictions missing columns {sorted(missing)}: file={path}")
    if df.is_empty():
        raise ValueError(f"Predictions are empty: file={path}")

    ts_col = df["timestamp"]
    if ts_col.null_count() > 0:
        raise ValueError(
            f"Predictions timestamp has nulls:"
            f" actual={ts_col.null_count()}, file={path}"
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
            f"Invalid pred_label values: expected={{-1,0,1}},"
            f" actual={invalid}, file={path}"
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
    """Write compact diagnostics beside final_predictions.csv."""
    from thesis.stage_4_training.walk_forward.diagnostics import _counts_dict

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


def _label_suffix(class_label: int) -> str:
    """Return canonical probability-column suffix for a class label."""
    return f"minus{abs(class_label)}" if class_label < 0 else str(class_label)


def _one_hot_proba_columns(
    preds: np.ndarray,
    *,
    prefix: str = "pred_proba_class_",
) -> dict[str, np.ndarray]:
    """Build one-hot probability columns from predicted class labels."""
    preds = np.asarray(preds, dtype=np.int32)
    return {
        f"{prefix}{_label_suffix(int(cls))}": (preds == cls).astype(np.float64)
        for cls in _CLASS_ORDER
    }


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


_PROBA_COLS = ("pred_proba_class_minus1", "pred_proba_class_0", "pred_proba_class_1")
"""Canonical probability column names in ``[-1, 0, 1]`` order."""


def _add_confidence_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Attach ``max_confidence`` and ``confidence_bin`` to an OOF DataFrame."""
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
