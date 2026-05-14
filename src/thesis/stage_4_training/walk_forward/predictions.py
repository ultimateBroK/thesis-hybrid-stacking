"""Prediction helpers. Align probabilities, gate confidence, validate OOF."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import polars as pl

logger = logging.getLogger("thesis")

_CLASS_ORDER = np.array([-1, 0, 1], dtype=np.int32)


def _apply_confidence_threshold(proba: np.ndarray, threshold: float) -> np.ndarray:
    """Gate LONG/SHORT by confidence.

    When threshold > 0:
        LONG  (1) if P(LONG)  - P(SHORT) > threshold
        SHORT (-1) if P(SHORT) - P(LONG)  > threshold
        HOLD  (0) otherwise
    When threshold == 0: standard argmax.
    """
    if threshold <= 0:
        return _CLASS_ORDER[np.argmax(proba, axis=1)]

    diff = proba[:, 2] - proba[:, 0]  # Long edge over short
    return np.where(diff > threshold, 1, np.where(diff < -threshold, -1, 0)).astype(
        np.int32
    )


def _align_proba(proba: np.ndarray, class_order: list[int] | np.ndarray) -> np.ndarray:
    """Align probabilities to [-1, 0, 1]."""
    aligned = np.zeros((len(proba), 3), dtype=np.float64)
    index_map = {int(c): i for i, c in enumerate(class_order)}
    for target_idx, cls in enumerate(_CLASS_ORDER):
        src = index_map.get(int(cls))
        if src is not None:
            aligned[:, target_idx] = proba[:, src]
    return aligned


def proba_columns(
    proba: np.ndarray,
    class_order: list[int] | np.ndarray,
) -> dict[str, np.ndarray]:
    """Probability columns in canonical order."""
    aligned = _align_proba(proba, class_order)
    return {
        f"pred_proba_class_{'minus' + str(abs(c)) if c < 0 else str(c)}": aligned[:, i]
        for i, c in enumerate(_CLASS_ORDER)
    }


def _label_suffix(cls: int) -> str:
    """Probability column suffix."""
    return f"minus{abs(cls)}" if cls < 0 else str(cls)


def one_hot_proba(
    preds: np.ndarray,
    *,
    prefix: str = "pred_proba_class_",
) -> dict[str, np.ndarray]:
    """One-hot probabilities from labels."""
    preds = np.asarray(preds, dtype=np.int32)
    return {
        f"{prefix}{_label_suffix(int(c))}": (preds == c).astype(np.float64)
        for c in _CLASS_ORDER
    }


def _validate_predictions(df: pl.DataFrame, path: Path) -> None:
    """Validate OOF before write.

    Catch missing columns, nulls, duplicate time, bad labels.
    """
    required = {"timestamp", "pred_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Predictions missing columns {sorted(missing)}: {path}")
    if df.is_empty():
        raise ValueError(f"Predictions are empty: {path}")

    ts = df["timestamp"]
    if ts.null_count() > 0:
        raise ValueError(f"Timestamp has nulls ({ts.null_count()}): {path}")
    if ts.n_unique() < len(ts):
        raise ValueError(f"OOF predictions contain duplicate timestamps: {path}")
    if ts.to_list() != sorted(ts.to_list()):
        raise ValueError(f"OOF predictions not sorted by timestamp: {path}")

    pred = df["pred_label"]
    if pred.null_count() > 0:
        raise ValueError(f"pred_label has nulls: {path}")
    invalid = sorted(set(pred.unique().to_list()) - {-1, 0, 1})
    if invalid:
        raise ValueError(
            f"Invalid pred_label values: expected {{-1,0,1}}, got {invalid}: {path}"
        )

    null_cols = {c: df[c].null_count() for c in df.columns if df[c].null_count()}
    if null_cols:
        raise ValueError(f"Predictions contain nulls: {null_cols}: {path}")


def _write_prediction_manifest(
    df: pl.DataFrame,
    path: Path,
    *,
    windows_count: int,
) -> None:
    """Write prediction manifest beside CSV."""
    from thesis.stage_4_training.walk_forward.diagnostics import _counts

    manifest = {
        "row_count": len(df),
        "start": str(df["timestamp"][0]),
        "end": str(df["timestamp"][-1]),
        "label_distribution": (
            _counts(df["true_label"].to_numpy()) if "true_label" in df.columns else {}
        ),
        "prediction_distribution": _counts(df["pred_label"].to_numpy()),
        "mean_confidence": (
            float(df["max_confidence"].mean())
            if "max_confidence" in df.columns
            else None
        ),
        "windows_count": windows_count,
    }
    with open(path.with_name("prediction_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


_PROBA_COLS = ("pred_proba_class_minus1", "pred_proba_class_0", "pred_proba_class_1")


def _add_confidence_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Add confidence score and bin."""
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
