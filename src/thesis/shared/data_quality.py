"""Shared data-quality checks for OHLCV DataFrames.

Pure functions returning dicts — no logging, no I/O.  Reused by
stage_1 (data preparation) and stage_6 (reporting).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl


def check_ohlcv_consistency(df: pl.DataFrame) -> dict[str, Any]:
    """Check that OHLC relationships hold for every row.

    * high >= open, close, low
    * low  <= open, close, high
    * all prices > 0
    """
    total = len(df)
    ohlc_violations = 0
    price_negative = 0

    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            price_negative += int((df[col] <= 0).sum())

    if all(c in df.columns for c in ("open", "high", "low", "close")):
        ohlc_violations += int((df["high"] < df["open"]).sum())
        ohlc_violations += int((df["high"] < df["close"]).sum())
        ohlc_violations += int((df["high"] < df["low"]).sum())
        ohlc_violations += int((df["low"] > df["open"]).sum())
        ohlc_violations += int((df["low"] > df["close"]).sum())
        ohlc_violations += int((df["low"] > df["high"]).sum())

    return {
        "total_rows": total,
        "ohlc_violations": ohlc_violations,
        "price_negative_count": price_negative,
        "is_consistent": ohlc_violations == 0 and price_negative == 0,
    }


def check_gap_report(df: pl.DataFrame, timeframe_ms: int) -> dict[str, Any]:
    """Timestamp continuity: find gaps > *timeframe_ms*, count duplicates.

    Requires a ``timestamp`` column.
    """
    if "timestamp" not in df.columns or len(df) < 2:
        return {
            "gap_count": 0,
            "estimated_missing_bars": 0,
            "largest_gap_bars": 0,
            "duplicate_count": 0,
        }

    diffs = (
        df.select(
            (pl.col("timestamp").diff().dt.total_milliseconds()).alias("delta_ms"),
        )
        .drop_nulls()
        .get_column("delta_ms")
    )

    missing_gaps = diffs.filter(diffs > timeframe_ms)

    estimated_missing = int(
        ((missing_gaps / timeframe_ms).floor() - 1).sum() or 0,
    )
    largest_gap_bars = int(diffs.max() / timeframe_ms) if diffs.max() else 0

    ts_col = df["timestamp"]
    duplicate_count = len(df) - ts_col.n_unique()

    return {
        "gap_count": len(missing_gaps),
        "estimated_missing_bars": estimated_missing,
        "largest_gap_bars": largest_gap_bars,
        "duplicate_count": int(duplicate_count),
    }


def check_outlier_returns(
    df: pl.DataFrame,
    z_threshold: float = 5.0,
) -> dict[str, Any]:
    """Flag log-returns exceeding *z_threshold* standard deviations."""
    if "close" not in df.columns or len(df) < 2:
        return {
            "outlier_count": 0,
            "z_threshold": z_threshold,
            "max_zscore": 0.0,
            "outlier_indices": [],
        }

    close = df["close"].cast(pl.Float64).to_numpy()
    log_returns = np.diff(np.log(close))
    mean_r = float(np.mean(log_returns))
    std_r = float(np.std(log_returns))

    if std_r == 0:
        return {
            "outlier_count": 0,
            "z_threshold": z_threshold,
            "max_zscore": 0.0,
            "outlier_indices": [],
        }

    z_scores = np.abs((log_returns - mean_r) / std_r)
    outlier_mask = z_scores > z_threshold
    outlier_count = int(outlier_mask.sum())

    # Return first 20 outlier indices (row index in the DataFrame)
    all_indices = np.where(outlier_mask)[0] + 1  # +1 because diff shifts by 1
    outlier_indices: list[int] = all_indices[:20].tolist()

    return {
        "outlier_count": outlier_count,
        "z_threshold": z_threshold,
        "max_zscore": float(np.max(z_scores)),
        "outlier_indices": outlier_indices,
    }


def check_candle_quality(df: pl.DataFrame) -> dict[str, Any]:
    """Check high >= low, open/close inside [low, high], volume >= 0."""
    if df.is_empty():
        return {"invalid_count": 0, "total_rows": 0, "is_valid": True}

    conditions = [
        pl.col("high") >= pl.col("low"),
    ]

    if "open" in df.columns:
        conditions.append(pl.col("open") >= pl.col("low"))
        conditions.append(pl.col("open") <= pl.col("high"))

    if "close" in df.columns:
        conditions.append(pl.col("close") >= pl.col("low"))
        conditions.append(pl.col("close") <= pl.col("high"))

    if "volume" in df.columns:
        conditions.append(pl.col("volume") >= 0)

    valid = df.filter(
        pl.all_horizontal(conditions),
    )
    invalid_count = len(df) - len(valid)

    return {
        "invalid_count": invalid_count,
        "total_rows": len(df),
        "is_valid": invalid_count == 0,
    }
