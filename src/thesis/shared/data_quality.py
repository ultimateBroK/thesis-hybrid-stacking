"""Shared data-quality checks for OHLCV DataFrames.

Pure functions returning dicts — no logging, no I/O.  Reused by
stage_1 (data preparation) and stage_6 (reporting).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
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


DEFAULT_GOLD_CALENDARS: tuple[str, ...] = (
    "CME Globex Gold and Silver Futures",
    "CME Globex Commodities",
    "CME_FX",
)


@dataclass(frozen=True)
class GapClassification:
    """Classification summary for timestamp gaps."""

    calendar_gap_count: int
    real_gap_count: int
    estimated_missing_bars: int
    largest_gap_bars: int
    warnings: list[str]


def resolve_market_calendar(name: str | None = None):
    """Resolve configured market calendar, defaulting to a gold session."""
    candidates = [name] if name else list(DEFAULT_GOLD_CALENDARS)
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return mcal.get_calendar(candidate)
        except RuntimeError:
            continue
    raise RuntimeError(
        "Could not resolve market calendar for gold bars. "
        f"Tried: {candidates or list(DEFAULT_GOLD_CALENDARS)}"
    )


def classify_calendar_gaps(
    df: pl.DataFrame,
    timeframe_ms: int,
    *,
    calendar_name: str | None = None,
) -> GapClassification:
    """Classify gaps into calendar-expected closures and real missing bars."""
    if "timestamp" not in df.columns or len(df) < 2:
        return GapClassification(0, 0, 0, 0, [])
    try:
        ts = df["timestamp"].sort().to_list()
        actual_index = pd.DatetimeIndex(ts)
        if actual_index.tz is None:
            actual_index = actual_index.tz_localize("UTC")
        else:
            actual_index = actual_index.tz_convert("UTC")

        cal = resolve_market_calendar(calendar_name)
        start = actual_index.min().date()
        end = actual_index.max().date()
        schedule = cal.schedule(start_date=start, end_date=end)

        captured_warnings: list[str] = []
        with warnings.catch_warnings(record=True) as warns:
            warnings.simplefilter("always")
            expected_index = mcal.date_range(schedule, frequency=f"{timeframe_ms}ms")
            for warn in warns:
                captured_warnings.append(str(warn.message))

        expected_set = set(expected_index)
        calendar_gap_count = 0
        real_gap_count = 0
        estimated_missing_bars = 0
        largest_gap_bars = 0

        for prev_ts, curr_ts in zip(actual_index[:-1], actual_index[1:]):
            delta_ms = int((curr_ts - prev_ts).total_seconds() * 1000)
            if delta_ms <= timeframe_ms:
                continue
            gap_bars = delta_ms // timeframe_ms
            largest_gap_bars = max(largest_gap_bars, gap_bars)
            interior = pd.date_range(
                start=prev_ts + pd.Timedelta(milliseconds=timeframe_ms),
                end=curr_ts - pd.Timedelta(milliseconds=timeframe_ms),
                freq=pd.Timedelta(milliseconds=timeframe_ms),
                tz="UTC",
            )
            expected_missing = sum(1 for t in interior if t in expected_set)
            if expected_missing > 0:
                real_gap_count += 1
                estimated_missing_bars += expected_missing
            else:
                calendar_gap_count += 1

        return GapClassification(
            calendar_gap_count=calendar_gap_count,
            real_gap_count=real_gap_count,
            estimated_missing_bars=estimated_missing_bars,
            largest_gap_bars=largest_gap_bars,
            warnings=captured_warnings,
        )
    except Exception as exc:  # pragma: no cover - fallback for calendar edge cases
        ts_col = df["timestamp"].sort()
        deltas = ts_col.diff().drop_nulls().dt.total_milliseconds().to_list()
        calendar_gap_count = 0
        real_gap_count = 0
        missing = 0
        largest = 0
        ts_values = ts_col.to_list()
        for i, delta in enumerate(deltas):
            if delta <= timeframe_ms:
                continue
            gap_bars = int(delta // timeframe_ms)
            largest = max(largest, gap_bars)
            start = ts_values[i]
            end = ts_values[i + 1]
            if start.weekday() >= 5 or end.weekday() >= 5:
                calendar_gap_count += 1
            else:
                real_gap_count += 1
                missing += max(0, gap_bars - 1)
        return GapClassification(
            calendar_gap_count=calendar_gap_count,
            real_gap_count=real_gap_count,
            estimated_missing_bars=missing,
            largest_gap_bars=largest,
            warnings=[f"calendar_fallback:{type(exc).__name__}:{exc}"],
        )
