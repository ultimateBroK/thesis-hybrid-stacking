"""Data preparation helpers for chart builders."""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

logger = logging.getLogger("thesis.visualization")


def downsample_ohlcv(df: pl.DataFrame, max_bars: int) -> pl.DataFrame:
    """Aggregate OHLCV into max_bars rows via stride grouping."""
    stride = max(1, len(df) // max_bars)
    group_col = pl.int_range(0, len(df)) // stride
    agg_exprs = [
        pl.col("timestamp").first(),
        pl.col("open").first(),
        pl.col("high").max(),
        pl.col("low").min(),
        pl.col("close").last(),
    ]
    if "volume" in df.columns:
        agg_exprs.append(pl.col("volume").sum())
    return (
        df.with_columns(group_col.alias("_group"))
        .group_by("_group", maintain_order=True)
        .agg(*agg_exprs)
        .drop("_group")
    )


def parse_chart_timestamps(ts_col: pl.Series) -> list[str]:
    """Format timestamps for chart x-axis labels."""
    if ts_col.dtype == pl.Utf8:
        ts_col = ts_col.str.to_datetime()
    if ts_col.dtype.is_temporal():
        has_intraday = (ts_col.dt.hour().sum() + ts_col.dt.minute().sum()) > 0
        fmt = "%Y-%m-%d %H:%M" if has_intraday else "%Y-%m-%d"
        return ts_col.dt.strftime(fmt).to_list()
    return ts_col.cast(pl.Utf8).to_list()


def make_kline_series(df: pl.DataFrame) -> list[list[float]]:
    """Extract OHLC as [open, close, low, high] per bar."""
    opens = df["open"].to_numpy().astype(float)
    closes = df["close"].to_numpy().astype(float)
    lows = df["low"].to_numpy().astype(float)
    highs = df["high"].to_numpy().astype(float)
    return [
        [float(o), float(c), float(lo), float(hi)]
        for o, c, lo, hi in zip(opens, closes, lows, highs, strict=True)
    ]


def make_volume_series(df: pl.DataFrame) -> list[list[float]] | None:
    """Extract volume with direction sign per bar."""
    if "volume" not in df.columns:
        return None
    volumes = df["volume"].to_numpy().astype(float)
    closes = df["close"].to_numpy().astype(float)
    opens = df["open"].to_numpy().astype(float)
    return [
        [i, float(volumes[i]), 1 if closes[i] >= opens[i] else -1]
        for i in range(len(volumes))
    ]


def prepare_candlestick_data(
    df: pl.DataFrame,
    max_bars: int = 3000,
) -> tuple[pl.DataFrame, dict]:
    """Downsample OHLCV if needed, return prepared frame and metadata."""
    total_bars = len(df)
    if total_bars > max_bars:
        df = downsample_ohlcv(df, max_bars)
        logger.info("Candlestick: downsampled %d -> %d bars", total_bars, len(df))
        downsampled = True
    else:
        downsampled = False
    logger.info("Candlestick: rendering %d bars", len(df))
    return df, {
        "total_bars": total_bars,
        "displayed_bars": len(df),
        "downsampled": downsampled,
    }


def compute_normalized_confusion_matrix(
    true: np.ndarray,
    pred: np.ndarray,
    labels_order: list[int] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Row-normalized confusion matrix for ordered class labels."""
    if labels_order is None:
        labels_order = [-1, 1]
    display_labels = ["Short (-1)", "Long (1)"]
    n = len(labels_order)

    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(true, pred, strict=True):
        if t in labels_order and p in labels_order:
            ti = labels_order.index(int(t))
            pi = labels_order.index(int(p))
            cm[ti, pi] += 1

    cm_norm = cm.astype(float)
    for i in range(n):
        row_sum = cm[i].sum()
        if row_sum > 0:
            cm_norm[i] = cm[i] / row_sum

    return cm_norm, display_labels


__all__ = [
    "compute_normalized_confusion_matrix",
    "downsample_ohlcv",
    "make_kline_series",
    "make_volume_series",
    "parse_chart_timestamps",
    "prepare_candlestick_data",
]
