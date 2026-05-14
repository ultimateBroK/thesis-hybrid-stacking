"""Training targets. Share label math across trainers."""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

from thesis.shared.config import Config
from thesis.shared.constants import CENSORED_LABEL

logger = logging.getLogger("thesis")


def compute_regression_target(
    df: pl.DataFrame, config: Config
) -> tuple[pl.DataFrame, bool]:
    """Add forward-return target for regression.

    Tail rows cannot see horizon; mark censored then drop target-null rows.
    """
    is_regression = config.model.objective == "regression"
    if not is_regression:
        return df, False

    if "close" not in df.columns:
        raise ValueError("Regression objective requires 'close' column in labeled data")

    h = config.labels.horizon_bars
    close = df["close"].to_numpy()
    n = len(close)

    reg = np.full(n, np.nan, dtype=np.float64)
    future = np.roll(close, -h)[: n - h]
    reg[: n - h] = (future - close[: n - h]) / close[: n - h]

    # Tail cannot compute forward return
    label_arr = df["label"].to_numpy().copy()
    tail_start = max(0, n - h)
    label_arr[tail_start:] = CENSORED_LABEL

    df = df.with_columns(
        [
            pl.Series("regression_target", reg),
            pl.Series("label", label_arr),
        ]
    )

    # Drop target-null tail rows
    n_before = len(df)
    df = df.filter(pl.col("regression_target").is_not_nan())
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.info(
            "  Dropped %d regression tail rows (insufficient forward horizon)",
            n_dropped,
        )

    logger.info(
        "  Regression target: horizon=%d bars, mean=%.6f, std=%.6f",
        h,
        float(np.nanmean(reg)),
        float(np.nanstd(reg)),
    )
    return df, True
