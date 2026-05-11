"""Target helpers shared by tabular walk-forward trainers."""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

from thesis.shared.config import Config
from thesis.shared.constants import CENSORED_LABEL

logger = logging.getLogger("thesis.pipeline")


def _compute_regression_target(
    df: pl.DataFrame, config: Config
) -> tuple[pl.DataFrame, bool]:
    """Pre-compute regression target column when ``model.objective`` is regression."""
    is_regression = config.model.objective == "regression"
    if not is_regression:
        return df, False

    if "close" not in df.columns:
        raise ValueError(
            "Regression objective requires 'close' column in labeled data. "
            "Ensure feature engineering includes OHLCV data."
        )
    horizon = config.labels.horizon_bars
    close = df["close"].to_numpy()
    n = len(close)

    reg_target = np.full(n, np.nan, dtype=np.float64)
    close_future = np.roll(close, -horizon)[: n - horizon]
    reg_target[: n - horizon] = (close_future - close[: n - horizon]) / close[
        : n - horizon
    ]

    label_arr = df["label"].to_numpy().copy()
    tail_start = max(0, n - horizon)
    label_arr[tail_start:] = CENSORED_LABEL

    df = df.with_columns(
        [
            pl.Series("regression_target", reg_target),
            pl.Series("label", label_arr),
        ]
    )

    n_before = len(df)
    df = df.filter(pl.col("regression_target").is_not_nan())
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.info(
            "Dropped %d regression tail rows (%d horizon bars) — "
            "insufficient forward horizon",
            n_dropped,
            horizon,
        )

    logger.info(
        "Regression target computed: horizon=%d bars, mean=%.6f, std=%.6f",
        horizon,
        float(np.nanmean(reg_target)),
        float(np.nanstd(reg_target)),
    )
    return df, True
