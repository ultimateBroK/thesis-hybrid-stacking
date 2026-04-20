"""Correlation filtering for feature columns (computed on train only)."""

import logging

import numpy as np
import polars as pl

logger = logging.getLogger("thesis.data")

_EXCLUDE_COLS = frozenset(
    [
        "timestamp",
        "label",
        "tp_price",
        "sl_price",
        "touched_bar",
        "open_right",
        "high_right",
        "low_right",
        "close_right",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "avg_spread",
        "tick_count",
        "dead_hour",
    ]
)


def _drop_correlated(
    train: pl.DataFrame,
    val: pl.DataFrame,
    test: pl.DataFrame,
    threshold: float,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Drop highly correlated features using train-set correlations only.

    Args:
        train: Training dataframe used to compute feature correlations.
        val: Validation dataframe to apply the same feature drop list.
        test: Test dataframe to apply the same feature drop list.
        threshold: Absolute correlation threshold above which a feature is
            removed.

    Returns:
        Tuple of `(train, val, test)` with correlated features removed.
    """
    feature_cols = [c for c in train.columns if c not in _EXCLUDE_COLS]
    if len(feature_cols) < 2:
        return train, val, test

    corr = train.select(feature_cols).to_pandas().corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    if not to_drop:
        logger.info("No highly correlated features found (threshold=%.2f)", threshold)
        return train, val, test

    logger.info("Dropping %d correlated features: %s", len(to_drop), to_drop[:10])
    train = train.drop(to_drop)
    val = val.drop(to_drop)
    test = test.drop(to_drop)
    logger.info(
        "Remaining features: %d", len([c for c in feature_cols if c not in to_drop])
    )
    return train, val, test
