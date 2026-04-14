"""Data loading and train/val/test splitting with purge/embargo.

Reads labeled parquet, splits by date range, applies purge/embargo,
and optionally drops highly correlated features (computed on train only).
"""

import logging
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

from thesis.config import Config

logger = logging.getLogger("thesis.data")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def split_data(config: Config) -> None:
    """Split labeled data into train / val / test with purge & embargo.

    Args:
        config: Loaded application configuration.
    """
    labels_path = Path(config.paths.labels)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels not found: {labels_path}")

    logger.info("Loading labeled data: %s", labels_path)
    df = pl.read_parquet(labels_path)
    logger.info("Total rows: %d", len(df))

    ts_dtype = df["timestamp"].dtype

    # Parse date boundaries
    bounds = {}
    for key in (
        "train_start",
        "train_end",
        "val_start",
        "val_end",
        "test_start",
        "test_end",
    ):
        bounds[key] = (
            pl.lit(getattr(config.splitting, key)).str.to_datetime().cast(ts_dtype)
        )

    train_df = df.filter(
        (pl.col("timestamp") >= bounds["train_start"])
        & (pl.col("timestamp") <= bounds["train_end"])
    )
    val_df = df.filter(
        (pl.col("timestamp") >= bounds["val_start"])
        & (pl.col("timestamp") <= bounds["val_end"])
    )
    test_df = df.filter(
        (pl.col("timestamp") >= bounds["test_start"])
        & (pl.col("timestamp") <= bounds["test_end"])
    )

    logger.info(
        "Raw split — train: %d, val: %d, test: %d",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    # Purge & embargo
    purge = config.splitting.purge_bars
    embargo = config.splitting.embargo_bars
    if purge > 0 or embargo > 0:
        train_df, val_df, test_df = _apply_purge_embargo(
            train_df, val_df, test_df, purge, embargo
        )
        logger.info(
            "After purge (%d) + embargo (%d) — train: %d, val: %d, test: %d",
            purge,
            embargo,
            len(train_df),
            len(val_df),
            len(test_df),
        )

    # Correlation filtering (train-only computation)
    if config.features.correlation_threshold < 1.0:
        train_df, val_df, test_df = _drop_correlated(
            train_df, val_df, test_df, config.features.correlation_threshold
        )

    # Log class distribution
    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        _log_distribution(name, split)

    # Save
    for tag, data in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = Path(getattr(config.paths, f"{tag}_data"))
        path.parent.mkdir(parents=True, exist_ok=True)
        data.write_parquet(path)
        logger.info("Saved %s: %s (%d rows)", tag, path, len(data))


# ---------------------------------------------------------------------------
# Purge / Embargo
# ---------------------------------------------------------------------------

_EXCLUDE_COLS = frozenset(
    [
        "timestamp",
        "label",
        "tp_price",
        "sl_price",
        "touched_bar",
        "open_right",  # Label-derived — pure look-ahead
        "high_right",  # Label-derived — pure look-ahead
        "low_right",  # Label-derived — pure look-ahead
        "close_right",  # Label-derived — pure look-ahead
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


def _apply_purge_embargo(
    train: pl.DataFrame,
    val: pl.DataFrame,
    test: pl.DataFrame,
    purge: int,
    embargo: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Remove bars at split boundaries to prevent label leakage.

    Args:
        purge: Bars to remove from ends of each split (covers label lookahead).
        embargo: Additional bars to remove from start of test set.
    """
    purge_delta = timedelta(hours=purge)
    embargo_delta = timedelta(hours=embargo)

    # Purge: remove last N bars from train, first/last N from val, first N from test
    train = train.filter(pl.col("timestamp") <= train["timestamp"].max() - purge_delta)
    val = val.filter(
        (pl.col("timestamp") >= val["timestamp"].min() + purge_delta)
        & (pl.col("timestamp") <= val["timestamp"].max() - purge_delta)
    )
    test = test.filter(pl.col("timestamp") >= test["timestamp"].min() + purge_delta)

    # Embargo: remove additional bars from start of test set
    if embargo > 0:
        test = test.filter(
            pl.col("timestamp") >= test["timestamp"].min() + embargo_delta
        )

    return train, val, test


# ---------------------------------------------------------------------------
# Correlation filtering
# ---------------------------------------------------------------------------


def _drop_correlated(
    train: pl.DataFrame,
    val: pl.DataFrame,
    test: pl.DataFrame,
    threshold: float,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Drop features with absolute correlation > *threshold* (train-only)."""
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log_distribution(name: str, df: pl.DataFrame) -> None:
    """Log class distribution for a split."""
    if "label" not in df.columns:
        return
    total = len(df)
    for row in df["label"].value_counts().sort("label").iter_rows():
        label, count = row
        logger.info(
            "  %s class %s: %d (%.1f%%)", name, label, count, count / total * 100
        )
