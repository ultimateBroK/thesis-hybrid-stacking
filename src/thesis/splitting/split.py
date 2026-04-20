"""Data loading and train/val/test splitting with purge/embargo.

Reads labeled parquet, splits by date range, applies purge/embargo,
and optionally drops highly correlated features (computed on train only).
"""

import logging
from pathlib import Path

import polars as pl

from thesis.config import Config
from .correlation import _drop_correlated

logger = logging.getLogger("thesis.data")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def split_data(config: Config) -> None:
    """
    Split labeled data into train / val / test with purge & embargo.

    Reads labeled parquet, filters by date boundaries from config, applies purge and embargo
    to prevent label leakage between splits, optionally drops highly correlated features
    (computed on train only), and writes train/val/test parquet files.

    Args:
        config: Application configuration with:
            - paths.labels: input labeled data path
            - splitting.{train_start,train_end,val_start,val_end,test_start,test_end}: date boundaries
            - splitting.purge_bars: bars to remove at split boundaries
            - splitting.embargo_bars: additional bars to remove from test start
            - splitting.correlation_threshold: optional threshold for dropping correlated features
    """
    labels_path = Path(config.paths.labels)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels not found: {labels_path}")

    logger.info("Loading labeled data: %s", labels_path)
    df = pl.read_parquet(labels_path)
    logger.info("Total rows: %d", len(df))

    bounds = _parse_date_bounds(df, config)
    train_df, val_df, test_df = _filter_splits_by_dates(df, bounds)
    logger.info(
        "Raw split — train: %d, val: %d, test: %d",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    train_df, val_df, test_df = _apply_purge_and_embargo(
        train_df, val_df, test_df, config
    )
    if config.features.correlation_threshold < 1.0:
        train_df, val_df, test_df = _drop_correlated(
            train_df, val_df, test_df, config.features.correlation_threshold
        )
    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        _log_distribution(name, split)
    _save_split_files(config, train_df, val_df, test_df)


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

    if len(train) > purge:
        train = train.head(len(train) - purge)
    else:
        logger.warning("Train set is smaller than purge size!")

    if len(val) > 2 * purge:
        val = val.slice(purge, len(val) - 2 * purge)
    else:
        logger.warning("Val set is smaller than 2 * purge size!")

    drop_test = purge + embargo
    if len(test) > drop_test:
        test = test.slice(drop_test, len(test) - drop_test)
    else:
        logger.warning("Test set is smaller than purge + embargo size!")

    return train, val, test


def _parse_date_bounds(df: pl.DataFrame, config: Config) -> dict:
    """Parse date boundary strings from config into typed Polars expressions."""
    ts_dtype = df["timestamp"].dtype
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
    return bounds


def _filter_splits_by_dates(
    df: pl.DataFrame, bounds: dict
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Filter dataframe into train/val/test splits by date boundaries."""
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
    return train_df, val_df, test_df


def _apply_purge_and_embargo(
    train_df: pl.DataFrame, val_df: pl.DataFrame, test_df: pl.DataFrame, config: Config
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Apply purge and embargo, log the result, and return modified splits."""
    purge = config.splitting.purge_bars
    embargo = config.splitting.embargo_bars
    if purge <= 0 and embargo <= 0:
        return train_df, val_df, test_df
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
    return train_df, val_df, test_df


def _save_split_files(
    config: Config, train_df: pl.DataFrame, val_df: pl.DataFrame, test_df: pl.DataFrame
) -> None:
    """Write train/val/test DataFrames to their configured parquet paths."""
    for tag, data in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = Path(getattr(config.paths, f"{tag}_data"))
        path.parent.mkdir(parents=True, exist_ok=True)
        data.write_parquet(path)
        logger.info("Saved %s: %s (%d rows)", tag, path, len(data))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log_distribution(name: str, df: pl.DataFrame) -> None:
    """
    Log class distribution for a dataset split.

    Logs the count and percentage of each label class (-1=Short, 0=Hold, 1=Long)
    in the given DataFrame. Used during pipeline execution to verify class balance.

    Parameters:
        name: Name of the split (e.g., 'train', 'val', 'test') for logging purposes.
        df: DataFrame expected to contain a 'label' column.
    """
    if "label" not in df.columns:
        return
    total = len(df)
    for row in df["label"].value_counts().sort("label").iter_rows():
        label, count = row
        logger.info(
            "  %s class %s: %d (%.1f%%)", name, label, count, count / total * 100
        )
