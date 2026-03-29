"""Data splitting with purge and embargo for time-series.

Implements the market regime-based split scheme:
    Train (70%): 2018-2021 - Normal + Trade War + COVID shock
    Val (15%): 2022 - Russia-Ukraine + Fed hikes (stress test)
    Test (15%): 2023-03/2026 - SVB crisis + Gold ATH + "New Regime"

Purge/Embargo prevents data leakage between splits.
"""

import logging
from datetime import datetime
from pathlib import Path

import polars as pl

from thesis.config.loader import Config

logger = logging.getLogger("thesis.data")


def split_data(config: Config) -> None:
    """Split data into train/val/test with purge and embargo.
    
    Args:
        config: Configuration object.
    """
    # Load combined data
    labels_path = Path(config.labels.labels_path)
    
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels not found: {labels_path}")
    
    logger.info(f"Loading labeled data: {labels_path}")
    df = pl.read_parquet(labels_path)
    
    # Parse date strings and convert to Polars datetime with explicit dtype
    timestamp_dtype = df["timestamp"].dtype
    
    train_start = pl.lit(config.splitting.train_start).str.to_datetime().cast(timestamp_dtype)
    train_end = pl.lit(config.splitting.train_end).str.to_datetime().cast(timestamp_dtype)
    val_start = pl.lit(config.splitting.val_start).str.to_datetime().cast(timestamp_dtype)
    val_end = pl.lit(config.splitting.val_end).str.to_datetime().cast(timestamp_dtype)
    test_start = pl.lit(config.splitting.test_start).str.to_datetime().cast(timestamp_dtype)
    test_end = pl.lit(config.splitting.test_end).str.to_datetime().cast(timestamp_dtype)
    
    # Apply splits
    logger.info("Applying time-based splits...")
    
    train_mask = (pl.col("timestamp") >= train_start) & (pl.col("timestamp") <= train_end)
    val_mask = (pl.col("timestamp") >= val_start) & (pl.col("timestamp") <= val_end)
    test_mask = (pl.col("timestamp") >= test_start) & (pl.col("timestamp") <= test_end)
    
    train_df = df.filter(train_mask)
    val_df = df.filter(val_mask)
    test_df = df.filter(test_mask)
    
    # Log sizes
    total = len(df)
    train_pct = len(train_df) / total * 100
    val_pct = len(val_df) / total * 100
    test_pct = len(test_df) / total * 100
    
    logger.info(f"Raw split sizes:")
    logger.info(f"  Train: {len(train_df):,} ({train_pct:.1f}%)")
    logger.info(f"  Val: {len(val_df):,} ({val_pct:.1f}%)")
    logger.info(f"  Test: {len(test_df):,} ({test_pct:.1f}%)")
    
    # Apply purge and embargo
    if config.splitting.purge_bars > 0 or config.splitting.embargo_bars > 0:
        logger.info("Applying purge/embargo...")
        train_df, val_df, test_df = _apply_purge_embargo(
            train_df, val_df, test_df, config
        )
    
    # Log final sizes
    logger.info(f"Final split sizes (after purge/embargo):")
    logger.info(f"  Train: {len(train_df):,}")
    logger.info(f"  Val: {len(val_df):,}")
    logger.info(f"  Test: {len(test_df):,}")
    
    # Check class distribution
    _log_class_distribution(train_df, "Train")
    _log_class_distribution(val_df, "Val")
    _log_class_distribution(test_df, "Test")
    
    # Save splits
    train_path = Path(config.paths.train_data)
    val_path = Path(config.paths.val_data)
    test_path = Path(config.paths.test_data)
    
    train_path.parent.mkdir(parents=True, exist_ok=True)
    
    train_df.write_parquet(train_path)
    val_df.write_parquet(val_path)
    test_df.write_parquet(test_path)
    
    logger.info(f"Saved data splits to {train_path.parent}")


def _apply_purge_embargo(
    train: pl.DataFrame,
    val: pl.DataFrame,
    test: pl.DataFrame,
    config: Config,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Apply purge and embargo to prevent data leakage.
    
    Purge: Remove observations near boundaries between splits.
    Embargo: Don't evaluate on embargo bars (handled in CV).
    
    Args:
        train: Training DataFrame.
        val: Validation DataFrame.
        test: Test DataFrame.
        config: Configuration object.
        
    Returns:
        Tuple of (train, val, test) with purge applied.
    """
    purge = config.splitting.purge_bars
    
    if purge == 0:
        return train, val, test
    
    # Sort by timestamp
    train = train.sort("timestamp")
    val = val.sort("timestamp")
    test = test.sort("timestamp")
    
    # Get timestamps
    train_max = train["timestamp"].max()
    val_min = val["timestamp"].min()
    val_max = val["timestamp"].max()
    test_min = test["timestamp"].min()
    
    # Calculate purge boundaries (in terms of bar count, not time)
    # For H1, each bar is 1 hour
    from datetime import timedelta
    purge_delta = timedelta(hours=purge)
    
    # Purge from train end
    train_purge_cutoff = train_max - purge_delta
    train = train.filter(pl.col("timestamp") <= train_purge_cutoff)
    
    # Purge from val start and end
    val_purge_start = val_min + purge_delta
    val_purge_end = val_max - purge_delta
    val = val.filter(
        (pl.col("timestamp") >= val_purge_start) & 
        (pl.col("timestamp") <= val_purge_end)
    )
    
    # Purge from test start
    test_purge_start = test_min + purge_delta
    test = test.filter(pl.col("timestamp") >= test_purge_start)
    
    logger.info(f"Applied purge: {purge} bars ({purge} hours)")
    
    return train, val, test


def _log_class_distribution(df: pl.DataFrame, name: str) -> None:
    """Log class distribution for a split.
    
    Args:
        df: DataFrame with labels.
        name: Split name (Train/Val/Test).
    """
    if "label" not in df.columns:
        return
    
    counts = df["label"].value_counts().sort("label")
    total = len(df)
    
    logger.info(f"  {name} class distribution:")
    for row in counts.iter_rows():
        label, count = row
        pct = count / total * 100
        logger.info(f"    Class {label}: {count:,} ({pct:.1f}%)")
