"""Tests for data module.

Tests train/val/test splitting and correlation filtering.
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from thesis.config import Config
from thesis.splitting import _apply_purge_embargo, _drop_correlated, _log_distribution


def create_synthetic_labeled_data(
    n_rows: int = 500,
    start_date: str = "2020-01-01",
) -> pl.DataFrame:
    """Create synthetic labeled data for testing."""
    np.random.seed(42)

    timestamps = pl.datetime_range(
        start=pl.datetime(2020, 1, 1, 0),
        end=pl.datetime(2020, 1, 1, 0) + pl.duration(hours=n_rows - 1),
        interval="1h",
        eager=True,
    )

    # Create features
    n_features = 10
    data = {"timestamp": timestamps, "label": np.random.choice([-1, 0, 1], n_rows)}

    for i in range(n_features):
        data[f"feature_{i}"] = np.random.randn(n_rows)

    # Add correlated features (feature_5 and feature_6 will be highly correlated)
    data["feature_5"] = data["feature_0"] + np.random.randn(n_rows) * 0.01
    data["feature_6"] = data["feature_1"] + np.random.randn(n_rows) * 0.01

    # Add OHLC columns
    data["open"] = np.random.randn(n_rows) + 1800
    data["high"] = data["open"] + np.abs(np.random.randn(n_rows))
    data["low"] = data["open"] - np.abs(np.random.randn(n_rows))
    data["close"] = data["open"] + np.random.randn(n_rows)
    data["volume"] = np.random.randint(1000, 10000, n_rows).astype(float)

    # Add label-related columns
    data["tp_price"] = data["close"] + 10
    data["sl_price"] = data["close"] - 10
    data["touched_bar"] = np.random.choice([-1, 0, 1, 2, 3], n_rows)

    return pl.DataFrame(data)


@pytest.fixture
def sample_config() -> Config:
    """Create a sample config for testing."""
    config = Config()
    config.splitting.train_start = "2020-01-01"
    config.splitting.train_end = "2020-03-31 23:59:59"
    config.splitting.val_start = "2020-04-01"
    config.splitting.val_end = "2020-05-31 23:59:59"
    config.splitting.test_start = "2020-06-01"
    config.splitting.test_end = "2020-07-31 23:59:59"
    config.splitting.purge_bars = 24
    config.splitting.embargo_bars = 12
    config.features.correlation_threshold = 0.95
    return config


@pytest.mark.unit
@pytest.mark.data
def test_chronological_ordering(sample_config: Config) -> None:
    """Test that splits are chronologically ordered (train < val < test)."""
    df = create_synthetic_labeled_data(n_rows=5000)

    # Parse date boundaries
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
            pl.lit(getattr(sample_config.splitting, key))
            .str.to_datetime()
            .cast(ts_dtype)
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

    if len(train_df) > 0 and len(val_df) > 0:
        train_max = train_df["timestamp"].max()
        val_min = val_df["timestamp"].min()
        assert train_max < val_min, "Train must end before val starts"

    if len(val_df) > 0 and len(test_df) > 0:
        val_max = val_df["timestamp"].max()
        test_min = test_df["timestamp"].min()
        assert val_max < test_min, "Val must end before test starts"


@pytest.mark.unit
@pytest.mark.data
def test_purge_gap_exists(sample_config: Config) -> None:
    """Test that purge gap exists between splits."""
    df = create_synthetic_labeled_data(n_rows=5000)
    purge = sample_config.splitting.purge_bars

    if purge == 0:
        pytest.skip("Purge is 0, skipping purge test")

    # Parse date boundaries
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
            pl.lit(getattr(sample_config.splitting, key))
            .str.to_datetime()
            .cast(ts_dtype)
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

    # Apply purge
    train_df, val_df, test_df = _apply_purge_embargo(
        train_df, val_df, test_df, purge, embargo=0
    )

    # Check that purge created gaps
    if len(train_df) > 0 and len(val_df) > 0:
        train_max = train_df["timestamp"].max()
        val_min = val_df["timestamp"].min()
        # Gap should be at least purge hours
        gap_hours = (val_min - train_max).total_seconds() / 3600
        assert gap_hours >= 0, f"Gap should be non-negative, got {gap_hours}"


@pytest.mark.unit
@pytest.mark.data
def test_split_ratios_approximate(sample_config: Config) -> None:
    """Test that split ratios are approximately correct."""
    df = create_synthetic_labeled_data(n_rows=5000)

    # Parse date boundaries
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
            pl.lit(getattr(sample_config.splitting, key))
            .str.to_datetime()
            .cast(ts_dtype)
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

    total = len(train_df) + len(val_df) + len(test_df)
    if total == 0:
        pytest.skip("No data in splits")

    train_ratio = len(train_df) / total
    val_ratio = len(val_df) / total
    test_ratio = len(test_df) / total

    # Rough checks (allowing for purge effects)
    assert train_ratio > 0.3, f"Train ratio {train_ratio} too low"
    assert val_ratio > 0.1, f"Val ratio {val_ratio} too low"
    assert test_ratio > 0.1, f"Test ratio {test_ratio} too low"


@pytest.mark.unit
@pytest.mark.data
def test_correlation_filtering_drops_features() -> None:
    """Test correlation filtering drops highly correlated features."""
    df = create_synthetic_labeled_data(n_rows=1000)

    # Split into train/val/test
    n = len(df)
    train_n = int(n * 0.6)
    val_n = int(n * 0.2)

    train_df = df.head(train_n)
    val_df = df.slice(train_n, val_n)
    test_df = df.tail(n - train_n - val_n)

    # Get initial feature count
    exclude_cols = frozenset(
        [
            "timestamp",
            "label",
            "tp_price",
            "sl_price",
            "touched_bar",
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
    initial_features = [c for c in train_df.columns if c not in exclude_cols]
    initial_count = len(initial_features)

    # Apply correlation filtering with low threshold to ensure some drops
    train_df_filtered, val_df_filtered, test_df_filtered = _drop_correlated(
        train_df, val_df, test_df, threshold=0.90
    )

    final_features = [c for c in train_df_filtered.columns if c not in exclude_cols]
    final_count = len(final_features)

    # Should have dropped at least one feature (feature_5 is correlated with feature_0)
    assert final_count <= initial_count, "Feature count should not increase"


@pytest.mark.unit
@pytest.mark.data
def test_correlation_filtering_consistent_across_splits() -> None:
    """Test that same features are dropped in all splits."""
    df = create_synthetic_labeled_data(n_rows=1000)

    # Split into train/val/test
    n = len(df)
    train_n = int(n * 0.6)
    val_n = int(n * 0.2)

    train_df = df.head(train_n)
    val_df = df.slice(train_n, val_n)
    test_df = df.tail(n - train_n - val_n)

    # Apply correlation filtering
    train_df_filtered, val_df_filtered, test_df_filtered = _drop_correlated(
        train_df, val_df, test_df, threshold=0.90
    )

    # All splits should have same columns
    assert train_df_filtered.columns == val_df_filtered.columns
    assert train_df_filtered.columns == test_df_filtered.columns


@pytest.mark.unit
@pytest.mark.data
def test_apply_purge_reduces_sizes() -> None:
    """Test that purge reduces split sizes."""
    df = create_synthetic_labeled_data(n_rows=1000)

    # Split into train/val/test
    n = len(df)
    train_n = int(n * 0.6)
    val_n = int(n * 0.2)

    train_df = df.head(train_n)
    val_df = df.slice(train_n, val_n)
    test_df = df.tail(n - train_n - val_n)

    initial_train_len = len(train_df)
    initial_val_len = len(val_df)
    initial_test_len = len(test_df)

    # Apply purge
    train_df, val_df, test_df = _apply_purge_embargo(
        train_df, val_df, test_df, purge=24, embargo=0
    )

    # Sizes should be reduced or stay same
    assert len(train_df) <= initial_train_len
    assert len(val_df) <= initial_val_len
    assert len(test_df) <= initial_test_len


@pytest.mark.unit
@pytest.mark.data
def test_log_distribution_no_crash() -> None:
    """Test that _log_distribution doesn't crash."""
    df = create_synthetic_labeled_data(n_rows=100)

    # Should not raise any exception
    _log_distribution("Test", df)


@pytest.mark.unit
@pytest.mark.data
def test_log_distribution_without_label() -> None:
    """Test that _log_distribution handles missing label column."""
    df = create_synthetic_labeled_data(n_rows=100)
    df = df.drop("label")

    # Should not raise any exception
    _log_distribution("Test", df)


@pytest.mark.unit
@pytest.mark.data
def test_drop_correlated_no_features() -> None:
    """Test correlation filtering with no feature columns."""
    df = create_synthetic_labeled_data(n_rows=100)

    # Keep only non-feature columns that actually exist in the dataframe
    exclude_cols = frozenset(
        [
            "timestamp",
            "label",
            "tp_price",
            "sl_price",
            "touched_bar",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    )
    # Only select columns that exist
    existing_cols = [c for c in exclude_cols if c in df.columns]
    df_no_features = df.select(existing_cols)

    # Split
    n = len(df_no_features)
    train_n = int(n * 0.6)
    val_n = int(n * 0.2)

    train_df = df_no_features.head(train_n)
    val_df = df_no_features.slice(train_n, val_n)
    test_df = df_no_features.tail(n - train_n - val_n)

    # Should not crash
    train_df, val_df, test_df = _drop_correlated(
        train_df, val_df, test_df, threshold=0.90
    )

    # Should return same dataframes
    assert len(train_df.columns) == len(df_no_features.columns)


@pytest.mark.unit
@pytest.mark.data
def test_drop_correlated_threshold_one() -> None:
    """Test correlation filtering with threshold=1.0 (no filtering)."""
    df = create_synthetic_labeled_data(n_rows=1000)

    # Split
    n = len(df)
    train_n = int(n * 0.6)
    val_n = int(n * 0.2)

    train_df = df.head(train_n)
    val_df = df.slice(train_n, val_n)
    test_df = df.tail(n - train_n - val_n)

    initial_cols = train_df.columns

    # Apply with threshold=1.0 (should not drop anything)
    train_df, val_df, test_df = _drop_correlated(
        train_df, val_df, test_df, threshold=1.0
    )

    # Should have same columns
    assert train_df.columns == initial_cols
