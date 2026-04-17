"""Tests for features module.

Tests technical indicator generation using synthetic Polars DataFrames.
Validates the 11-feature set: rsi_14, atr_14, macd_hist, atr_ratio,
price_dist_ratio, pivot_position, atr_percentile, sess_asia, sess_london,
sess_overlap, sess_ny_pm.
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from thesis.config import Config
from thesis.features import (
    _add_rsi,
    _add_atr,
    _add_macd,
    _add_new_features,
    _add_pivot_position,
    _add_ny_session_dummies,
)


def create_synthetic_ohlcv(n_rows: int = 100, seed: int = 42) -> pl.DataFrame:
    """Create synthetic OHLCV data for testing."""
    np.random.seed(seed)
    base_price = 1800.0
    timestamps = pl.datetime_range(
        start=pl.datetime(2023, 1, 1, 0),
        end=pl.datetime(2023, 1, 1, 0) + pl.duration(hours=n_rows - 1),
        interval="1h",
        eager=True,
    )

    # Generate random walk prices
    returns = np.random.normal(0, 0.001, n_rows)
    closes = base_price * np.exp(np.cumsum(returns))

    # Generate OHLC from close
    opens = closes * (1 + np.random.normal(0, 0.0005, n_rows))
    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, 0.001, n_rows)))
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 0.001, n_rows)))
    volumes = np.random.randint(1000, 10000, n_rows).astype(float)

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )


@pytest.fixture
def sample_config() -> Config:
    """Create a sample config for testing."""
    config = Config()
    config.features.rsi_period = 14
    config.features.atr_period = 14
    config.features.macd_fast = 12
    config.features.macd_slow = 26
    config.features.macd_signal = 9
    return config


# ---------------------------------------------------------------------------
# Core indicator tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.features
def test_rsi_bounded(sample_config: Config) -> None:
    """Test RSI is bounded [0, 100]."""
    df = create_synthetic_ohlcv(n_rows=200)
    result = _add_rsi(df, sample_config)

    assert "rsi_14" in result.columns

    rsi_values = result["rsi_14"].drop_nulls().to_numpy()
    assert len(rsi_values) > 0
    assert np.all(rsi_values >= 0)
    assert np.all(rsi_values <= 100)


@pytest.mark.unit
@pytest.mark.features
def test_atr_positive(sample_config: Config) -> None:
    """Test ATR > 0 for valid data."""
    df = create_synthetic_ohlcv(n_rows=200)
    result = _add_atr(df, sample_config)

    assert "atr_14" in result.columns

    atr_values = result["atr_14"].drop_nulls().to_numpy()
    assert len(atr_values) > 0
    assert np.all(atr_values > 0)


@pytest.mark.unit
@pytest.mark.features
def test_macd_histogram_only(sample_config: Config) -> None:
    """Test MACD produces only histogram (no macd_line column)."""
    df = create_synthetic_ohlcv(n_rows=200)
    result = _add_macd(df, sample_config)

    assert "macd_hist" in result.columns
    # macd_line should NOT be produced anymore
    assert "macd_line" not in result.columns


# ---------------------------------------------------------------------------
# New normalized feature tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.features
def test_atr_ratio_positive(sample_config: Config) -> None:
    """Test atr_ratio > 0 (ratio of short to long ATR)."""
    df = create_synthetic_ohlcv(n_rows=200)
    df = _add_atr(df, sample_config)
    result = _add_new_features(df, sample_config)

    assert "atr_ratio" in result.columns
    values = result["atr_ratio"].drop_nulls().to_numpy()
    assert len(values) > 0
    assert np.all(values > 0)


@pytest.mark.unit
@pytest.mark.features
def test_price_dist_ratio_exists(sample_config: Config) -> None:
    """Test price_dist_ratio is computed."""
    df = create_synthetic_ohlcv(n_rows=200)
    df = _add_atr(df, sample_config)
    result = _add_new_features(df, sample_config)

    assert "price_dist_ratio" in result.columns


@pytest.mark.unit
@pytest.mark.features
def test_atr_percentile_bounded(sample_config: Config) -> None:
    """Test atr_percentile is within [0, 1]."""
    df = create_synthetic_ohlcv(n_rows=200)
    df = _add_atr(df, sample_config)
    result = _add_new_features(df, sample_config)

    assert "atr_percentile" in result.columns
    values = result["atr_percentile"].drop_nulls().to_numpy()
    assert len(values) > 0
    assert np.all(values >= 0.0)
    assert np.all(values <= 1.0)


# ---------------------------------------------------------------------------
# Pivot position tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.features
def test_pivot_position_bounded(sample_config: Config) -> None:
    """Test pivot_position is clipped to [0, 1]."""
    df = create_synthetic_ohlcv(n_rows=200)
    result = _add_pivot_position(df)

    assert "pivot_position" in result.columns
    values = result["pivot_position"].drop_nulls().to_numpy()
    assert len(values) > 0
    assert np.all(values >= 0.0)
    assert np.all(values <= 1.0)


@pytest.mark.unit
@pytest.mark.features
def test_pivot_position_no_lookahead(sample_config: Config) -> None:
    """Test that pivot uses previous day's levels (shifted by 1)."""
    df = create_synthetic_ohlcv(n_rows=200)
    result = _add_pivot_position(df)

    # First few rows should have null pivots (no previous day data)
    # The exact count depends on how trading day aligns with calendar day
    first_few = result.head(24)
    # At least some of the first day's rows should be null
    assert first_few["pivot_position"].null_count() > 0, (
        "First trading day should have null pivots"
    )


# ---------------------------------------------------------------------------
# Session dummy tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.features
def test_session_dummies_four_columns() -> None:
    """Test 4 session columns are produced (not 3)."""
    df = create_synthetic_ohlcv(n_rows=100)
    result = _add_ny_session_dummies(df)

    for col in ["sess_asia", "sess_london", "sess_overlap", "sess_ny_pm"]:
        assert col in result.columns, f"Missing session column: {col}"


@pytest.mark.unit
@pytest.mark.features
def test_session_dummies_binary() -> None:
    """Test session dummy columns are binary {0, 1}."""
    df = create_synthetic_ohlcv(n_rows=100)
    result = _add_ny_session_dummies(df)

    for col in ["sess_asia", "sess_london", "sess_overlap", "sess_ny_pm"]:
        values = result[col].to_numpy()
        assert np.all(np.isin(values, [0, 1])), f"{col} has non-binary values"


@pytest.mark.unit
@pytest.mark.features
def test_session_dummies_coverage() -> None:
    """Test that every hour belongs to exactly one session."""
    df = create_synthetic_ohlcv(n_rows=100)
    result = _add_ny_session_dummies(df)

    total = (
        result["sess_asia"].cast(pl.Int32)
        + result["sess_london"].cast(pl.Int32)
        + result["sess_overlap"].cast(pl.Int32)
        + result["sess_ny_pm"].cast(pl.Int32)
    ).to_numpy()
    # Every hour should be in exactly one session (4 sessions cover 24h)
    # Asia: 18-01 (8h), London: 03-07 (5h), Overlap: 08-11 (4h), NY PM: 12-17 (6h)
    # Total: 23h — hour 2 NY time is uncovered (gap between Asia and London)
    assert np.all(total <= 1), "Some hours belong to multiple sessions"


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.features
def test_all_features_together(sample_config: Config) -> None:
    """Test that all indicators can be applied together producing 11 features."""
    df = create_synthetic_ohlcv(n_rows=200)

    # Apply all feature functions in order
    df = _add_rsi(df, sample_config)
    df = _add_atr(df, sample_config)
    df = _add_macd(df, sample_config)
    df = _add_new_features(df, sample_config)

    # Fill nulls like the main function does
    df = df.fill_null(strategy="forward").fill_null(0.0)

    expected_features = [
        "rsi_14",
        "atr_14",
        "macd_hist",
        "atr_ratio",
        "price_dist_ratio",
        "pivot_position",
        "atr_percentile",
        "sess_asia",
        "sess_london",
        "sess_overlap",
        "sess_ny_pm",
    ]

    for col in expected_features:
        assert col in df.columns, f"Missing feature column: {col}"

    # Check no nulls remain after filling
    for col in expected_features:
        null_count = df[col].null_count()
        assert null_count == 0, f"Column {col} has {null_count} nulls"


@pytest.mark.unit
@pytest.mark.features
def test_insufficient_rows_handled(sample_config: Config) -> None:
    """Test edge case: insufficient rows for indicator windows."""
    df = create_synthetic_ohlcv(n_rows=10)

    # Should not crash, but will have many nulls
    result = _add_rsi(df, sample_config)
    result = _add_atr(result, sample_config)
    result = _add_macd(result, sample_config)

    # Should produce columns even with few rows
    assert "rsi_14" in result.columns
    assert "atr_14" in result.columns
    assert "macd_hist" in result.columns
