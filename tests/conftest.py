"""
Test fixtures for the thesis testing suite.
Uses small slices of real XAUUSD data for testing.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import polars as pl
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from thesis.config.loader import load_config, Config


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def config_path() -> Path:
    """Path to the main config file."""
    return Path(__file__).parent.parent / "config.toml"


@pytest.fixture(scope="session")
def config(config_path: Path) -> Config:
    """Loaded configuration object."""
    return load_config(str(config_path))


@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Minimal test configuration."""
    return {
        "data": {
            "raw_data_path": "data/raw/XAUUSD",
            "ohlcv_path": "tests/fixtures/sample_ohlcv.parquet",
            "timeframe": "1H",
            "timeframe_minutes": 60,
            "market_tz": "America/New_York",
            "day_roll_hour": 17,
            "start_date": "2020-01-01",
            "end_date": "2020-03-31",
            "tick_size": 0.01,
            "contract_size": 100,
        },
        "features": {
            "ema_windows": [34, 89],
            "rsi_window": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "atr_window": 14,
            "bb_window": 20,
            "bb_std": 2.0,
            "session_col": "session",
            "pivot_col": "pivot_point",
            "max_correlation": 0.90,
            "target_feature_count": 20,
        },
        "labels": {
            "horizon": 20,
            "tp_multiplier": 1.5,
            "sl_multiplier": 1.5,
            "atr_col": "atr_14",
            "min_holding_bars": 3,
        },
        "splitting": {
            "train_ratio": 0.6,
            "val_ratio": 0.15,
            "test_ratio": 0.25,
            "purge_window": 15,
            "embargo_window": 10,
            "enforce_temporal": True,
            "train_start": "2020-01-01",
            "train_end": "2020-01-31",
            "val_start": "2020-02-01",
            "val_end": "2020-02-29",
            "test_start": "2020-03-01",
            "test_end": "2020-03-31",
        },
        "models": {
            "lstm": {
                "input_size": 5,
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.2,
                "sequence_length": 60,
                "batch_size": 32,
                "epochs": 2,
                "learning_rate": 0.001,
                "early_stopping_patience": 5,
            },
            "lightgbm": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "num_leaves": 31,
                "min_child_samples": 20,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
                "random_state": 42,
                "class_weight": "balanced",
            },
        },
        "backtest": {
            "initial_capital": 10000.0,
            "risk_per_trade": 0.01,
            "max_position_size": 2.0,
            "spread_pips": 2,
            "slippage_pips": 1,
            "commission_per_lot": 0.0,
            "tp_atr_multiplier": 1.5,
            "sl_atr_multiplier": 1.5,
            "use_dynamic_sizing": False,
            "max_trades_per_day": 5,
        },
    }


# =============================================================================
# Data Fixtures (Real Data Samples)
# =============================================================================


@pytest.fixture(scope="session")
def raw_ohlcv_data() -> pl.DataFrame:
    """
    Load and aggregate raw tick data from real XAU/USD parquet files.

    Loads tick data from data/raw/XAUUSD/ and aggregates to H1 OHLCV.
    Uses 3 months of data (2020-01 to 2020-03) for fast testing.
    """
    raw_data_path = Path(__file__).parent.parent / "data" / "raw" / "XAUUSD"

    if not raw_data_path.exists():
        pytest.skip(f"Raw data directory not found at {raw_data_path}")

    # Find parquet files for 2020-01, 2020-02, 2020-03 (3 months of data)
    target_files = [
        "2020-01.parquet",
        "2020-02.parquet",
        "2020-03.parquet",
    ]

    available_files = []
    for fname in target_files:
        fpath = raw_data_path / fname
        if fpath.exists():
            available_files.append(fpath)

    if not available_files:
        # Fallback: use any available parquet files (limit to first 3)
        all_files = sorted(raw_data_path.glob("*.parquet"))[:3]
        if not all_files:
            pytest.skip(f"No parquet files found in {raw_data_path}")
        available_files = all_files

    # Load and aggregate tick data
    ohlcv_data = []

    for tick_file in available_files:
        try:
            # Read tick data
            ticks = pl.read_parquet(tick_file)

            # Calculate mid price
            ticks = ticks.with_columns(
                [
                    ((pl.col("ask") + pl.col("bid")) / 2).alias("mid"),
                    (pl.col("ask") - pl.col("bid")).alias("spread"),
                ]
            )

            # Convert timestamp to datetime (UTC)
            ticks = ticks.with_columns(pl.col("timestamp").cast(pl.Datetime("ms")))

            # Convert to market timezone (America/New_York) for proper day roll
            ticks = ticks.with_columns(
                pl.col("timestamp")
                .dt.replace_time_zone("UTC")
                .dt.convert_time_zone("America/New_York")
                .alias("market_time")
            )

            # Floor to hour and aggregate
            ticks = ticks.with_columns(
                pl.col("market_time").dt.truncate("1h").alias("hour")
            )

            # Aggregate to H1 OHLCV
            ohlcv = ticks.group_by("hour").agg(
                [
                    pl.col("mid").first().alias("open"),
                    pl.col("mid").max().alias("high"),
                    pl.col("mid").min().alias("low"),
                    pl.col("mid").last().alias("close"),
                    pl.col("ask_volume").sum().alias("ask_volume"),
                    pl.col("bid_volume").sum().alias("bid_volume"),
                    pl.col("spread").mean().alias("avg_spread"),
                    pl.len().alias("tick_count"),
                ]
            )

            ohlcv = ohlcv.sort("hour")
            ohlcv = ohlcv.rename({"hour": "timestamp"})

            # Add total volume
            ohlcv = ohlcv.with_columns(
                [(pl.col("ask_volume") + pl.col("bid_volume")).alias("volume")]
            )

            # Drop separate ask/bid volume
            ohlcv = ohlcv.drop(["ask_volume", "bid_volume"])

            # Select and reorder columns
            ohlcv = ohlcv.select(
                [
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "avg_spread",
                    "tick_count",
                ]
            )

            ohlcv_data.append(ohlcv)
        except Exception as e:
            pytest.skip(f"Failed to process {tick_file.name}: {e}")

    if not ohlcv_data:
        pytest.skip("No valid data processed from tick files")

    # Concatenate all chunks
    full_ohlcv = pl.concat(ohlcv_data, how="vertical")
    full_ohlcv = full_ohlcv.sort("timestamp")
    full_ohlcv = full_ohlcv.unique(subset=["timestamp"], keep="first")

    # Take first 2000 rows for fast testing
    if len(full_ohlcv) > 2000:
        full_ohlcv = full_ohlcv.head(2000)

    return full_ohlcv


@pytest.fixture(scope="session")
def sample_ohlcv_df(raw_ohlcv_data) -> pd.DataFrame:
    """Pandas version of sample OHLCV data."""
    return raw_ohlcv_data.to_pandas()


@pytest.fixture(scope="session")
def sample_features_df(raw_ohlcv_data) -> pl.DataFrame:
    """
    Generate features from real OHLCV data.

    Uses the thesis features engineering module to generate technical indicators,
    microstructure features, and lag features from the real XAU/USD OHLCV data.
    """
    try:
        from thesis.features.engineering import (
            _add_microstructure_features,
            _add_lag_features,
            _add_spread_features,
        )
        from thesis.config.loader import load_config
    except ImportError as e:
        pytest.skip(f"Could not import feature engineering modules: {e}")

    # Load minimal config
    config_path = Path(__file__).parent.parent / "config.toml"
    try:
        config = load_config(str(config_path))
    except Exception as e:
        pytest.skip(f"Could not load config: {e}")

    # Convert to pandas for feature engineering
    df_pd = raw_ohlcv_data.to_pandas()

    # Ensure timestamp is datetime with timezone
    df_pd["timestamp"] = pd.to_datetime(df_pd["timestamp"])
    if df_pd["timestamp"].dt.tz is None:
        df_pd["timestamp"] = df_pd["timestamp"].dt.tz_localize(
            "America/New_York", ambiguous="infer"
        )

    # Generate microstructure features
    df_pd = _add_microstructure_features(df_pd, config)

    # Generate lag features
    df_pd = _add_lag_features(df_pd, config)

    # Generate spread features
    df_pd = _add_spread_features(df_pd, config)

    # Convert back to polars
    df = pl.from_pandas(df_pd)

    # Handle missing values
    df = df.fill_null(strategy="forward")
    df = df.fill_null(0)

    # Take first 1000 rows
    if len(df) > 1000:
        df = df.head(1000)

    return df


@pytest.fixture(scope="session")
def sample_labels_df(raw_ohlcv_data) -> pl.DataFrame:
    """
    Generate triple-barrier labels from real OHLCV data.

    Generates labels (-1=Short, 0=Hold, 1=Long) using triple-barrier method
    from the real XAU/USD OHLCV data.
    """
    import numpy as np
    import talib

    # Convert to pandas for processing
    df_pd = raw_ohlcv_data.to_pandas()

    # Calculate ATR
    df_pd["atr_14"] = talib.ATR(
        df_pd["high"], df_pd["low"], df_pd["close"], timeperiod=14
    )

    # Triple-barrier parameters
    horizon = 20
    tp_multiplier = 1.5
    sl_multiplier = 1.5
    min_atr = 0.0001

    # Generate labels
    labels = []
    timestamps = []

    n = len(df_pd)
    close_arr = df_pd["close"].to_numpy()
    high_arr = df_pd["high"].to_numpy()
    low_arr = df_pd["low"].to_numpy()
    atr_arr = df_pd["atr_14"].to_numpy()
    time_arr = df_pd["timestamp"].tolist()

    for i in range(n - horizon):
        current_close = close_arr[i]
        current_atr = atr_arr[i]

        # Skip if ATR is too small
        if current_atr < min_atr or np.isnan(current_atr):
            labels.append(0)
            timestamps.append(time_arr[i])
            continue

        # Calculate barriers
        tp_price = current_close + tp_multiplier * current_atr
        sl_price = current_close - sl_multiplier * current_atr

        # Look ahead
        future_high = high_arr[i + 1 : i + 1 + horizon]
        future_low = low_arr[i + 1 : i + 1 + horizon]

        # Check which barrier hit first
        label = 0

        for h, l in zip(future_high, future_low):
            if h >= tp_price:
                label = 1  # Long
                break
            elif l <= sl_price:
                label = -1  # Short
                break

        labels.append(label)
        timestamps.append(time_arr[i])

    # Handle last horizon bars (can't look ahead)
    for i in range(n - horizon, n):
        labels.append(0)  # Neutral for bars at end
        timestamps.append(time_arr[i])

    # Create labels DataFrame
    labels_df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "label": labels,
        }
    )

    # Take first 1000 rows
    if len(labels_df) > 1000:
        labels_df = labels_df.head(1000)

    return labels_df


@pytest.fixture(scope="session")
def train_data(sample_features_df, sample_labels_df) -> pl.DataFrame:
    """
    Create training data by merging features and labels.
    Takes the first 60% of data for training to maintain temporal ordering.
    """
    try:
        if "timestamp" not in sample_features_df.columns:
            pytest.skip("Features DataFrame missing timestamp column")
        if "timestamp" not in sample_labels_df.columns:
            pytest.skip("Labels DataFrame missing timestamp column")

        # Ensure timestamp columns have the same dtype
        features_df = sample_features_df.with_columns(
            pl.col("timestamp").cast(pl.Datetime("us", "America/New_York"))
        )
        labels_df = sample_labels_df.with_columns(
            pl.col("timestamp").cast(pl.Datetime("us", "America/New_York"))
        )

        # Merge on timestamp and sort
        merged = features_df.join(labels_df, on="timestamp", how="inner")
        merged = merged.sort("timestamp")

        if len(merged) == 0:
            pytest.skip("No matching timestamps between features and labels")

        # Take first 60% for training (temporal split)
        n_train = int(len(merged) * 0.6)
        train_df = merged.head(n_train)

        if len(train_df) > 500:
            train_df = train_df.head(500)

        return train_df
    except Exception as e:
        pytest.skip(f"Failed to merge features and labels: {e}")


@pytest.fixture(scope="session")
def val_data(sample_features_df, sample_labels_df) -> pl.DataFrame:
    """Create validation data from middle 20% of data (after train)."""
    try:
        if "timestamp" not in sample_features_df.columns:
            pytest.skip("Features DataFrame missing timestamp column")
        if "timestamp" not in sample_labels_df.columns:
            pytest.skip("Labels DataFrame missing timestamp column")

        # Ensure timestamp columns have the same dtype
        features_df = sample_features_df.with_columns(
            pl.col("timestamp").cast(pl.Datetime("us", "America/New_York"))
        )
        labels_df = sample_labels_df.with_columns(
            pl.col("timestamp").cast(pl.Datetime("us", "America/New_York"))
        )

        # Merge on timestamp and sort
        merged = features_df.join(labels_df, on="timestamp", how="inner")
        merged = merged.sort("timestamp")

        if len(merged) == 0:
            pytest.skip("No matching timestamps between features and labels")

        # Take middle 20% for validation (after first 60%)
        n_total = len(merged)
        start_idx = int(n_total * 0.6)
        n_val = int(n_total * 0.2)

        val_df = merged.slice(start_idx, n_val)

        if len(val_df) > 200:
            val_df = val_df.head(200)

        return val_df
    except Exception as e:
        pytest.skip(f"Failed to create validation data: {e}")


@pytest.fixture(scope="session")
def test_data(sample_features_df, sample_labels_df) -> pl.DataFrame:
    """Create test data from last 20% of data (after val)."""
    try:
        if "timestamp" not in sample_features_df.columns:
            pytest.skip("Features DataFrame missing timestamp column")
        if "timestamp" not in sample_labels_df.columns:
            pytest.skip("Labels DataFrame missing timestamp column")

        # Ensure timestamp columns have the same dtype
        features_df = sample_features_df.with_columns(
            pl.col("timestamp").cast(pl.Datetime("us", "America/New_York"))
        )
        labels_df = sample_labels_df.with_columns(
            pl.col("timestamp").cast(pl.Datetime("us", "America/New_York"))
        )

        # Merge on timestamp and sort
        merged = features_df.join(labels_df, on="timestamp", how="inner")
        merged = merged.sort("timestamp")

        if len(merged) == 0:
            pytest.skip("No matching timestamps between features and labels")

        # Take last 20% for test (after first 80%)
        n_total = len(merged)
        start_idx = int(n_total * 0.8)

        test_df = merged.slice(start_idx, n_total - start_idx)

        if len(test_df) > 200:
            test_df = test_df.head(200)

        return test_df
    except Exception as e:
        pytest.skip(f"Failed to create test data: {e}")


# =============================================================================
# Synthetic Data Fixtures
# =============================================================================


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    n = 200

    dates = pd.date_range("2020-01-01", periods=n, freq="h", tz="America/New_York")

    # Generate realistic OHLCV data
    close = 1500 + np.cumsum(np.random.randn(n) * 2)
    open_price = close + np.random.randn(n) * 1
    high = np.maximum(open_price, close) + np.random.uniform(0.5, 3, n)
    low = np.minimum(open_price, close) - np.random.uniform(0.5, 3, n)
    volume = np.random.randint(100, 1000, n)

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "spread": np.random.uniform(0.1, 0.5, n),
        }
    )

    df.set_index("timestamp", inplace=True)
    return df


@pytest.fixture
def synthetic_features() -> pd.DataFrame:
    """Generate synthetic feature data."""
    np.random.seed(42)
    n = 200

    dates = pd.date_range("2020-01-01", periods=n, freq="h", tz="America/New_York")

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "close": 1500 + np.cumsum(np.random.randn(n) * 2),
            "ema_34": 1500 + np.cumsum(np.random.randn(n) * 1.5),
            "ema_89": 1500 + np.cumsum(np.random.randn(n) * 1.2),
            "rsi_14": np.random.uniform(20, 80, n),
            "macd": np.random.randn(n) * 5,
            "macd_signal": np.random.randn(n) * 4,
            "atr_14": np.random.uniform(2, 10, n),
            "bb_upper": 1550 + np.random.uniform(0, 20, n),
            "bb_middle": 1500 + np.random.uniform(-10, 10, n),
            "bb_lower": 1450 - np.random.uniform(0, 20, n),
            "pivot_point": 1500 + np.random.uniform(-5, 5, n),
            "r1": 1510 + np.random.uniform(0, 10, n),
            "s1": 1490 - np.random.uniform(0, 10, n),
            "hour": dates.hour,
            "session": np.random.choice(["Asia", "London", "NY"], n),
            "spread": np.random.uniform(0.1, 0.5, n),
            "atr_10": np.random.uniform(2, 10, n),
        }
    )

    return df


# =============================================================================
# Synthetic Candlestick Pattern Fixtures
# =============================================================================


@pytest.fixture
def synthetic_bull_engulfing() -> pd.DataFrame:
    """
    Generate synthetic OHLCV with bullish engulfing patterns.

    Pattern: Bearish bar followed by larger bullish bar that engulfs it.
    """
    np.random.seed(100)
    n = 50

    dates = pd.date_range("2020-01-01", periods=n, freq="h", tz="America/New_York")

    # Base price level
    base_price = 1500.0

    # Create pattern at index 10-11
    opens = np.full(n, base_price)
    closes = np.full(n, base_price)
    highs = np.full(n, base_price + 2)
    lows = np.full(n, base_price - 2)
    volumes = np.random.randint(100, 500, n)

    # Bar 10: Bearish (close < open)
    opens[10] = base_price + 5
    closes[10] = base_price - 3  # -8 body, bearish
    highs[10] = base_price + 6
    lows[10] = base_price - 4
    volumes[10] = 300

    # Bar 11: Bullish Engulfing (close > open, larger body)
    opens[11] = base_price - 5  # Open below previous close
    closes[11] = base_price + 8  # Close above previous open (+13 body)
    highs[11] = base_price + 9
    lows[11] = base_price - 6
    volumes[11] = 400  # Higher volume

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "avg_spread": np.full(n, 0.2),
        }
    )

    return df


@pytest.fixture
def synthetic_bear_engulfing() -> pd.DataFrame:
    """
    Generate synthetic OHLCV with bearish engulfing patterns.

    Pattern: Bullish bar followed by larger bearish bar that engulfs it.
    """
    np.random.seed(101)
    n = 50

    dates = pd.date_range("2020-01-01", periods=n, freq="h", tz="America/New_York")

    base_price = 1500.0

    opens = np.full(n, base_price)
    closes = np.full(n, base_price)
    highs = np.full(n, base_price + 2)
    lows = np.full(n, base_price - 2)
    volumes = np.random.randint(100, 500, n)

    # Bar 20: Bullish (close > open)
    opens[20] = base_price - 3
    closes[20] = base_price + 5  # +8 body, bullish
    highs[20] = base_price + 6
    lows[20] = base_price - 4
    volumes[20] = 300

    # Bar 21: Bearish Engulfing (close < open, larger body)
    opens[21] = base_price + 6  # Open above previous close
    closes[21] = base_price - 7  # Close below previous open (-13 body)
    highs[21] = base_price + 7
    lows[21] = base_price - 8
    volumes[21] = 400

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "avg_spread": np.full(n, 0.2),
        }
    )

    return df


@pytest.fixture
def synthetic_doji() -> pd.DataFrame:
    """
    Generate synthetic OHLCV with doji patterns.

    Pattern: Open and close very close together (body < 10% of range).
    """
    np.random.seed(102)
    n = 50

    dates = pd.date_range("2020-01-01", periods=n, freq="h", tz="America/New_York")

    base_price = 1500.0

    opens = np.full(n, base_price) + np.random.randn(n) * 2
    closes = opens + np.random.randn(n) * 3  # Normal variation
    highs = np.maximum(opens, closes) + np.random.uniform(2, 5, n)
    lows = np.minimum(opens, closes) - np.random.uniform(2, 5, n)
    volumes = np.random.randint(100, 500, n)

    # Bar 30: Doji (very small body, long wicks)
    opens[30] = base_price
    closes[30] = base_price + 0.1  # Tiny body (0.1 vs range of ~10)
    highs[30] = base_price + 5
    lows[30] = base_price - 5
    volumes[30] = 250

    # Bar 31: Another doji with slightly larger body but still < 10%
    opens[31] = base_price + 2
    closes[31] = base_price + 2.3  # Body = 0.3, range = 8.3
    highs[31] = base_price + 6
    lows[31] = base_price - 2
    volumes[31] = 200

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "avg_spread": np.full(n, 0.2),
        }
    )

    return df


@pytest.fixture
def synthetic_hammer() -> pd.DataFrame:
    """
    Generate synthetic OHLCV with hammer patterns.

    Pattern: Bullish bar with small body near high, long lower wick.
    """
    np.random.seed(103)
    n = 50

    dates = pd.date_range("2020-01-01", periods=n, freq="h", tz="America/New_York")

    base_price = 1500.0

    opens = np.full(n, base_price) + np.random.randn(n) * 2
    closes = opens + np.random.randn(n) * 2
    highs = np.maximum(opens, closes) + np.random.uniform(1, 3, n)
    lows = np.minimum(opens, closes) - np.random.uniform(1, 3, n)
    volumes = np.random.randint(100, 500, n)

    # Bar 40: Hammer (bullish, small body, long lower wick, small upper wick)
    opens[40] = base_price - 1
    closes[40] = base_price + 1  # Small bullish body (2)
    highs[40] = base_price + 2  # Small upper wick (1)
    lows[40] = base_price - 10  # Long lower wick (11 > 2*2)
    volumes[40] = 350

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "avg_spread": np.full(n, 0.2),
        }
    )

    return df


@pytest.fixture
def synthetic_shooting_star() -> pd.DataFrame:
    """
    Generate synthetic OHLCV with shooting star patterns.

    Pattern: Bearish bar with small body near low, long upper wick.
    """
    np.random.seed(104)
    n = 50

    dates = pd.date_range("2020-01-01", periods=n, freq="h", tz="America/New_York")

    base_price = 1500.0

    opens = np.full(n, base_price) + np.random.randn(n) * 2
    closes = opens + np.random.randn(n) * 2
    highs = np.maximum(opens, closes) + np.random.uniform(1, 3, n)
    lows = np.minimum(opens, closes) - np.random.uniform(1, 3, n)
    volumes = np.random.randint(100, 500, n)

    # Bar 45: Shooting Star (bearish, small body, long upper wick, small lower wick)
    opens[45] = base_price + 2
    closes[45] = base_price - 1  # Small bearish body (3)
    highs[45] = base_price + 12  # Long upper wick (13 > 2*3)
    lows[45] = base_price - 2  # Small lower wick (1 < 3)
    volumes[45] = 350

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "avg_spread": np.full(n, 0.2),
        }
    )

    return df


@pytest.fixture
def synthetic_marubozu() -> pd.DataFrame:
    """
    Generate synthetic OHLCV with marubozu patterns.

    Pattern: Strong directional bar with no wicks (or very small).
    """
    np.random.seed(105)
    n = 50

    dates = pd.date_range("2020-01-01", periods=n, freq="h", tz="America/New_York")

    base_price = 1500.0

    opens = np.full(n, base_price) + np.random.randn(n) * 2
    closes = opens + np.random.randn(n) * 2
    highs = np.maximum(opens, closes) + np.random.uniform(1, 5, n)
    lows = np.minimum(opens, closes) - np.random.uniform(1, 5, n)
    volumes = np.random.randint(100, 500, n)

    # Bar 48: Bullish Marubozu (no upper or lower wick)
    opens[48] = base_price
    closes[48] = base_price + 10  # Large bullish body
    highs[48] = base_price + 10.1  # Tiny upper wick (< 10% of range)
    lows[48] = base_price - 0.1  # Tiny lower wick (< 10% of range)
    volumes[48] = 500

    # Bar 49: Bearish Marubozu
    opens[49] = base_price + 10
    closes[49] = base_price  # Large bearish body
    highs[49] = base_price + 10.1
    lows[49] = base_price - 0.1
    volumes[49] = 500

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "avg_spread": np.full(n, 0.2),
        }
    )

    return df


@pytest.fixture
def synthetic_volume_delta_positive() -> pd.DataFrame:
    """
    Generate synthetic OHLCV where close > open (positive volume delta).
    """
    np.random.seed(106)
    n = 30

    dates = pd.date_range("2020-01-01", periods=n, freq="h", tz="America/New_York")

    # All bars are bullish (close > open)
    opens = 1500 + np.arange(n) * 0.5
    closes = opens + np.random.uniform(1, 5, n)  # Always positive body
    highs = closes + np.random.uniform(0.5, 2, n)
    lows = opens - np.random.uniform(0.5, 2, n)
    volumes = np.random.randint(100, 500, n)

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "avg_spread": np.full(n, 0.2),
        }
    )

    return df


@pytest.fixture
def synthetic_volume_delta_negative() -> pd.DataFrame:
    """
    Generate synthetic OHLCV where close < open (negative volume delta).
    """
    np.random.seed(107)
    n = 30

    dates = pd.date_range("2020-01-01", periods=n, freq="h", tz="America/New_York")

    # All bars are bearish (close < open)
    opens = 1500 + np.arange(n) * 0.5
    closes = opens - np.random.uniform(1, 5, n)  # Always negative body
    highs = opens + np.random.uniform(0.5, 2, n)
    lows = closes - np.random.uniform(0.5, 2, n)
    volumes = np.random.randint(100, 500, n)

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "avg_spread": np.full(n, 0.2),
        }
    )

    return df


@pytest.fixture
def synthetic_consecutive_bars() -> pd.DataFrame:
    """
    Generate synthetic OHLCV with consecutive bullish and bearish bars.
    """
    np.random.seed(108)
    n = 50

    dates = pd.date_range("2020-01-01", periods=n, freq="h", tz="America/New_York")

    opens = np.zeros(n)
    closes = np.zeros(n)
    highs = np.zeros(n)
    lows = np.zeros(n)
    volumes = np.random.randint(100, 500, n)

    base_price = 1500.0

    # Create pattern: 5 bullish, 3 bearish, 4 bullish
    # Indices 0-4: Bullish
    for i in range(5):
        opens[i] = base_price + i
        closes[i] = opens[i] + 2  # Bullish
        highs[i] = closes[i] + 1
        lows[i] = opens[i] - 1

    # Indices 5-7: Bearish
    for i in range(5, 8):
        opens[i] = base_price + i
        closes[i] = opens[i] - 2  # Bearish
        highs[i] = opens[i] + 1
        lows[i] = closes[i] - 1

    # Indices 8-11: Bullish
    for i in range(8, 12):
        opens[i] = base_price + i
        closes[i] = opens[i] + 2  # Bullish
        highs[i] = closes[i] + 1
        lows[i] = opens[i] - 1

    # Fill rest randomly
    for i in range(12, n):
        opens[i] = base_price + i
        closes[i] = opens[i] + np.random.choice([-2, 2])
        highs[i] = max(opens[i], closes[i]) + 1
        lows[i] = min(opens[i], closes[i]) - 1

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "avg_spread": np.full(n, 0.2),
        }
    )

    return df


@pytest.fixture
def synthetic_high_volume_intensity() -> pd.DataFrame:
    """
    Generate synthetic OHLCV with high tick intensity (volume >> avg).
    """
    np.random.seed(109)
    n = 50

    dates = pd.date_range("2020-01-01", periods=n, freq="h", tz="America/New_York")

    base_price = 1500.0

    # Low volume baseline
    volumes = np.random.randint(50, 150, n)

    opens = base_price + np.random.randn(n) * 2
    closes = opens + np.random.randn(n) * 2
    highs = np.maximum(opens, closes) + np.random.uniform(1, 3, n)
    lows = np.minimum(opens, closes) - np.random.uniform(1, 3, n)

    # Bar 25: Very high volume (3x average)
    volumes[25] = 600
    volumes[26] = 550

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "avg_spread": np.full(n, 0.2),
        }
    )

    return df


@pytest.fixture
def synthetic_close_at_high() -> pd.DataFrame:
    """
    Generate synthetic OHLCV with close near high (strong bullish).
    """
    np.random.seed(110)
    n = 30

    dates = pd.date_range("2020-01-01", periods=n, freq="h", tz="America/New_York")

    base_price = 1500.0

    opens = base_price + np.random.randn(n) * 2
    closes = np.zeros(n)
    highs = np.zeros(n)
    lows = np.zeros(n)
    volumes = np.random.randint(100, 500, n)

    # Most bars: close near high
    for i in range(n):
        closes[i] = opens[i] + np.random.uniform(3, 8)  # Bullish
        highs[i] = closes[i] + np.random.uniform(0, 0.5)  # Close very near high
        lows[i] = opens[i] - np.random.uniform(2, 5)

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "avg_spread": np.full(n, 0.2),
        }
    )

    return df


@pytest.fixture
def synthetic_close_at_low() -> pd.DataFrame:
    """
    Generate synthetic OHLCV with close near low (strong bearish).
    """
    np.random.seed(111)
    n = 30

    dates = pd.date_range("2020-01-01", periods=n, freq="h", tz="America/New_York")

    base_price = 1500.0

    opens = base_price + np.random.randn(n) * 2
    closes = np.zeros(n)
    highs = np.zeros(n)
    lows = np.zeros(n)
    volumes = np.random.randint(100, 500, n)

    # Most bars: close near low
    for i in range(n):
        closes[i] = opens[i] - np.random.uniform(3, 8)  # Bearish
        highs[i] = opens[i] + np.random.uniform(2, 5)
        lows[i] = closes[i] - np.random.uniform(0, 0.5)  # Close very near low

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "avg_spread": np.full(n, 0.2),
        }
    )

    return df


# =============================================================================
# Model Fixtures
# =============================================================================


@pytest.fixture
def mock_lstm_predictions() -> Dict[str, np.ndarray]:
    """Generate mock LSTM predictions for testing stacking."""
    np.random.seed(42)
    n = 100
    # 3-class probabilities
    probs = np.random.dirichlet([1, 1, 1], n)
    labels = np.random.choice([-1, 0, 1], size=n)
    return {
        "probabilities": probs,
        "labels": labels,
    }


@pytest.fixture
def mock_lightgbm_predictions() -> Dict[str, np.ndarray]:
    """Generate mock LightGBM predictions for testing stacking."""
    np.random.seed(43)
    n = 100
    # 3-class probabilities
    probs = np.random.dirichlet([1, 1, 1], n)
    labels = np.random.choice([-1, 0, 1], size=n)
    return {
        "probabilities": probs,
        "labels": labels,
    }


@pytest.fixture
def mock_predictions_with_low_confidence() -> np.ndarray:
    """
    Generate mock predictions with specific low confidence cases.

    Returns: (predictions, max_probs, expected_after_threshold)
    """
    np.random.seed(112)
    n = 50

    # Create predictions with specific confidence levels
    probs = np.zeros((n, 3))

    # Cases 0-9: Low confidence (< 0.6) - should become Hold
    for i in range(10):
        # Create distribution with max around 0.55
        probs[i] = [0.55, 0.25, 0.20]

    # Cases 10-19: Just below threshold (0.59) - should become Hold
    for i in range(10, 20):
        probs[i] = [0.59, 0.21, 0.20]

    # Cases 20-29: At threshold (0.60) - should be preserved
    for i in range(20, 30):
        probs[i] = [0.60, 0.20, 0.20]

    # Cases 30-39: Above threshold (0.70) - should be preserved
    for i in range(30, 40):
        probs[i] = [0.70, 0.15, 0.15]

    # Cases 40-49: High confidence (0.85) - should be preserved
    for i in range(40, 50):
        probs[i] = [0.85, 0.10, 0.05]

    max_probs = np.max(probs, axis=1)
    original_preds = np.argmax(probs, axis=1) - 1  # -1, 0, 1

    # Expected after threshold: cases 0-19 become 0 (Hold)
    expected = original_preds.copy()
    expected[0:20] = 0  # Low confidence -> Hold

    return {
        "probabilities": probs,
        "max_probabilities": max_probs,
        "original_predictions": original_preds,
        "expected_after_threshold": expected,
        "n_low_confidence": 20,
        "n_high_confidence": 30,
    }


@pytest.fixture
def mock_predictions_exact_threshold() -> Dict[str, np.ndarray]:
    """
    Generate mock predictions to test exact threshold boundaries.
    """
    n = 6

    probs = np.array(
        [
            [0.599, 0.201, 0.200],  # Just below threshold -> Hold
            [0.600, 0.200, 0.200],  # At threshold -> Original (first class)
            [0.601, 0.199, 0.200],  # Just above threshold -> Original
            [0.200, 0.599, 0.201],  # Hold class, low confidence -> Hold (no change)
            [0.200, 0.600, 0.200],  # Hold class, at threshold -> Hold
            [0.199, 0.601, 0.200],  # Hold class, above threshold -> Hold
        ]
    )

    max_probs = np.max(probs, axis=1)
    original_preds = np.argmax(probs, axis=1) - 1

    # Expected: all except index 0 and 3 remain the same
    # Index 0: 0.599 < 0.6 -> Hold (class 0)
    # Index 3: 0.599 < 0.6, but it's already Hold -> stays Hold
    expected = np.array([0, -1, -1, 0, 0, 0])

    return {
        "probabilities": probs,
        "max_probabilities": max_probs,
        "original_predictions": original_preds,
        "expected_after_threshold": expected,
    }


@pytest.fixture
def mock_predictions_high_confidence() -> Dict[str, np.ndarray]:
    """
    Generate mock predictions with all high confidence (above 0.6 threshold).
    """
    np.random.seed(47)
    n = 50

    # All probabilities have max >= 0.65 (above 0.6 threshold)
    probs = np.array(
        [
            [0.15, 0.65, 0.20],
            [0.70, 0.15, 0.15],
            [0.20, 0.20, 0.60],
        ]
    )

    # Repeat to get 50 samples
    full_probs = np.tile(probs, (17, 1))[:n]

    # Add small variations to make them unique
    noise = np.random.randn(n, 3) * 0.01
    full_probs = full_probs + noise

    # Normalize to ensure they sum to 1
    full_probs = full_probs / full_probs.sum(axis=1, keepdims=True)

    predictions = np.argmax(full_probs, axis=1) - 1  # Convert to -1, 0, 1

    return {
        "predictions": predictions,
        "probabilities": full_probs,
        "max_probabilities": np.max(full_probs, axis=1),
    }


@pytest.fixture
def mock_labels() -> np.ndarray:
    """Generate mock labels for testing."""
    np.random.seed(44)
    return np.random.choice([-1, 0, 1], size=100)


@pytest.fixture
def mock_predictions_with_labels() -> Dict[str, np.ndarray]:
    """Generate mock predictions with labels for backtesting."""
    np.random.seed(45)
    n_samples = 200

    # Generate class predictions (-1, 0, 1)
    predictions = np.random.choice([-1, 0, 1], size=n_samples)

    return {"predictions": predictions, "n_samples": n_samples}


@pytest.fixture
def mock_balanced_labels() -> np.ndarray:
    """
    Generate mock labels with balanced distribution for testing.

    Approximate 35% Short, 30% Hold, 35% Long
    """
    np.random.seed(113)
    n = 300

    # Create exact distribution
    short_count = int(n * 0.35)
    hold_count = int(n * 0.30)
    long_count = n - short_count - hold_count

    labels = np.concatenate(
        [
            np.full(short_count, -1),
            np.full(hold_count, 0),
            np.full(long_count, 1),
        ]
    )

    # Shuffle
    np.random.shuffle(labels)

    return labels


@pytest.fixture
def mock_symmetric_labels() -> np.ndarray:
    """
    Generate mock labels simulating symmetric barriers (1.5x/1.5x).

    Should be roughly balanced between Long and Short.
    """
    np.random.seed(114)
    n = 400

    # With symmetric barriers, expect roughly equal Long and Short
    # Hold represents cases where neither barrier is hit
    short_count = int(n * 0.33)
    long_count = int(n * 0.33)
    hold_count = n - short_count - long_count

    labels = np.concatenate(
        [
            np.full(short_count, -1),
            np.full(hold_count, 0),
            np.full(long_count, 1),
        ]
    )

    np.random.shuffle(labels)

    return labels


# =============================================================================
# Path Fixtures
# =============================================================================


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def src_path(project_root: Path) -> Path:
    """Return the src directory path."""
    return project_root / "src"


@pytest.fixture
def data_path(project_root: Path) -> Path:
    """Return the data directory path."""
    return project_root / "data"


@pytest.fixture
def models_path(project_root: Path) -> Path:
    """Return the models directory path."""
    return project_root / "models"


# =============================================================================
# Shared Fixtures for Data Leakage Testing
# =============================================================================


@pytest.fixture
def temp_norm_stats_file(tmp_path) -> Path:
    """Create a temporary normalization stats file for LSTM testing."""
    stats_path = tmp_path / "test_norm_stats.npz"

    # Save mock stats
    np.savez(
        stats_path,
        mean=np.array([1500.0, 0.0, 1500.0, 1500.0, 500.0]),
        std=np.array([50.0, 5.0, 50.0, 50.0, 200.0]),
    )

    return stats_path


@pytest.fixture
def data_with_overlap() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create train/test data with intentional overlap for testing purge/embargo."""
    dates_train = pd.date_range("2020-01-01", periods=100, freq="h")
    dates_test = pd.date_range(
        "2020-01-05", periods=50, freq="h"
    )  # Overlaps with train

    train_df = pd.DataFrame(
        {
            "timestamp": dates_train,
            "close": np.random.randn(100),
            "label": np.random.choice([-1, 0, 1], 100),
        }
    )

    test_df = pd.DataFrame(
        {
            "timestamp": dates_test,
            "close": np.random.randn(50),
            "label": np.random.choice([-1, 0, 1], 50),
        }
    )

    return train_df, test_df
