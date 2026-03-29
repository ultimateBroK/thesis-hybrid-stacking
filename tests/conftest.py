"""
Test fixtures for the thesis testing suite.
Uses small slices of real XAUUSD data for testing.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import polars as pl
import pandas as pd
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
            "ema_windows": [20, 50, 200],
            "rsi_window": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "atr_window": 14,
            "bb_window": 20,
            "bb_std": 2.0,
            "session_col": "session",
            "pivot_col": "pivot_point",
            "max_correlation": 0.95,
            "target_feature_count": 20,
        },
        "labels": {
            "horizon": 10,
            "tp_multiplier": 2.0,
            "sl_multiplier": 1.0,
            "atr_col": "atr_14",
            "min_holding_bars": 3,
        },
        "splitting": {
            "train_ratio": 0.6,
            "val_ratio": 0.15,
            "test_ratio": 0.25,
            "purge_window": 10,
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
            "tp_atr_multiplier": 2.0,
            "sl_atr_multiplier": 1.0,
            "use_dynamic_sizing": False,
            "max_trades_per_day": 5,
        },
    }


# =============================================================================
# Data Fixtures (Real Data Samples)
# =============================================================================


@pytest.fixture(scope="session")
def raw_ohlcv_data() -> pl.DataFrame:
    """Load a sample of real OHLCV data (first 1000 rows from actual data)."""
    ohlcv_path = Path(__file__).parent.parent / "data" / "processed" / "ohlcv.parquet"

    if not ohlcv_path.exists():
        pytest.skip(f"OHLCV data not found at {ohlcv_path}")

    df = pl.read_parquet(ohlcv_path)

    # Take first 1000 rows for fast testing
    if len(df) > 1000:
        df = df.head(1000)

    return df


@pytest.fixture(scope="session")
def sample_ohlcv_df(raw_ohlcv_data) -> pd.DataFrame:
    """Pandas version of sample OHLCV data."""
    return raw_ohlcv_data.to_pandas()


@pytest.fixture(scope="session")
def sample_features_df() -> pl.DataFrame:
    """Load a sample of real features data."""
    features_path = (
        Path(__file__).parent.parent / "data" / "processed" / "features.parquet"
    )

    if not features_path.exists():
        pytest.skip(f"Features data not found at {features_path}")

    df = pl.read_parquet(features_path)

    if len(df) > 500:
        df = df.head(500)

    return df


@pytest.fixture(scope="session")
def sample_labels_df() -> pl.DataFrame:
    """Load a sample of real labels data."""
    labels_path = Path(__file__).parent.parent / "data" / "processed" / "labels.parquet"

    if not labels_path.exists():
        pytest.skip(f"Labels data not found at {labels_path}")

    df = pl.read_parquet(labels_path)

    if len(df) > 500:
        df = df.head(500)

    return df


@pytest.fixture(scope="session")
def train_data() -> pl.DataFrame:
    """Load a sample of real training data."""
    train_path = Path(__file__).parent.parent / "data" / "processed" / "train.parquet"

    if not train_path.exists():
        pytest.skip(f"Train data not found at {train_path}")

    df = pl.read_parquet(train_path)

    if len(df) > 500:
        df = df.head(500)

    return df


@pytest.fixture(scope="session")
def val_data() -> pl.DataFrame:
    """Load a sample of real validation data."""
    val_path = Path(__file__).parent.parent / "data" / "processed" / "val.parquet"

    if not val_path.exists():
        pytest.skip(f"Validation data not found at {val_path}")

    df = pl.read_parquet(val_path)

    if len(df) > 200:
        df = df.head(200)

    return df


@pytest.fixture(scope="session")
def test_data() -> pl.DataFrame:
    """Load a sample of real test data."""
    test_path = Path(__file__).parent.parent / "data" / "processed" / "test.parquet"

    if not test_path.exists():
        pytest.skip(f"Test data not found at {test_path}")

    df = pl.read_parquet(test_path)

    if len(df) > 200:
        df = df.head(200)

    return df


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
            "ema_20": 1500 + np.cumsum(np.random.randn(n) * 1.5),
            "ema_50": 1500 + np.cumsum(np.random.randn(n) * 1.2),
            "ema_200": 1500 + np.cumsum(np.random.randn(n) * 0.8),
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
        'probabilities': probs,
        'labels': labels,
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
        'probabilities': probs,
        'labels': labels,
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
    
    return {
        'predictions': predictions,
        'n_samples': n_samples
    }


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
