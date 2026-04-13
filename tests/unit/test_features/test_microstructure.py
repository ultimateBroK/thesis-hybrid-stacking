"""
Tests for microstructure features: body-wick ratio, streaks, close position, orderflow proxy.

These tests verify that:
1. No lookahead bias exists in feature computation
2. Features compute correctly on synthetic data
3. Features work on real XAU/USD data
4. Edge cases are handled properly
"""

import numpy as np
import pandas as pd
import pytest

from thesis.config.loader import Config, load_config
from thesis.features.engineering import _add_microstructure_features


@pytest.fixture
def sample_ohlcv():
    """Create a minimal OHLCV DataFrame for feature testing."""
    n = 100
    np.random.seed(42)
    base = 2000.0 + np.random.randn(n).cumsum()
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2020-01-01", periods=n, freq="1h", tz="America/New_York"
            ),
            "open": base - np.random.randn(n) * 0.5,
            "high": base + np.abs(np.random.randn(n)),
            "low": base - np.abs(np.random.randn(n)),
            "close": base,
            "volume": np.random.randint(100, 2000, n).astype(float),
            "avg_spread": np.random.uniform(0.1, 0.5, n),
        }
    )


# ---------------------------------------------------------------------------
# Feature existence tests
# ---------------------------------------------------------------------------


class TestMicrostructureFeatureExistence:
    """Verify that all expected microstructure features are generated."""

    EXPECTED_FEATURES = [
        "body_wick_ratio",
        "close_in_range",
        "consecutive_bull",
        "consecutive_bear",
        "volume_delta_ma_20",
    ]

    def test_all_features_present(self, sample_ohlcv):
        config = Config()
        result = _add_microstructure_features(sample_ohlcv, config)
        for feat in self.EXPECTED_FEATURES:
            assert feat in result.columns, f"Missing feature: {feat}"

    def test_no_intermediate_columns_leaked(self, sample_ohlcv):
        """Intermediate cols (body, range, wicks) should NOT appear in output."""
        config = Config()
        result = _add_microstructure_features(sample_ohlcv, config)
        for col in [
            "body",
            "range",
            "upper_wick",
            "lower_wick",
            "is_bullish",
            "is_bearish",
        ]:
            assert col not in result.columns, f"Intermediate column leaked: {col}"

    def test_no_removed_pattern_columns(self, sample_ohlcv):
        """Removed candlestick pattern features should NOT appear."""
        config = Config()
        result = _add_microstructure_features(sample_ohlcv, config)
        removed = [
            "bullish_engulfing",
            "bearish_engulfing",
            "doji",
            "hammer",
            "shooting_star",
            "marubozu",
            "large_body",
            "volume_delta",
            "is_bullish",
            "tick_intensity",
        ]
        for col in removed:
            assert col not in result.columns, f"Removed feature still present: {col}"


# ---------------------------------------------------------------------------
# No-lookahead tests
# ---------------------------------------------------------------------------


class TestMicrostructureNoLookahead:
    """Verify that microstructure features do not use future data."""

    def test_no_shift_negative_one_in_source(self):
        """Meta-test: _add_microstructure_features must not contain .shift(-1)."""
        import inspect

        source = inspect.getsource(_add_microstructure_features)
        assert ".shift(-1)" not in source, (
            "CRITICAL: _add_microstructure_features contains .shift(-1) — LOOKAHEAD BIAS!"
        )

    def test_body_wick_ratio_uses_only_current_bar(self, sample_ohlcv):
        """body_wick_ratio should be computable from a single bar's OHLC."""
        config = Config()
        result = _add_microstructure_features(sample_ohlcv, config)
        # Recompute manually for row 50
        row = sample_ohlcv.iloc[50]
        body = abs(row["close"] - row["open"])
        high, low = row["high"], row["low"]
        close, opn = row["close"], row["open"]
        upper_wick = high - max(close, opn)
        lower_wick = min(close, opn) - low
        expected = body / (upper_wick + lower_wick + 1e-10)
        assert abs(result["body_wick_ratio"].iloc[50] - expected) < 1e-6


# ---------------------------------------------------------------------------
# Feature value tests
# ---------------------------------------------------------------------------


class TestBodyWickRatio:
    """Tests for body_wick_ratio feature."""

    def test_values_non_negative_for_valid_ohlcv(self):
        """body_wick_ratio should be non-negative for valid OHLCV bars."""
        n = 50
        np.random.seed(42)
        base = 2000.0 + np.random.randn(n).cumsum()
        # Ensure valid OHLCV: low <= open,close <= high
        spread = np.abs(np.random.randn(n)) * 2
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2020-01-01", periods=n, freq="1h", tz="UTC"
                ),
                "open": base - np.random.randn(n) * 0.3,
                "close": base + np.random.randn(n) * 0.3,
                "high": base + spread + 0.5,  # always above open/close
                "low": base - spread - 0.5,  # always below open/close
                "volume": np.random.randint(100, 2000, n).astype(float),
                "avg_spread": np.random.uniform(0.1, 0.5, n),
            }
        )
        config = Config()
        result = _add_microstructure_features(df, config)
        valid = result["body_wick_ratio"].notna()
        assert (result["body_wick_ratio"][valid] >= 0).all()

    def test_high_ratio_for_marubozu(self):
        """A bar with no wicks should have very high body_wick_ratio."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2020-01-01", periods=5, freq="1h", tz="UTC"
                ),
                "open": [100, 100, 100, 100, 100],
                "high": [102, 100.5, 100, 100, 100],
                "low": [100, 99.5, 100, 100, 100],
                "close": [102, 100.5, 100, 101, 99],
                "volume": [1000] * 5,
                "avg_spread": [0.2] * 5,
            }
        )
        config = Config()
        result = _add_microstructure_features(df, config)
        # Bar 0: open=100, close=102, high=102, low=100 → body=2, upper_wick=0, lower_wick=0
        # Ratio = 2 / (0 + 0 + 1e-10) → very large
        assert result["body_wick_ratio"].iloc[0] > 1e6


class TestCloseInRange:
    """Tests for close_in_range feature."""

    def test_close_at_high_gives_near_one(self):
        """Close at high of bar should give close_in_range ≈ 1."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2020-01-01", periods=3, freq="1h", tz="UTC"
                ),
                "open": [100, 100, 100],
                "high": [102, 101, 103],
                "low": [100, 99, 100],
                "close": [102, 101, 103],  # close == high
                "volume": [1000] * 3,
                "avg_spread": [0.2] * 3,
            }
        )
        config = Config()
        result = _add_microstructure_features(df, config)
        assert (result["close_in_range"] > 0.99).all()

    def test_close_at_low_gives_near_zero(self):
        """Close at low of bar should give close_in_range ≈ 0."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2020-01-01", periods=3, freq="1h", tz="UTC"
                ),
                "open": [102, 101, 103],
                "high": [102, 101, 103],
                "low": [100, 99, 100],
                "close": [100, 99, 100],  # close == low
                "volume": [1000] * 3,
                "avg_spread": [0.2] * 3,
            }
        )
        config = Config()
        result = _add_microstructure_features(df, config)
        assert (result["close_in_range"] < 0.01).all()

    def test_values_between_zero_and_one(self, sample_ohlcv):
        config = Config()
        result = _add_microstructure_features(sample_ohlcv, config)
        valid = result["close_in_range"].notna()
        assert result["close_in_range"][valid].between(0.0, 1.0).all()


class TestConsecutiveStreaks:
    """Tests for consecutive_bull and consecutive_bear features."""

    def test_streaks_start_at_zero_or_one(self, sample_ohlcv):
        """First bar should have streak of 1 if bullish/bearish, or 0."""
        config = Config()
        result = _add_microstructure_features(sample_ohlcv, config)
        assert result["consecutive_bull"].iloc[0] >= 0
        assert result["consecutive_bear"].iloc[0] >= 0

    def test_streaks_count_correctly(self):
        """Verify streak counting on a controlled sequence."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2020-01-01", periods=6, freq="1h", tz="UTC"
                ),
                "open": [100, 100, 100, 100, 100, 100],
                "high": [101, 101, 101, 101, 101, 101],
                "low": [99, 99, 99, 99, 99, 99],
                "close": [101, 101, 99, 99, 99, 101],  # B B S S S B
                "volume": [1000] * 6,
                "avg_spread": [0.2] * 6,
            }
        )
        config = Config()
        result = _add_microstructure_features(df, config)
        assert result["consecutive_bull"].tolist() == [1, 2, 0, 0, 0, 1]
        assert result["consecutive_bear"].tolist() == [0, 0, 1, 2, 3, 0]


class TestVolumeDeltaMA:
    """Tests for volume_delta_ma_20 feature."""

    def test_first_19_values_are_nan(self):
        """Rolling 20-bar MA should produce NaN for first 19 bars."""
        n = 50
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2020-01-01", periods=n, freq="1h", tz="UTC"
                ),
                "open": np.random.randn(n) + 100,
                "high": np.random.randn(n) + 101,
                "low": np.random.randn(n) + 99,
                "close": np.random.randn(n) + 100,
                "volume": np.random.randint(100, 2000, n).astype(float),
                "avg_spread": np.random.uniform(0.1, 0.5, n),
            }
        )
        config = Config()
        result = _add_microstructure_features(df, config)
        assert result["volume_delta_ma_20"].iloc[:19].isna().all()
        assert result["volume_delta_ma_20"].iloc[19:].notna().all()

    def test_positive_when_bars_are_bullish(self):
        """If all bars are bullish (close > open), volume_delta_ma_20 should be positive."""
        n = 25
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2020-01-01", periods=n, freq="1h", tz="UTC"
                ),
                "open": [100] * n,
                "high": [101] * n,
                "low": [99] * n,
                "close": [100.5] * n,  # always close > open
                "volume": [1000] * n,
                "avg_spread": [0.2] * n,
            }
        )
        config = Config()
        result = _add_microstructure_features(df, config)
        # Bar 20 (index 19) should have positive MA (all bars contributed +volume)
        assert result["volume_delta_ma_20"].iloc[19] > 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestMicrostructureEdgeCases:
    """Edge case tests for microstructure features."""

    def test_single_row(self):
        """Should handle a single row without error."""
        df = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2020-01-01", tz="UTC")],
                "open": [100],
                "high": [101],
                "low": [99],
                "close": [100.5],
                "volume": [1000],
                "avg_spread": [0.2],
            }
        )
        config = Config()
        result = _add_microstructure_features(df, config)
        assert len(result) == 1
        assert "body_wick_ratio" in result.columns

    def test_doji_bar_zero_body(self):
        """A doji bar (open == close) should produce body_wick_ratio ≈ 0."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2020-01-01", periods=3, freq="1h", tz="UTC"
                ),
                "open": [100, 100, 100],
                "high": [101, 101, 101],
                "low": [99, 99, 99],
                "close": [100, 100, 100],  # open == close → zero body
                "volume": [1000] * 3,
                "avg_spread": [0.2] * 3,
            }
        )
        config = Config()
        result = _add_microstructure_features(df, config)
        assert (result["body_wick_ratio"] < 1e-6).all()


# ---------------------------------------------------------------------------
# Real data tests
# ---------------------------------------------------------------------------


class TestMicrostructureRealData:
    """Tests using real XAU/USD OHLCV data."""

    @pytest.fixture
    def real_data(self):
        from pathlib import Path

        ohlcv_path = Path("data/processed/ohlcv.parquet")
        if not ohlcv_path.exists():
            pytest.skip("No processed OHLCV data available")
        import polars as pl

        return pl.read_parquet(ohlcv_path).to_pandas()

    def test_features_generated_on_real_data(self, real_data):
        config = load_config()
        result = _add_microstructure_features(real_data, config)
        for feat in TestMicrostructureFeatureExistence.EXPECTED_FEATURES:
            assert feat in result.columns, f"Missing feature on real data: {feat}"

    def test_body_wick_ratio_reasonable_range(self, real_data):
        config = load_config()
        result = _add_microstructure_features(real_data, config)
        valid = result["body_wick_ratio"].notna()
        # Most values should be reasonable; extreme values can occur on zero-wick bars
        median = result["body_wick_ratio"][valid].median()
        assert median < 10, f"Median body_wick_ratio unusually high: {median}"

    def test_close_in_range_valid(self, real_data):
        config = load_config()
        result = _add_microstructure_features(real_data, config)
        valid = result["close_in_range"].notna()
        assert result["close_in_range"][valid].between(0.0, 1.0).all()
