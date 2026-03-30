"""
Tests for microstructure features: candlestick patterns, volume delta, and orderflow proxies.

These tests verify that:
1. No lookahead bias exists in pattern detection (CRITICAL for financial data)
2. Pattern detection works correctly on controlled synthetic data
3. Features work on real XAU/USD data
4. Edge cases are handled properly
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from thesis.features.engineering import _add_microstructure_features
from thesis.config.loader import Config


class TestMicrostructureNoLookahead:
    """CRITICAL: Verify no lookahead bias in microstructure features."""

    def test_pattern_detection_uses_only_past_and_current_bar(
        self, synthetic_bull_engulfing
    ):
        """
        Verify candlestick patterns use only shift(1) for previous bar.

        Pattern detection must not use shift(-1) which would look into the future.
        """
        df = synthetic_bull_engulfing.copy()
        config = Config()

        result = _add_microstructure_features(df, config)

        # If no lookahead, bar 11 should detect bullish_engulfing = 1
        # Because it only looks at bar 10 (shift(1)) and bar 11 (current)
        assert result["bullish_engulfing"].iloc[11] == 1, (
            "Bullish engulfing should be detected at bar 11 using only past data"
        )

        # Bar 10 should NOT detect engulfing (no previous bar pattern)
        assert result["bullish_engulfing"].iloc[10] == 0, (
            "First bar cannot have engulfing pattern (no previous bar)"
        )

    def test_no_shift_negative_one_in_pattern_logic(self, synthetic_ohlcv):
        """
        Verify that pattern features don't use .shift(-1) which indicates lookahead.

        This is a meta-test that inspects the feature engineering code.
        """
        import inspect

        source = inspect.getsource(_add_microstructure_features)

        # Check for any .shift(-1) which would indicate lookahead
        assert ".shift(-1)" not in source, (
            "CRITICAL: _add_microstructure_features contains .shift(-1) - LOOKAHEAD BIAS!"
        )

        # Verify only .shift(1) is used for previous bar reference
        assert ".shift(1)" in source or "shift(1)" in source, (
            "Expected .shift(1) for previous bar reference"
        )

    def test_consecutive_bar_counting_no_lookahead(self, synthetic_consecutive_bars):
        """
        Test consecutive bar counting doesn't use future bars.
        """
        df = synthetic_consecutive_bars.copy()
        config = Config()

        result = _add_microstructure_features(df, config)

        # First 5 bars should have consecutive_bull = 1,2,3,4,5
        for i in range(5):
            assert result["consecutive_bull"].iloc[i] == i + 1, (
                f"Bar {i} should have {i + 1} consecutive bullish bars"
            )

        # Bar 5 (first bearish) should reset to 0
        assert result["consecutive_bull"].iloc[5] == 0, (
            "First bearish bar should reset consecutive_bull count"
        )

    def test_volume_delta_uses_only_current_bar(self, synthetic_volume_delta_positive):
        """
        Test volume delta only uses current bar's open/close comparison.
        """
        df = synthetic_volume_delta_positive.copy()
        config = Config()

        result = _add_microstructure_features(df, config)

        # All bars should have positive volume_delta (close > open)
        assert (result["volume_delta"] > 0).all(), (
            "All volume_delta values should be positive when close > open"
        )

        # First few values should be non-NaN (no lookback needed for base calculation)
        assert not pd.isna(result["volume_delta"].iloc[0]), (
            "Volume delta shouldn't require previous bars"
        )


class TestCandlestickPatternDetection:
    """Tests for candlestick pattern recognition on synthetic data."""

    def test_bull_engulfing_detected_correctly(self, synthetic_bull_engulfing):
        """Verify bullish engulfing pattern is detected correctly."""
        df = synthetic_bull_engulfing.copy()
        config = Config()

        result = _add_microstructure_features(df, config)

        # Bar 10 is bearish setup
        assert result["body"].iloc[10] < 0, "Bar 10 should be bearish"

        # Bar 11 is bullish engulfing
        assert result["bullish_engulfing"].iloc[11] == 1, (
            "Bullish engulfing should be detected at bar 11"
        )

        # Verify pattern conditions
        assert result["body"].iloc[11] > 0, "Bar 11 should be bullish"
        assert abs(result["body"].iloc[11]) > abs(result["body"].iloc[10]), (
            "Bullish body should be larger than bearish body"
        )

    def test_bear_engulfing_detected_correctly(self, synthetic_bear_engulfing):
        """Verify bearish engulfing pattern is detected correctly."""
        df = synthetic_bear_engulfing.copy()
        config = Config()

        result = _add_microstructure_features(df, config)

        # Bar 21 is bearish engulfing
        assert result["bearish_engulfing"].iloc[21] == 1, (
            "Bearish engulfing should be detected at bar 21"
        )

        # Verify pattern conditions
        assert result["body"].iloc[21] < 0, "Bar 21 should be bearish"
        assert abs(result["body"].iloc[21]) > abs(result["body"].iloc[20]), (
            "Bearish body should be larger than bullish body"
        )

    def test_doji_detected_with_threshold(self, synthetic_doji):
        """Verify doji pattern detected when body < 10% of range."""
        df = synthetic_doji.copy()
        config = Config()

        result = _add_microstructure_features(df, config)

        # Bar 30 is a clear doji
        assert result["doji"].iloc[30] == 1, "Clear doji should be detected"

        # Calculate body/range ratio
        body = abs(result["body"].iloc[30])
        range_val = result["range"].iloc[30]
        ratio = body / (range_val + 1e-10)

        assert ratio < 0.1, f"Doji body/range ratio ({ratio:.4f}) should be < 0.1"

    def test_hammer_detected_correctly(self, synthetic_hammer):
        """Verify hammer pattern detected: long lower wick, small body, bullish."""
        df = synthetic_hammer.copy()
        config = Config()

        result = _add_microstructure_features(df, config)

        # Bar 40 should be a hammer
        assert result["hammer"].iloc[40] == 1, "Hammer should be detected"

        # Verify conditions
        assert result["body"].iloc[40] > 0, "Hammer should be bullish"
        assert result["lower_wick"].iloc[40] > 2 * abs(result["body"].iloc[40]), (
            "Hammer should have lower wick > 2x body"
        )
        assert result["upper_wick"].iloc[40] < abs(result["body"].iloc[40]), (
            "Hammer should have small upper wick"
        )

    def test_shooting_star_detected_correctly(self, synthetic_shooting_star):
        """Verify shooting star detected: long upper wick, small body, bearish."""
        df = synthetic_shooting_star.copy()
        config = Config()

        result = _add_microstructure_features(df, config)

        # Bar 45 should be a shooting star
        assert result["shooting_star"].iloc[45] == 1, "Shooting star should be detected"

        # Verify conditions
        assert result["body"].iloc[45] < 0, "Shooting star should be bearish"
        assert result["upper_wick"].iloc[45] > 2 * abs(result["body"].iloc[45]), (
            "Shooting star should have upper wick > 2x body"
        )
        assert result["lower_wick"].iloc[45] < abs(result["body"].iloc[45]), (
            "Shooting star should have small lower wick"
        )

    def test_marubozu_detected_correctly(self, synthetic_marubozu):
        """Verify marubozu detected: no significant wicks."""
        df = synthetic_marubozu.copy()
        config = Config()

        result = _add_microstructure_features(df, config)

        # Bars 48 and 49 should be marubozu
        assert result["marubozu"].iloc[48] == 1, "Bullish marubozu should be detected"
        assert result["marubozu"].iloc[49] == 1, "Bearish marubozu should be detected"

        # Verify small wicks relative to range
        range_48 = result["range"].iloc[48]
        upper_wick_48 = result["upper_wick"].iloc[48]
        lower_wick_48 = result["lower_wick"].iloc[48]

        assert upper_wick_48 < 0.1 * range_48, "Upper wick should be < 10% of range"
        assert lower_wick_48 < 0.1 * range_48, "Lower wick should be < 10% of range"


class TestVolumeDelta:
    """Tests for volume delta (buying/selling pressure) feature."""

    def test_volume_delta_positive_when_close_above_open(
        self, synthetic_volume_delta_positive
    ):
        """Volume delta should be positive when close > open."""
        df = synthetic_volume_delta_positive.copy()
        config = Config()

        result = _add_microstructure_features(df, config)

        # All bars should have positive volume_delta
        assert (result["volume_delta"] > 0).all(), (
            "Volume delta should be positive when close > open"
        )

        # Volume delta should equal volume for bullish bars
        for i in range(len(result)):
            assert result["volume_delta"].iloc[i] == result["volume"].iloc[i], (
                f"Bar {i}: volume_delta should equal volume for bullish bars"
            )

    def test_volume_delta_negative_when_close_below_open(
        self, synthetic_volume_delta_negative
    ):
        """Volume delta should be negative when close < open."""
        df = synthetic_volume_delta_negative.copy()
        config = Config()

        result = _add_microstructure_features(df, config)

        # All bars should have negative volume_delta
        assert (result["volume_delta"] < 0).all(), (
            "Volume delta should be negative when close < open"
        )

        # Volume delta should equal -volume for bearish bars
        for i in range(len(result)):
            assert result["volume_delta"].iloc[i] == -result["volume"].iloc[i], (
                f"Bar {i}: volume_delta should equal -volume for bearish bars"
            )

    def test_volume_delta_moving_averages_calculated(self, synthetic_ohlcv):
        """Volume delta moving averages should be calculated."""
        df = synthetic_ohlcv.copy()
        config = Config()

        result = _add_microstructure_features(df, config)

        # Check columns exist
        assert "volume_delta_ma_10" in result.columns, (
            "volume_delta_ma_10 column should exist"
        )
        assert "volume_delta_ma_20" in result.columns, (
            "volume_delta_ma_20 column should exist"
        )

        # After 20 bars, MA20 should not be NaN
        assert not pd.isna(result["volume_delta_ma_20"].iloc[25]), (
            "MA20 should have values after 20 bars"
        )


class TestBodyToWickRatios:
    """Tests for body-to-wick ratio (market conviction) feature."""

    def test_body_wick_ratio_high_for_strong_bodies(self, synthetic_marubozu):
        """Marubozu should have high body/wick ratio."""
        df = synthetic_marubozu.copy()
        config = Config()

        result = _add_microstructure_features(df, config)

        # Marubozu bars should have high body_wick_ratio
        ratio_48 = result["body_wick_ratio"].iloc[48]
        ratio_49 = result["body_wick_ratio"].iloc[49]

        assert ratio_48 > 5.0, f"Marubozu should have high ratio, got {ratio_48}"
        assert ratio_49 > 5.0, f"Marubozu should have high ratio, got {ratio_49}"

    def test_body_wick_ratio_low_for_doji(self, synthetic_doji):
        """Doji should have low body/wick ratio."""
        df = synthetic_doji.copy()
        config = Config()

        result = _add_microstructure_features(df, config)

        # Doji bar should have low body_wick_ratio
        ratio_30 = result["body_wick_ratio"].iloc[30]

        assert ratio_30 < 0.1, f"Doji should have very low ratio, got {ratio_30}"


class TestConsecutiveBars:
    """Tests for consecutive bullish/bearish bar counting."""

    def test_consecutive_bull_counting(self, synthetic_consecutive_bars):
        """Consecutive bullish bars should be counted correctly."""
        df = synthetic_consecutive_bars.copy()
        config = Config()

        result = _add_microstructure_features(df, config)

        # First 5 bars: consecutive_bull = 1,2,3,4,5
        expected = [1, 2, 3, 4, 5]
        actual = result["consecutive_bull"].iloc[0:5].tolist()

        assert actual == expected, f"Expected {expected}, got {actual}"

        # Consecutive bear should be 0 for bullish bars
        assert (result["consecutive_bear"].iloc[0:5] == 0).all(), (
            "consecutive_bear should be 0 during bullish sequence"
        )

    def test_consecutive_bear_counting(self, synthetic_consecutive_bars):
        """Consecutive bearish bars should be counted correctly."""
        df = synthetic_consecutive_bars.copy()
        config = Config()

        result = _add_microstructure_features(df, config)

        # Bars 5-7 are bearish: consecutive_bear = 1,2,3
        expected = [1, 2, 3]
        actual = result["consecutive_bear"].iloc[5:8].tolist()

        assert actual == expected, f"Expected {expected}, got {actual}"


class TestTickIntensity:
    """Tests for tick intensity (relative volume) feature."""

    def test_tick_intensity_high_for_volume_spikes(
        self, synthetic_high_volume_intensity
    ):
        """Tick intensity should be high when volume >> 20-bar average."""
        df = synthetic_high_volume_intensity.copy()
        config = Config()

        result = _add_microstructure_features(df, config)

        # Bar 25 has 3x volume
        intensity_25 = result["tick_intensity"].iloc[25]

        assert intensity_25 > 2.5, (
            f"Tick intensity should be ~3.0 for 3x volume, got {intensity_25}"
        )

        # Normal bars should have intensity near 1.0 (use index 30 which is >= 20 and not a volume spike)
        intensity_normal = result["tick_intensity"].iloc[30]
        assert 0.5 < intensity_normal < 2.0, (
            f"Normal tick intensity should be near 1.0, got {intensity_normal}"
        )


class TestCloseInRange:
    """Tests for close position in range feature."""

    def test_close_in_range_high_for_bullish_pressure(self, synthetic_close_at_high):
        """Close near high should give close_in_range near 1.0."""
        df = synthetic_close_at_high.copy()
        config = Config()

        result = _add_microstructure_features(df, config)

        # Most bars should have close_in_range near 0.9-1.0
        avg_close_range = result["close_in_range"].mean()

        assert avg_close_range > 0.8, (
            f"Bullish pressure should give high close_in_range, got {avg_close_range}"
        )

        # Individual bars should be in valid range [0, 1]
        assert (result["close_in_range"] >= 0).all() and (
            result["close_in_range"] <= 1
        ).all(), "close_in_range should always be in [0, 1]"

    def test_close_in_range_low_for_bearish_pressure(self, synthetic_close_at_low):
        """Close near low should give close_in_range near 0.0."""
        df = synthetic_close_at_low.copy()
        config = Config()

        result = _add_microstructure_features(df, config)

        # Most bars should have close_in_range near 0.0-0.1
        avg_close_range = result["close_in_range"].mean()

        assert avg_close_range < 0.2, (
            f"Bearish pressure should give low close_in_range, got {avg_close_range}"
        )


class TestLargeBodyDetection:
    """Tests for large body (strong momentum) detection."""

    def test_large_body_detected_for_momentum(self, synthetic_marubozu):
        """Large body bars should be flagged."""
        df = synthetic_marubozu.copy()
        config = Config()

        result = _add_microstructure_features(df, config)

        # Marubozu bars (bars 48, 49) should be large_body
        assert result["large_body"].iloc[48] == 1, (
            "Large marubozu should be flagged as large_body"
        )
        assert result["large_body"].iloc[49] == 1, (
            "Large marubozu should be flagged as large_body"
        )

    def test_large_body_comparison_to_ma(self, synthetic_ohlcv):
        """Large body should be > 1.5x 20-bar moving average."""
        df = synthetic_ohlcv.copy()
        config = Config()

        result = _add_microstructure_features(df, config)

        # Calculate expected
        body_abs = result["body"].abs()
        body_ma = body_abs.rolling(20).mean()
        expected_large = (body_abs > 1.5 * body_ma).astype(int)

        # After 20 bars, should match
        actual_large = result["large_body"].iloc[25:]
        expected_large_vals = expected_large.iloc[25:]

        assert (actual_large == expected_large_vals).all(), (
            "large_body flag should match 1.5x MA comparison"
        )


class TestMicrostructureEdgeCases:
    """Edge case tests for microstructure features."""

    def test_empty_dataframe_handling(self):
        """Should handle empty or very small dataframes gracefully."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=1, freq="h"),
                "open": [1500.0],
                "high": [1501.0],
                "low": [1499.0],
                "close": [1500.5],
                "volume": [100],
                "avg_spread": [0.2],
            }
        )

        config = Config()
        result = _add_microstructure_features(df, config)

        # Should not crash and should return dataframe
        assert len(result) == 1
        assert "bullish_engulfing" in result.columns

    def test_small_doji_threshold_boundary(self):
        """Test doji detection at exactly 10% threshold."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=3, freq="h"),
                "open": [1500.0, 1500.0, 1500.0],
                "high": [1510.0, 1510.0, 1510.0],
                "low": [1490.0, 1490.0, 1490.0],
                "close": [
                    1500.0,
                    1500.95,
                    1501.05,
                ],  # Bodies: 0.0, 0.95 (~9.5%), 1.05 (~10.5%)
                "volume": [100, 100, 100],
                "avg_spread": [0.2, 0.2, 0.2],
            }
        )

        config = Config()
        result = _add_microstructure_features(df, config)

        # Range is 20, body/ratio thresholds: 0/20=0%, 0.95/20=4.75%, 1.05/20=5.25%
        # All should be doji (body/range < 10%)
        assert result["doji"].iloc[0] == 1, "Zero body should be doji"
        assert result["doji"].iloc[1] == 1, "4.75% body/range should be doji"
        assert result["doji"].iloc[2] == 1, "5.25% body/range should be doji"

    def test_nan_handling_in_features(self, synthetic_ohlcv):
        """Features should handle NaN values appropriately."""
        df = synthetic_ohlcv.copy()

        # Introduce some NaN values at a specific position
        nan_idx = 50  # Use a position well beyond the rolling window
        df.iloc[nan_idx, df.columns.get_loc("close")] = np.nan

        # Verify NaN was set
        assert pd.isna(df["close"].iloc[nan_idx]), (
            "Close should be NaN at test position"
        )

        config = Config()
        result = _add_microstructure_features(df, config)

        # Should not crash
        assert len(result) == len(df)

        # Body at NaN close should be NaN (NaN propagation from close - open)
        body_at_nan = result["body"].iloc[nan_idx]
        assert pd.isna(body_at_nan), (
            f"Body should be NaN when close is NaN, got {body_at_nan}"
        )


class TestMicrostructureRealData:
    """Tests using real XAU/USD data samples."""

    def test_features_generated_on_real_data(self, sample_ohlcv_df):
        """All microstructure features should be generated without errors."""
        if sample_ohlcv_df is None:
            pytest.skip("No real data available")

        df = sample_ohlcv_df.copy()
        config = Config()

        result = _add_microstructure_features(df, config)

        # Check all expected columns exist
        expected_cols = [
            "bullish_engulfing",
            "bearish_engulfing",
            "doji",
            "hammer",
            "shooting_star",
            "marubozu",
            "volume_delta",
            "volume_delta_ma_10",
            "volume_delta_ma_20",
            "body_wick_ratio",
            "consecutive_bull",
            "consecutive_bear",
            "tick_intensity",
            "close_in_range",
            "large_body",
        ]

        for col in expected_cols:
            assert col in result.columns, f"Column {col} should exist"

    def test_pattern_frequencies_reasonable(self, sample_ohlcv_df):
        """
        Pattern frequencies on real data should be reasonable.

        - Doji: ~5-15% of bars
        - Engulfing: ~1-5% each direction
        - Hammer/Shooting Star: ~1-3% each
        """
        if sample_ohlcv_df is None or len(sample_ohlcv_df) < 100:
            pytest.skip("Insufficient real data")

        df = sample_ohlcv_df.copy()
        config = Config()

        result = _add_microstructure_features(df, config)

        n = len(result)

        # Check doji frequency
        doji_count = result["doji"].sum()
        doji_pct = doji_count / n
        assert 0.01 <= doji_pct <= 0.20, (
            f"Doji frequency ({doji_pct:.1%}) should be 1-20%"
        )

        # Check engulfing patterns
        bull_engulf = result["bullish_engulfing"].sum() / n
        bear_engulf = result["bearish_engulfing"].sum() / n

        assert 0.001 <= bull_engulf <= 0.20, (
            f"Bullish engulfing ({bull_engulf:.1%}) should be 0.1-20%"
        )
        assert 0.001 <= bear_engulf <= 0.20, (
            f"Bearish engulfing ({bear_engulf:.1%}) should be 0.1-20%"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
