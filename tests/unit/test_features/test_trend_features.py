"""Tests for zero-lag trend features: Donchian position, LSQ slope."""

import inspect

import numpy as np
import pandas as pd
import pytest


class TestDonchianPosition:
    """Tests for Donchian Channel position feature."""

    def test_donchian_position_at_high(self):
        """Price at 20-bar high should give position ≈ 1."""
        n = 30
        df = pd.DataFrame(
            {
                "close": np.concatenate([np.linspace(100, 120, 29), [120.0]]),
                "high": np.concatenate([np.linspace(100, 120, 29), [120.0]]),
                "low": np.linspace(95, 115, 30),
            }
        )

        donchian_high = df["high"].rolling(20).max()
        donchian_low = df["low"].rolling(20).min()
        position = (df["close"] - donchian_low) / (donchian_high - donchian_low + 1e-10)

        # Last bar: close == high == donchian_high → position should be ≈ 1
        assert position.iloc[-1] > 0.99, f"Expected ~1.0, got {position.iloc[-1]}"

    def test_donchian_position_at_low(self):
        """Price at 20-bar low should give position ≈ 0."""
        n = 30
        df = pd.DataFrame(
            {
                "close": np.concatenate([np.linspace(120, 100, 29), [95.0]]),
                "high": np.linspace(120, 100, 30),
                "low": np.concatenate([np.linspace(115, 95, 29), [95.0]]),
            }
        )

        donchian_high = df["high"].rolling(20).max()
        donchian_low = df["low"].rolling(20).min()
        position = (df["close"] - donchian_low) / (donchian_high - donchian_low + 1e-10)

        # Last bar: close == low == donchian_low → position should be ≈ 0
        assert position.iloc[-1] < 0.01, f"Expected ~0.0, got {position.iloc[-1]}"

    def test_donchian_position_range(self):
        """Donchian position should be in [0, 1] range."""
        rng = np.random.default_rng(42)
        n = 100
        df = pd.DataFrame(
            {
                "close": rng.normal(1900, 10, n),
                "high": rng.normal(1905, 10, n),
                "low": rng.normal(1895, 10, n),
            }
        )

        donchian_high = df["high"].rolling(20).max()
        donchian_low = df["low"].rolling(20).min()
        position = (df["close"] - donchian_low) / (donchian_high - donchian_low + 1e-10)

        # Skip NaN rows
        valid = position.dropna()
        assert (valid >= -0.01).all(), f"Min position: {valid.min()}"
        assert (valid <= 1.01).all(), f"Max position: {valid.max()}"


class TestLSQSlope:
    """Tests for linear regression slope feature."""

    def test_lsq_slope_positive_trend(self):
        """Upward trend should produce positive slope."""
        n = 40
        df = pd.DataFrame(
            {
                "close": np.linspace(100, 120, n),  # Linear upward
            }
        )

        slopes = _compute_lsq_slope(df["close"], window=20)

        # All slopes should be positive for a linear uptrend
        valid = slopes[~np.isnan(slopes)]
        assert (valid > 0).all(), f"Expected all positive, got min={valid.min()}"

    def test_lsq_slope_negative_trend(self):
        """Downward trend should produce negative slope."""
        n = 40
        df = pd.DataFrame(
            {
                "close": np.linspace(120, 100, n),  # Linear downward
            }
        )

        slopes = _compute_lsq_slope(df["close"], window=20)

        valid = slopes[~np.isnan(slopes)]
        assert (valid < 0).all(), f"Expected all negative, got max={valid.max()}"

    def test_lsq_slope_insufficient_data(self):
        """First 19 bars should have NaN slope."""
        n = 25
        df = pd.DataFrame(
            {
                "close": np.random.randn(n) + 100,
            }
        )

        slopes = _compute_lsq_slope(df["close"], window=20)

        # First 19 should be NaN
        assert np.all(np.isnan(slopes[:19])), "First 19 bars should be NaN"
        # Last 6 should have values
        assert not np.all(np.isnan(slopes[19:])), "Bars 20+ should have values"

    def test_lsq_slope_normalized_by_price(self):
        """Slope should be normalized by mean price level."""
        # Two trends with same slope but different price levels
        slopes_low = _compute_lsq_slope(pd.Series(np.linspace(10, 12, 40)), window=20)
        slopes_high = _compute_lsq_slope(
            pd.Series(np.linspace(1000, 1020, 40)), window=20
        )

        # After normalization by mean, slopes should be similar
        valid_low = slopes_low[~np.isnan(slopes_low)]
        valid_high = slopes_high[~np.isnan(slopes_high)]

        # Both should have similar normalized values (within an order of magnitude)
        ratio = np.mean(np.abs(valid_high)) / (np.mean(np.abs(valid_low)) + 1e-10)
        assert 0.1 < ratio < 10, f"Normalized slopes too different: ratio={ratio}"


class TestTrendFeaturesNoLookahead:
    """Meta-tests verifying no lookahead bias in trend features."""

    def test_no_shift_negative_one_in_trend_features(self):
        """Meta-test: _add_trend_features must not contain .shift(-1)."""
        try:
            from thesis.features.engineering import _add_trend_features

            source = inspect.getsource(_add_trend_features)
            assert ".shift(-1)" not in source, (
                "CRITICAL: _add_trend_features contains .shift(-1) — LOOKAHEAD BIAS!"
            )
        except ImportError:
            pytest.skip("thesis.features.engineering not available")


def _compute_lsq_slope(series: pd.Series, window: int = 20) -> np.ndarray:
    """Compute rolling linear regression slope (mirrors engineering.py implementation)."""
    from numpy.polynomial.polynomial import polyfit

    slopes = np.full(len(series), np.nan)
    x = np.arange(window)

    for i in range(window - 1, len(series)):
        y = series.iloc[i - window + 1 : i + 1].values.astype(float)
        if not np.any(np.isnan(y)):
            coeffs = polyfit(x, y, 1)
            slope = coeffs[1]
            slopes[i] = slope / (y.mean() + 1e-10)

    return slopes
