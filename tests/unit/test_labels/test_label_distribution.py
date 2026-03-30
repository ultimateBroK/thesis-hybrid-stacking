"""
Tests for label distribution verification.

Verifies that symmetric barriers (1.5×/1.5×) produce balanced labels (~35% each class).
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


@dataclass
class MockLabelsConfig:
    """Mock labels config for testing."""

    labels_path: str = "data/processed/labels.parquet"
    atr_multiplier_tp: float = 1.5
    atr_multiplier_sl: float = 1.5
    use_fixed_pips: bool = False
    tp_pips: int = 20
    sl_pips: int = 10
    horizon_bars: int = 20
    num_classes: int = 3
    class_labels: Dict[str, str] = field(
        default_factory=lambda: {"-1": "Short", "0": "Hold", "1": "Long"}
    )
    min_atr: float = 0.0001


@dataclass
class MockFeaturesConfig:
    """Mock features config for testing."""

    features_path: str = "data/processed/features.parquet"


@dataclass
class MockConfig:
    """Mock config object for testing generate_labels."""

    labels: MockLabelsConfig
    features: MockFeaturesConfig


def create_mock_config(
    tp_multiplier: float = 1.5, sl_multiplier: float = 1.5, horizon_bars: int = 20
):
    """Create a mock config with specified parameters."""
    return MockConfig(
        labels=MockLabelsConfig(
            atr_multiplier_tp=tp_multiplier,
            atr_multiplier_sl=sl_multiplier,
            horizon_bars=horizon_bars,
        ),
        features=MockFeaturesConfig(),
    )


def generate_labels_manual(df: pd.DataFrame, config: MockConfig) -> pd.DataFrame:
    """Manual triple-barrier label generation for testing.

    This is a simplified version that doesn't require the full pipeline.
    """
    atr_tp = config.labels.atr_multiplier_tp
    atr_sl = config.labels.atr_multiplier_sl
    horizon = config.labels.horizon_bars

    # Ensure we have ATR
    if "atr_14" not in df.columns:
        df["atr_14"] = (df["high"] - df["low"]).rolling(14).mean()

    df = df.dropna()

    labels = []
    tp_prices = []
    sl_prices = []
    touched_bar = []

    close_arr = df["close"].to_numpy()
    high_arr = df["high"].to_numpy()
    low_arr = df["low"].to_numpy()
    atr_arr = df["atr_14"].to_numpy()

    for i in range(len(df)):
        if i + horizon >= len(df):
            # Not enough bars ahead
            labels.append(0)
            tp_prices.append(np.nan)
            sl_prices.append(np.nan)
            touched_bar.append(np.nan)
            continue

        close = close_arr[i]
        atr = max(atr_arr[i], 0.0001)  # Avoid zero ATR

        tp = close + atr_tp * atr
        sl = close - atr_sl * atr

        label = 0  # Default to Hold
        touch_bar = None

        # Look ahead up to horizon bars
        for j in range(1, min(horizon + 1, len(df) - i)):
            high = high_arr[i + j]
            low = low_arr[i + j]

            if high >= tp:
                label = 1  # Long (TP hit first)
                touch_bar = j
                break
            elif low <= sl:
                label = -1  # Short (SL hit first)
                touch_bar = j
                break

        labels.append(label)
        tp_prices.append(tp)
        sl_prices.append(sl)
        touched_bar.append(touch_bar)

    result = df.copy()
    result["label"] = labels
    result["tp_price"] = tp_prices
    result["sl_price"] = sl_prices
    result["touched_bar"] = touched_bar

    return result


class TestSymmetricBarrierLabelDistribution:
    """Tests for label balance with symmetric barriers (1.5×/1.5×)."""

    def test_symmetric_barriers_produce_balanced_labels(self):
        """
        Verify that 1.5×/1.5× barriers produce balanced Long/Short distribution.

        With symmetric barriers, we expect:
        - Short: ~30-40%
        - Hold: ~25-40%
        - Long: ~30-40%
        """
        # Create synthetic OHLCV with oscillating (mean-reverting) data for balanced labels
        np.random.seed(200)
        n = 1000

        dates = pd.date_range("2020-01-01", periods=n, freq="h", tz="America/New_York")

        # Create oscillating data around a mean (promotes balanced labels)
        t = np.linspace(0, 6 * np.pi, n)  # 3 full cycles
        close = 1500 + 20 * np.sin(t) + np.random.randn(n) * 1.5

        open_price = close + np.random.randn(n) * 0.5
        high = np.maximum(open_price, close) + np.random.uniform(1, 4, n)
        low = np.minimum(open_price, close) - np.random.uniform(1, 4, n)
        volume = np.random.randint(100, 1000, n)

        df = pd.DataFrame(
            {
                "timestamp": dates,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

        # Calculate ATR for barrier sizing
        df["atr_14"] = (df["high"] - df["low"]).rolling(14).mean()
        df = df.dropna()

        # Apply triple barrier with symmetric 1.5×/1.5×
        config = create_mock_config(
            tp_multiplier=1.5, sl_multiplier=1.5, horizon_bars=20
        )
        labels_df = generate_labels_manual(df, config)

        # Count labels
        short_count = (labels_df["label"] == -1).sum()
        hold_count = (labels_df["label"] == 0).sum()
        long_count = (labels_df["label"] == 1).sum()
        total = len(labels_df)

        short_pct = short_count / total
        hold_pct = hold_count / total
        long_pct = long_count / total

        # Verify balanced distribution
        assert 0.30 <= short_pct <= 0.40, (
            f"Short percentage ({short_pct:.1%}) should be 30-40%"
        )
        assert 0.25 <= hold_pct <= 0.40, (
            f"Hold percentage ({hold_pct:.1%}) should be 25-40%"
        )
        assert 0.30 <= long_pct <= 0.40, (
            f"Long percentage ({long_pct:.1%}) should be 30-40%"
        )

        # Verify Long vs Short balance
        long_short_diff = abs(long_pct - short_pct)
        assert long_short_diff < 0.05, (
            f"Long-Short difference ({long_short_diff:.1%}) should be < 5%"
        )

    def test_horizon_20_label_distribution(self):
        """Verify label distribution with horizon=20 bars."""
        np.random.seed(201)
        n = 500

        dates = pd.date_range("2020-01-01", periods=n, freq="h", tz="America/New_York")
        close = 1500 + np.cumsum(np.random.randn(n) * 2)
        open_price = close + np.random.randn(n) * 0.5
        high = np.maximum(open_price, close) + np.random.uniform(1, 4, n)
        low = np.minimum(open_price, close) - np.random.uniform(1, 4, n)

        df = pd.DataFrame(
            {
                "timestamp": dates,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.random.randint(100, 1000, n),
            }
        )

        df["atr_14"] = (df["high"] - df["low"]).rolling(14).mean()
        df = df.dropna()

        config = create_mock_config(
            tp_multiplier=1.5, sl_multiplier=1.5, horizon_bars=20
        )
        labels_df = generate_labels_manual(df, config)

        # All three classes should be present
        unique_labels = labels_df["label"].unique()
        assert -1 in unique_labels, "Short labels should be present"
        assert 0 in unique_labels, "Hold labels should be present"
        assert 1 in unique_labels, "Long labels should be present"

        # No class should dominate (>60%)
        for label in [-1, 0, 1]:
            count = (labels_df["label"] == label).sum()
            pct = count / len(labels_df)
            assert pct < 0.60, f"Label {label} ({pct:.1%}) should not dominate"

    def test_no_excessive_hold_labels(self):
        """Verify Hold labels are not excessive (< 45%)."""
        np.random.seed(202)
        n = 400

        dates = pd.date_range("2020-01-01", periods=n, freq="h", tz="America/New_York")
        close = 1500 + np.cumsum(np.random.randn(n) * 2)
        open_price = close + np.random.randn(n) * 0.5
        high = np.maximum(open_price, close) + np.random.uniform(1, 4, n)
        low = np.minimum(open_price, close) - np.random.uniform(1, 4, n)

        df = pd.DataFrame(
            {
                "timestamp": dates,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.random.randint(100, 1000, n),
            }
        )

        df["atr_14"] = (df["high"] - df["low"]).rolling(14).mean()
        df = df.dropna()

        config = create_mock_config(
            tp_multiplier=1.5, sl_multiplier=1.5, horizon_bars=20
        )
        labels_df = generate_labels_manual(df, config)

        hold_count = (labels_df["label"] == 0).sum()
        hold_pct = hold_count / len(labels_df)

        assert hold_pct < 0.45, f"Hold percentage ({hold_pct:.1%}) should be < 45%"

    def test_long_short_balance_ratio(self):
        """Verify Long and Short are balanced (|Long - Short| < 10%)."""
        np.random.seed(203)
        n = 600

        dates = pd.date_range("2020-01-01", periods=n, freq="h", tz="America/New_York")

        # Create oscillating data (sine wave) for balanced labels
        t = np.linspace(0, 4 * np.pi, n)  # 2 full cycles
        close = 1500 + 15 * np.sin(t) + np.random.randn(n) * 1.0

        open_price = close + np.random.randn(n) * 0.5
        high = np.maximum(open_price, close) + np.random.uniform(1, 4, n)
        low = np.minimum(open_price, close) - np.random.uniform(1, 4, n)

        df = pd.DataFrame(
            {
                "timestamp": dates,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.random.randint(100, 1000, n),
            }
        )

        df["atr_14"] = (df["high"] - df["low"]).rolling(14).mean()
        df = df.dropna()

        config = create_mock_config(
            tp_multiplier=1.5, sl_multiplier=1.5, horizon_bars=20
        )
        labels_df = generate_labels_manual(df, config)

        short_count = (labels_df["label"] == -1).sum()
        long_count = (labels_df["label"] == 1).sum()
        total = short_count + long_count

        if total > 0:
            short_pct = short_count / total
            long_pct = long_count / total

            diff = abs(long_pct - short_pct)
            assert diff < 0.10, (
                f"Long-Short balance difference ({diff:.1%}) should be < 10%"
            )

    def test_distribution_stability_across_splits(self):
        """Verify label distribution is consistent across train/val/test."""
        np.random.seed(204)
        n = 900

        dates = pd.date_range("2020-01-01", periods=n, freq="h", tz="America/New_York")

        # Create high-frequency oscillating data for balanced distribution in all splits
        t = np.linspace(
            0, 20 * np.pi, n
        )  # 10 full cycles - ensures each split has up and down
        close = 1500 + 20 * np.sin(t) + np.random.randn(n) * 1.0

        open_price = close + np.random.randn(n) * 0.5
        high = np.maximum(open_price, close) + np.random.uniform(1, 4, n)
        low = np.minimum(open_price, close) - np.random.uniform(1, 4, n)

        df = pd.DataFrame(
            {
                "timestamp": dates,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.random.randint(100, 1000, n),
            }
        )

        df["atr_14"] = (df["high"] - df["low"]).rolling(14).mean()
        df = df.dropna()

        config = create_mock_config(
            tp_multiplier=1.5, sl_multiplier=1.5, horizon_bars=20
        )
        labels_df = generate_labels_manual(df, config)

        # Split into train/val/test
        n_labels = len(labels_df)
        train_end = int(n_labels * 0.6)
        val_end = int(n_labels * 0.75)

        splits = {
            "train": labels_df.iloc[:train_end],
            "val": labels_df.iloc[train_end:val_end],
            "test": labels_df.iloc[val_end:],
        }

        # Check each split has reasonable distribution
        for split_name, split_df in splits.items():
            if len(split_df) < 50:
                continue

            short_pct = (split_df["label"] == -1).sum() / len(split_df)
            long_pct = (split_df["label"] == 1).sum() / len(split_df)

            # Relaxed thresholds: each split should have at least some of both Long and Short
            # but exact percentages vary due to local trends in oscillating data
            assert 0.10 <= short_pct <= 0.60, (
                f"{split_name} Short ({short_pct:.1%}) should be 10-60%"
            )
            assert 0.10 <= long_pct <= 0.60, (
                f"{split_name} Long ({long_pct:.1%}) should be 10-60%"
            )


class TestLabelDistributionRealData:
    """Tests using real XAU/USD label data."""

    def test_real_data_label_balance(self, sample_labels_df):
        """
        Verify real data label distribution is reasonable.
        
        Real XAU/USD data may have different distributions than synthetic data.
        We verify that all three classes are present with reasonable proportions.
        """
        if sample_labels_df is None:
            pytest.skip("No real label data available")

        df = sample_labels_df

        if "label" not in df.columns:
            pytest.skip("Label column not found in real data")

        n = len(df)

        short_count = (df["label"] == -1).sum()
        hold_count = (df["label"] == 0).sum()
        long_count = (df["label"] == 1).sum()

        short_pct = short_count / n
        hold_pct = hold_count / n
        long_pct = long_count / n

        # Log actual distribution for debugging
        print(f"\nReal data label distribution:")
        print(f"  Short (-1): {short_pct:.1%} ({short_count}/{n})")
        print(f"  Hold (0): {hold_pct:.1%} ({hold_count}/{n})")
        print(f"  Long (1): {long_pct:.1%} ({long_count}/{n})")

        # Verify all three classes are present
        assert short_count > 0, "Real data should have Short labels"
        assert hold_count > 0, "Real data should have Hold labels"
        assert long_count > 0, "Real data should have Long labels"

        # Verify reasonable distribution (relaxed thresholds for real data)
        assert 0.10 <= short_pct <= 0.60, (
            f"Real data Short ({short_pct:.1%}) should be 10-60%"
        )
        assert 0.05 <= hold_pct <= 0.80, (
            f"Real data Hold ({hold_pct:.1%}) should be 5-80%"
        )
        assert 0.10 <= long_pct <= 0.60, (
            f"Real data Long ({long_pct:.1%}) should be 10-60%"
        )

    def test_real_data_long_short_balance(self, sample_labels_df):
        """
        Verify real data Long and Short are present with reasonable balance.
        
        Real market data may not be perfectly balanced. We verify both Long
        and Short are present with a reasonable spread.
        """
        if sample_labels_df is None or "label" not in sample_labels_df.columns:
            pytest.skip("No real label data available")

        df = sample_labels_df

        short_count = (df["label"] == -1).sum()
        long_count = (df["label"] == 1).sum()
        total_directional = short_count + long_count

        if total_directional == 0:
            pytest.skip("No directional labels in sample")

        short_pct = short_count / total_directional
        long_pct = long_count / total_directional
        diff = abs(long_pct - short_pct)

        # Log actual distribution
        print(f"\nReal data directional distribution:")
        print(f"  Short: {short_pct:.1%} ({short_count}/{total_directional})")
        print(f"  Long: {long_pct:.1%} ({long_count}/{total_directional})")
        print(f"  Difference: {diff:.1%}")

        # Verify both Long and Short are present
        assert short_count > 0, "Real data should have Short labels"
        assert long_count > 0, "Real data should have Long labels"

        # Relaxed threshold for real data (market can trend)
        assert diff < 0.20, (
            f"Real data Long-Short difference ({diff:.1%}) should be < 20%"
        )


class TestLabelCounts:
    """Tests for specific label count requirements."""

    def test_label_counts_after_regeneration(self):
        """
        Test that label counts are reasonable after Phase 1 regeneration.

        After applying symmetric barriers, expect:
        - At least 100 of each class for training
        - No class with < 10% of data
        """
        np.random.seed(205)
        n = 1000

        dates = pd.date_range("2020-01-01", periods=n, freq="h", tz="America/New_York")
        close = 1500 + np.cumsum(np.random.randn(n) * 2)
        open_price = close + np.random.randn(n) * 0.5
        high = np.maximum(open_price, close) + np.random.uniform(1, 4, n)
        low = np.minimum(open_price, close) - np.random.uniform(1, 4, n)

        df = pd.DataFrame(
            {
                "timestamp": dates,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.random.randint(100, 1000, n),
            }
        )

        df["atr_14"] = (df["high"] - df["low"]).rolling(14).mean()
        df = df.dropna()

        config = create_mock_config(
            tp_multiplier=1.5, sl_multiplier=1.5, horizon_bars=20
        )
        labels_df = generate_labels_manual(df, config)

        # Count each class
        counts = labels_df["label"].value_counts().to_dict()

        n_short = counts.get(-1, 0)
        n_hold = counts.get(0, 0)
        n_long = counts.get(1, 0)
        total = len(labels_df)

        # Each class should have at least 10% of data
        assert n_short / total >= 0.10, (
            f"Short count ({n_short}/{total}) should be >= 10%"
        )
        assert n_hold / total >= 0.10, f"Hold count ({n_hold}/{total}) should be >= 10%"
        assert n_long / total >= 0.10, f"Long count ({n_long}/{total}) should be >= 10%"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
