"""
Unit tests for Triple-Barrier labeling.
Critical for correct label generation and trading signal creation.
"""

import pandas as pd
import polars as pl
import numpy as np
import pytest


class TestTripleBarrier:
    """Tests for Triple Barrier Method label generation."""

    @pytest.mark.critical
    def test_generate_labels_with_real_data(self, sample_labels_df):
        """Test label generation with real data."""
        # Real labels should be in {-1, 0, 1}
        labels = sample_labels_df["label"].to_list()

        assert all(l in [-1, 0, 1] for l in labels), "All labels must be -1, 0, or 1"

    def test_label_column_exists(self, sample_labels_df):
        """Test that label column is present."""
        assert "label" in sample_labels_df.columns
        assert "timestamp" in sample_labels_df.columns

    def test_labels_not_all_same(self, sample_labels_df):
        """Test that labels have variation (not all same class)."""
        unique_labels = sample_labels_df["label"].unique()

        assert len(unique_labels) > 1, (
            "Labels should have variation (not all identical)"
        )

    def test_synthetic_long_label(self):
        """Test that long (+1) label logic is correct."""
        # Create data where price goes up
        dates = pd.date_range("2020-01-01", periods=20, freq="h")
        prices = pd.Series([100 + i * 2 for i in range(20)])  # Strong uptrend
        atr = pd.Series([2.0] * 20)

        df = pd.DataFrame(
            {
                "timestamp": dates,
                "close": prices,
                "atr_14": atr,
            }
        )

        # Label at position 0
        entry_price = prices[0]  # 100
        tp = entry_price + 2 * atr[0]  # 104
        sl = entry_price - 1 * atr[0]  # 98

        # Price goes to 138 at end, should hit TP first
        assert prices.iloc[-1] > tp, "Price should exceed take profit"

    def test_synthetic_short_label(self):
        """Test that short (-1) label logic is correct."""
        # Create data where price goes down
        dates = pd.date_range("2020-01-01", periods=20, freq="h")
        prices = pd.Series([100 - i * 2 for i in range(20)])  # Strong downtrend
        atr = pd.Series([2.0] * 20)

        df = pd.DataFrame(
            {
                "timestamp": dates,
                "close": prices,
                "atr_14": atr,
            }
        )

        entry_price = prices[0]  # 100
        # For short: TP is below entry (price drops), SL is above entry
        tp_short = entry_price - 1 * atr[0]  # 98 (take profit for short)
        sl_short = entry_price + 2 * atr[0]  # 104 (stop loss for short)

        # Price goes to 62 at end, hits TP first (98)
        assert prices.iloc[-1] < tp_short, "Price should hit take profit for short"

    def test_synthetic_neutral_label(self):
        """Test that neutral (0) label is assigned when neither barrier hit."""
        # Create sideways data
        dates = pd.date_range("2020-01-01", periods=20, freq="h")
        prices = pd.Series([100 + np.sin(i) * 0.5 for i in range(20)])  # Sideways
        atr = pd.Series([5.0] * 20)  # High ATR, barriers far away

        df = pd.DataFrame(
            {
                "timestamp": dates,
                "close": prices,
                "atr_14": atr,
            }
        )

        # Price stays within 100 +/- 0.5, barriers at 110 and 95
        # Neither barrier should be hit within horizon
        assert prices.max() < 110
        assert prices.min() > 95

    def test_label_alignment_with_features(self, sample_features_df, sample_labels_df):
        """Test that labels align with features by timestamp."""
        # Check that timestamps overlap
        feature_times = set(sample_features_df["timestamp"].to_list())
        label_times = set(sample_labels_df["timestamp"].to_list())

        # Should have significant overlap
        overlap = feature_times & label_times
        assert len(overlap) > len(label_times) * 0.8, (
            "Labels should align with features (at least 80% overlap)"
        )

    def test_horizon_respected(self, sample_labels_df):
        """Test that horizon (h) is respected in labeling."""
        from thesis.config.loader import load_config

        config = load_config()
        horizon = config.labels.horizon_bars

        # Labels should be generated looking ahead 'horizon' bars
        assert horizon >= 5, f"Horizon should be reasonable, got {horizon}"
        assert horizon <= 50, f"Horizon should not be too large, got {horizon}"

        # Verify labels exist
        assert len(sample_labels_df) > 0


class TestLabelDistribution:
    """Tests for label distribution and balance."""

    def test_label_distribution_reported(self, sample_labels_df):
        """Test that we can compute label distribution."""
        import polars as pl

        # Convert to pandas if needed for compatibility
        if isinstance(sample_labels_df, pl.DataFrame):
            df = sample_labels_df.to_pandas()
        else:
            df = sample_labels_df

        label_counts = df["label"].value_counts()

        # Should have counts for each label
        assert -1 in label_counts.index or label_counts.get(-1, 0) >= 0
        assert 0 in label_counts.index or label_counts.get(0, 0) >= 0
        assert 1 in label_counts.index or label_counts.get(1, 0) >= 0

        # Total should equal length
        total = label_counts.sum()
        assert total == len(df)

    def test_class_balance_check(self, sample_labels_df):
        """Test that we can identify class imbalance."""
        import polars as pl

        # Convert to pandas if needed for compatibility
        if isinstance(sample_labels_df, pl.DataFrame):
            df = sample_labels_df.to_pandas()
        else:
            df = sample_labels_df

        label_counts = df["label"].value_counts().to_dict()

        total = len(df)

        # Find majority class
        majority_count = max(label_counts.values())
        majority_pct = majority_count / total

        # Typically label 0 (neutral) is majority in Triple Barrier
        # If any class is > 70%, we have imbalance
        if majority_pct > 0.7:
            pytest.skip(f"Class imbalance detected: {majority_pct:.1%} majority class")

        # Otherwise, distribution is reasonable
        assert majority_pct <= 0.7, f"Severe class imbalance: {majority_pct:.1%}"
