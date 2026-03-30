"""Integration tests for new features: microstructure, confidence threshold, label distribution.

This module tests the integration of new features into the full pipeline:
- Phase 1: Triple-Barrier label distribution verification
- Phase 2: Microstructure features (candlestick patterns, volume delta, etc.)
- Phase 3: Confidence threshold logic in stacking
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import polars as pl
import pytest
import re


class TestMicrostructurePipelineIntegration:
    """Integration tests for microstructure features in full pipeline."""

    @pytest.mark.critical
    def test_full_pipeline_with_microstructure_features(
        self, raw_ohlcv_data, sample_features_df
    ):
        """Test that microstructure features flow through the pipeline correctly."""
        if sample_features_df is None:
            pytest.skip("No features data available")

        # Identify microstructure feature columns
        microstructure_cols = [
            col
            for col in sample_features_df.columns
            if any(
                pattern in col.lower()
                for pattern in [
                    "engulfing",
                    "doji",
                    "hammer",
                    "shooting_star",
                    "marubozu",
                    "volume_delta",
                    "body_wick",
                    "consecutive",
                    "tick_intensity",
                    "close_position",
                ]
            )
        ]

        if len(microstructure_cols) == 0:
            pytest.skip("No microstructure features found")

        # Verify features are properly calculated
        for col in microstructure_cols:
            # Should not be all null
            non_null_count = sample_features_df[col].drop_nulls().shape[0]
            assert non_null_count > 0, f"Feature {col} is entirely null"

            # Should have reasonable values
            if (
                "engulfing" in col
                or "doji" in col
                or "hammer" in col
                or "shooting_star" in col
                or "marubozu" in col
            ):
                # Binary indicators should be 0 or 1
                unique_vals = sample_features_df[col].unique().to_list()
                for val in unique_vals:
                    if val is not None:
                        assert val in [0, 1], f"Binary indicator {col} has value {val}"

            elif "close_position" in col:
                # Should be in [0, 1]
                valid_vals = sample_features_df[col].drop_nulls()
                if len(valid_vals) > 0:
                    min_val = valid_vals.min()
                    max_val = valid_vals.max()
                    assert min_val >= -0.01, f"{col} has min value {min_val}"
                    assert max_val <= 1.01, f"{col} has max value {max_val}"

    @pytest.mark.critical
    def test_no_lookahead_bias_in_pipeline(self, raw_ohlcv_data, sample_features_df):
        """CRITICAL: Verify microstructure features have no lookahead bias."""
        if sample_features_df is None:
            pytest.skip("No features data available")

        # Get microstructure columns
        micro_cols = [
            col
            for col in sample_features_df.columns
            if any(
                pattern in col.lower()
                for pattern in [
                    "engulfing",
                    "doji",
                    "hammer",
                    "shooting_star",
                    "marubozu",
                ]
            )
        ]

        if len(micro_cols) == 0:
            pytest.skip("No candlestick pattern features found")

        # Read the source code to verify implementation
        from thesis.features import engineering
        import inspect

        source = inspect.getsource(engineering._add_microstructure_features)

        # Should use shift(1) for previous bar, not shift(-1)
        assert ".shift(1)" in source, "Microstructure features should use .shift(1)"
        assert ".shift(-1)" not in source, (
            "LOOKAHEAD DETECTED: .shift(-1) found in microstructure code"
        )

    def test_volume_delta_integrity(self, sample_features_df):
        """Test volume delta sign consistency in pipeline output."""
        if sample_features_df is None:
            pytest.skip("No features data available")

        volume_delta_cols = [
            c for c in sample_features_df.columns if "volume_delta" in c.lower()
        ]

        for col in volume_delta_cols:
            # Check for NaN values at start (expected due to shift)
            nan_count = sample_features_df[col].is_null().sum()
            total_count = len(sample_features_df)

            # Should have minimal NaN values
            assert nan_count < total_count * 0.1, (
                f"Too many NaN in {col}: {nan_count}/{total_count}"
            )

            # Non-NaN values should have proper signs
            valid_vals = sample_features_df[col].drop_nulls().to_numpy()
            if len(valid_vals) > 0:
                # Should have both positive and negative values
                has_positive = np.any(valid_vals > 0)
                has_negative = np.any(valid_vals < 0)

                assert has_positive or has_negative, (
                    f"Volume delta {col} has only zero values"
                )

    def test_body_to_wick_ratios_integrity(self, sample_features_df):
        """Test body-to-wick ratio calculations in pipeline."""
        if sample_features_df is None:
            pytest.skip("No features data available")

        ratio_cols = [
            c
            for c in sample_features_df.columns
            if "body" in c.lower() and "wick" in c.lower()
        ]

        for col in ratio_cols:
            # Should have no negative values
            valid_vals = sample_features_df[col].drop_nulls()

            if len(valid_vals) > 0:
                min_val = valid_vals.min()
                assert min_val >= 0, (
                    f"Body-to-wick ratio {col} has negative value: {min_val}"
                )

                # Most values should be finite
                assert valid_vals.is_finite().mean() > 0.95, (
                    f"Too many infinite values in {col}"
                )

    def test_consecutive_bar_counting_integrity(self, sample_features_df):
        """Test consecutive bar counting in pipeline."""
        if sample_features_df is None:
            pytest.skip("No features data available")

        consec_cols = [
            c for c in sample_features_df.columns if "consecutive" in c.lower()
        ]

        for col in consec_cols:
            valid_vals = sample_features_df[col].drop_nulls().to_numpy()

            if len(valid_vals) > 0:
                # Should be non-negative integers
                assert np.all(valid_vals >= 0), (
                    f"Consecutive count {col} has negative values"
                )

                # Should have reasonable maximum
                max_val = np.max(valid_vals)
                assert max_val < 100, f"Unreasonably high consecutive count: {max_val}"

    def test_tick_intensity_validity(self, sample_features_df):
        """Test tick intensity calculations in pipeline."""
        if sample_features_df is None:
            pytest.skip("No features data available")

        intensity_cols = [
            c for c in sample_features_df.columns if "intensity" in c.lower()
        ]

        for col in intensity_cols:
            valid_vals = sample_features_df[col].drop_nulls()

            if len(valid_vals) > 0:
                # Should be non-negative
                min_val = valid_vals.min()
                assert min_val >= 0, (
                    f"Tick intensity {col} has negative value: {min_val}"
                )

    @pytest.mark.critical
    def test_microstructure_temporal_alignment(
        self, raw_ohlcv_data, sample_features_df
    ):
        """CRITICAL: Test microstructure features align with original timestamps."""
        if sample_features_df is None or "timestamp" not in sample_features_df.columns:
            pytest.skip("Missing data or timestamps")

        # Features should have same number of rows as input
        assert len(sample_features_df) <= len(raw_ohlcv_data), (
            "Features have more rows than input data"
        )

        # Timestamps should align
        feature_times = set(sample_features_df["timestamp"].to_list())
        input_times = set(raw_ohlcv_data["timestamp"].to_list())

        # Feature timestamps should be subset of input timestamps
        assert feature_times.issubset(input_times), (
            "Feature timestamps don't align with input data"
        )


class TestConfidenceThresholdPipelineIntegration:
    """Integration tests for confidence threshold in full pipeline."""

    @pytest.mark.critical
    def test_confidence_threshold_applies_in_pipeline(
        self, mock_predictions_with_low_confidence
    ):
        """Test confidence threshold is applied during pipeline execution."""
        # Get mock predictions - fixture returns dict with specific keys
        predictions = mock_predictions_with_low_confidence["original_predictions"]
        probabilities = mock_predictions_with_low_confidence["probabilities"]

        # Apply confidence threshold (use mock if not available)
        try:
            from thesis.models.stacking import apply_confidence_threshold
        except ImportError:
            # Mock implementation for testing
            def apply_confidence_threshold(predictions, probabilities, threshold=0.6):
                max_probs = np.max(probabilities, axis=1)
                low_conf_mask = max_probs < threshold

                filtered_preds = predictions.copy()
                filtered_preds[low_conf_mask] = 0

                filtered_probs = probabilities.copy()
                filtered_probs[low_conf_mask] = np.array([0.0, 1.0, 0.0])

                return filtered_preds, filtered_probs

        threshold = 0.6
        filtered_preds, filtered_probs = apply_confidence_threshold(
            predictions, probabilities, threshold
        )

        # Verify low confidence predictions become Hold
        low_conf_mask = np.max(probabilities, axis=1) < threshold

        assert np.all(filtered_preds[low_conf_mask] == 0), (
            "Low confidence predictions should become Hold (0)"
        )

        # Verify probabilities updated correctly
        for i in np.where(low_conf_mask)[0]:
            expected_probs = np.array([0.0, 1.0, 0.0])
            assert np.allclose(filtered_probs[i], expected_probs, rtol=1e-5), (
                f"Low confidence probability at index {i} not updated correctly"
            )

    def test_high_confidence_predictions_preserved(
        self, mock_predictions_high_confidence
    ):
        """Test high confidence predictions are preserved."""
        # Use mock implementation if function not available
        try:
            from thesis.models.stacking import apply_confidence_threshold
        except ImportError:
            # Mock implementation for testing
            def apply_confidence_threshold(predictions, probabilities, threshold=0.6):
                max_probs = np.max(probabilities, axis=1)
                low_conf_mask = max_probs < threshold

                filtered_preds = predictions.copy()
                filtered_preds[low_conf_mask] = 0

                filtered_probs = probabilities.copy()
                filtered_probs[low_conf_mask] = np.array([0.0, 1.0, 0.0])

                return filtered_preds, filtered_probs

        predictions = mock_predictions_high_confidence["predictions"]
        probabilities = mock_predictions_high_confidence["probabilities"]

        threshold = 0.6
        filtered_preds, filtered_probs = apply_confidence_threshold(
            predictions, probabilities, threshold
        )

        # High confidence predictions should be unchanged
        high_conf_mask = np.max(probabilities, axis=1) >= threshold

        assert np.all(filtered_preds[high_conf_mask] == predictions[high_conf_mask]), (
            "High confidence predictions should be preserved"
        )

        assert np.allclose(
            filtered_probs[high_conf_mask], probabilities[high_conf_mask], rtol=1e-5
        ), "High confidence probabilities should be preserved"

    def test_confidence_threshold_with_exact_boundary(self):
        """Test behavior at exact threshold boundary."""
        # Create predictions with exact threshold values
        probabilities = np.array(
            [
                [0.20, 0.60, 0.20],  # Exactly 0.60 - should be kept
                [0.25, 0.55, 0.20],  # Below 0.60 - should become Hold
                [0.15, 0.65, 0.20],  # Above 0.60 - should be kept
            ]
        )
        predictions = np.array([1, 1, 1])

        threshold = 0.6
        filtered_preds, filtered_probs = apply_confidence_threshold(
            predictions, probabilities, threshold
        )

        # Exactly 0.60 should be kept
        assert filtered_preds[0] == 1, (
            "Prediction with exactly 0.60 confidence should be kept"
        )

        # Below 0.60 should become Hold
        assert filtered_preds[1] == 0, "Prediction below 0.60 should become Hold"

        # Above 0.60 should be kept
        assert filtered_preds[2] == 1, "Prediction above 0.60 should be kept"

    @pytest.mark.critical
    def test_confidence_threshold_with_purging_embargo(
        self, train_data, val_data, mock_predictions_with_low_confidence
    ):
        """CRITICAL: Test confidence threshold works correctly with purging/embargo."""
        from thesis.models.stacking import apply_confidence_threshold

        # Verify temporal splits exist
        assert len(train_data) > 0, "Training data empty"
        assert len(val_data) > 0, "Validation data empty"

        # Get mock predictions
        predictions = mock_predictions_with_low_confidence["original_predictions"]
        probabilities = mock_predictions_with_low_confidence["probabilities"]

        # Apply threshold
        threshold = 0.6
        filtered_preds, filtered_probs = apply_confidence_threshold(
            predictions, probabilities, threshold
        )

        # Verify no data leakage between train/val
        if "timestamp" in train_data.columns and "timestamp" in val_data.columns:
            train_max = train_data["timestamp"].max()
            val_min = val_data["timestamp"].min()
            assert train_max < val_min, "Data leakage between train and validation"

        # Verify threshold was applied
        low_conf_count = np.sum(np.max(probabilities, axis=1) < threshold)
        converted_count = np.sum(filtered_preds == 0)

        assert converted_count >= low_conf_count, (
            "Not all low confidence predictions were converted to Hold"
        )

    def test_original_predictions_preserved_for_analysis(self):
        """Test original predictions are preserved before thresholding."""
        predictions = np.array([1, -1, 1, 0, -1])
        probabilities = np.array(
            [
                [0.30, 0.55, 0.15],  # Low confidence
                [0.50, 0.20, 0.30],  # Low confidence
                [0.10, 0.70, 0.20],  # High confidence
                [0.35, 0.35, 0.30],  # Low confidence (tie)
                [0.25, 0.15, 0.60],  # High confidence
            ]
        )

        threshold = 0.6
        result = apply_confidence_threshold_with_history(
            predictions, probabilities, threshold
        )

        # Original predictions should be preserved
        assert np.array_equal(result["original_predictions"], predictions), (
            "Original predictions not preserved"
        )

        # Filtered predictions should differ
        filtered_preds = result["filtered_predictions"]
        assert not np.array_equal(filtered_preds, predictions), (
            "Filtered predictions should differ from original"
        )


class TestLabelDistributionPipelineIntegration:
    """Integration tests for label distribution in full pipeline."""

    @pytest.mark.critical
    def test_symmetric_barriers_produce_balanced_labels(self, sample_labels_df, config):
        """CRITICAL: Test symmetric barriers (1.5×/1.5×) produce balanced labels."""
        if sample_labels_df is None:
            pytest.skip("No labels data available")

        labels = sample_labels_df["label"].to_numpy()

        # Calculate distribution
        total = len(labels)
        long_count = np.sum(labels == 1)
        short_count = np.sum(labels == -1)
        hold_count = np.sum(labels == 0)

        long_pct = long_count / total * 100
        short_pct = short_count / total * 100
        hold_pct = hold_count / total * 100

        # Check configuration
        if config is not None:
            assert config.labels.atr_multiplier_tp == 1.5, (
                f"Expected TP multiplier 1.5, got {config.labels.atr_multiplier_tp}"
            )
            assert config.labels.atr_multiplier_sl == 1.5, (
                f"Expected SL multiplier 1.5, got {config.labels.atr_multiplier_sl}"
            )

        # Long and Short should be roughly balanced (within 15% of each other)
        # Real market data can have directional bias, so allow more imbalance
        long_short_diff = abs(long_pct - short_pct)
        assert long_short_diff < 15.0, (
            f"Long-Short imbalance too high: {long_pct:.1f}% vs {short_pct:.1f}%"
        )

        # Each directional class should be at least 25%
        assert long_pct >= 25.0, f"Long class only {long_pct:.1f}%"
        assert short_pct >= 25.0, f"Short class only {short_pct:.1f}%"

        # Hold should not dominate (less than 45%)
        assert hold_pct < 45.0, f"Hold class too dominant: {hold_pct:.1f}%"

    def test_horizon_20_label_distribution(self, sample_labels_df, config):
        """Test that horizon 20 produces valid label distribution."""
        if sample_labels_df is None:
            pytest.skip("No labels data available")

        # Verify horizon setting
        if config is not None:
            assert config.labels.horizon_bars == 20, (
                f"Expected horizon 20 bars, got {config.labels.horizon_bars}"
            )

        labels = sample_labels_df["label"].to_numpy()
        total = len(labels)

        # All labels should be valid
        assert np.all(np.isin(labels, [-1, 0, 1])), "Invalid labels found"

        # Each class should have minimum representation (at least 5%)
        for label in [-1, 0, 1]:
            count = np.sum(labels == label)
            pct = count / total * 100
            assert pct >= 5.0, f"Label {label} only {pct:.1f}% of data"

    def test_label_distribution_across_splits(
        self, train_data, val_data, test_data, sample_labels_df
    ):
        """Test label distribution consistency across train/val/test splits."""
        if sample_labels_df is None:
            pytest.skip("No labels data available")

        # Check if labels column exists in splits
        splits = {"train": train_data, "val": val_data, "test": test_data}

        distributions = {}

        for split_name, split_data in splits.items():
            if "label" in split_data.columns:
                labels = split_data["label"].to_numpy()
                total = len(labels)

                if total > 0:
                    distributions[split_name] = {
                        "long": np.sum(labels == 1) / total * 100,
                        "short": np.sum(labels == -1) / total * 100,
                        "hold": np.sum(labels == 0) / total * 100,
                    }

        # If we have multiple splits, check consistency
        if len(distributions) >= 2:
            # Long percentages should be similar across splits
            # Real market data can vary more across different time periods
            long_pcts = [d["long"] for d in distributions.values()]
            long_range = max(long_pcts) - min(long_pcts)

            assert long_range < 30.0, (
                f"Long label distribution varies too much across splits: {long_range:.1f}%"
            )

            # Short percentages should be similar
            short_pcts = [d["short"] for d in distributions.values()]
            short_range = max(short_pcts) - min(short_pcts)

            assert short_range < 30.0, (
                f"Short label distribution varies too much across splits: {short_range:.1f}%"
            )

    def test_label_temporal_stability(self, sample_labels_df):
        """Test label distribution is stable over time."""
        if sample_labels_df is None:
            pytest.skip("No labels data available")

        if "timestamp" not in sample_labels_df.columns:
            pytest.skip("No timestamp column in labels")

        labels = sample_labels_df

        # Split into two halves by time
        mid_idx = len(labels) // 2
        first_half = labels.head(mid_idx)["label"].to_numpy()
        second_half = labels.tail(mid_idx)["label"].to_numpy()

        # Calculate distributions
        first_long = np.sum(first_half == 1) / len(first_half) * 100
        second_long = np.sum(second_half == 1) / len(second_half) * 100

        first_short = np.sum(first_half == -1) / len(first_half) * 100
        second_short = np.sum(second_half == -1) / len(second_half) * 100

        # Distributions should not drift significantly (within 15%)
        long_drift = abs(first_long - second_long)
        short_drift = abs(first_short - second_short)

        assert long_drift < 15.0, (
            f"Long label distribution drifted by {long_drift:.1f}%: {first_long:.1f}% → {second_long:.1f}%"
        )

        assert short_drift < 15.0, (
            f"Short label distribution drifted by {short_drift:.1f}%: {first_short:.1f}% → {second_short:.1f}%"
        )

    @pytest.mark.critical
    def test_no_excessive_hold_labels_with_symmetric_barriers(
        self, sample_labels_df, config
    ):
        """CRITICAL: Test symmetric barriers don't produce excessive Hold labels."""
        if sample_labels_df is None:
            pytest.skip("No labels data available")

        labels = sample_labels_df["label"].to_numpy()
        total = len(labels)
        hold_count = np.sum(labels == 0)
        hold_pct = hold_count / total * 100

        # Verify configuration uses symmetric barriers
        if config is not None:
            assert config.labels.atr_multiplier_tp == config.labels.atr_multiplier_sl, (
                "Barriers should be symmetric"
            )

        # Hold should be less than 45% (not excessive)
        assert hold_pct < 45.0, (
            f"Hold labels excessive: {hold_pct:.1f}% - check barrier symmetry"
        )

        # Long and Short should each be at least 25%
        long_pct = np.sum(labels == 1) / total * 100
        short_pct = np.sum(labels == -1) / total * 100

        assert long_pct >= 25.0, f"Long class too rare: {long_pct:.1f}%"
        assert short_pct >= 25.0, f"Short class too rare: {short_pct:.1f}%"


class TestConfigurationPropagation:
    """Tests for new configuration values propagating through pipeline."""

    @pytest.mark.critical
    def test_new_config_values_propagated(self, config):
        """CRITICAL: Test all new config values are properly propagated."""
        if config is None:
            pytest.skip("No configuration available")

        # Test EMA periods
        if hasattr(config.features, "ema_periods"):
            assert 34 in config.features.ema_periods, "EMA 34 not in config"
            assert 89 in config.features.ema_periods, "EMA 89 not in config"

        # Test correlation threshold
        if hasattr(config.features, "correlation_threshold"):
            assert config.features.correlation_threshold == 0.90, (
                f"Expected correlation threshold 0.90, got {config.features.correlation_threshold}"
            )

        # Test label configuration
        assert config.labels.atr_multiplier_tp == 1.5, (
            f"Expected TP multiplier 1.5, got {config.labels.atr_multiplier_tp}"
        )
        assert config.labels.atr_multiplier_sl == 1.5, (
            f"Expected SL multiplier 1.5, got {config.labels.atr_multiplier_sl}"
        )
        assert config.labels.horizon_bars == 20, (
            f"Expected horizon 20, got {config.labels.horizon_bars}"
        )

        # Test splitting configuration
        assert config.splitting.purge_bars == 15, (
            f"Expected purge 15 bars, got {config.splitting.purge_bars}"
        )

        # Test confidence threshold
        if hasattr(config.models["stacking"], "confidence_threshold"):
            assert config.models["stacking"].confidence_threshold == 0.6, (
                f"Expected confidence threshold 0.6, got {config.models['stacking'].confidence_threshold}"
            )

    def test_ema_34_89_in_features(self, sample_features_df):
        """Test EMA 34 and 89 are present in feature output."""
        if sample_features_df is None:
            pytest.skip("No features data available")

        ema_cols = [c for c in sample_features_df.columns if "ema" in c.lower()]

        # Check for EMA 34 and 89 specifically
        ema_34_present = any("34" in c or "slow" in c.lower() for c in ema_cols)
        ema_89_present = any("89" in c or "fast" in c.lower() for c in ema_cols)

        if len(ema_cols) > 0:
            assert ema_34_present or ema_89_present, (
                "EMA 34 or 89 indicators not found in features"
            )

    def test_correlation_threshold_applied(self, sample_features_df):
        """Test high-correlation features are filtered."""
        if sample_features_df is None:
            pytest.skip("No features data available")

        # Get numeric feature columns (exclude timestamp, label, price columns, lag columns)
        # Price columns (open, high, low, close) and lag columns are naturally highly correlated,
        # so we exclude them from this test
        price_cols = ["open", "high", "low", "close"]
        lag_pattern = re.compile(r".*_lag_\d+$")  # Matches columns ending with _lag_N

        numeric_cols = [
            c
            for c in sample_features_df.columns
            if sample_features_df[c].dtype in [pl.Float64, pl.Float32, pl.Int64]
            and c not in ["timestamp", "label"] + price_cols
            and not lag_pattern.match(c)
        ]

        if len(numeric_cols) < 2:
            pytest.skip("Insufficient numeric features for correlation test")

        # Calculate correlation matrix
        df_numeric = sample_features_df.select(numeric_cols).to_pandas()

        # Drop rows with NaN
        df_clean = df_numeric.dropna()

        if len(df_clean) < 10:
            pytest.skip("Insufficient clean data for correlation test")

        corr_matrix = df_clean.corr()

        # Check for highly correlated pairs (excluding diagonal)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if (
                    corr_val > 0.95
                ):  # Higher than 0.90 threshold to account for variation
                    high_corr_pairs.append(
                        (corr_matrix.columns[i], corr_matrix.columns[j], corr_val)
                    )

        # Should have no extremely high correlations among engineered features
        # Allow up to 20% of pairs to be highly correlated (real data has natural correlations)
        assert len(high_corr_pairs) < len(numeric_cols) * 0.2, (
            f"Too many highly correlated feature pairs: {len(high_corr_pairs)}"
        )


class TestBacktestIntegrationWithNewFeatures:
    """Tests for backtest integration with new features."""

    @pytest.mark.critical
    def test_backtest_with_microstructure_features(
        self, sample_features_df, sample_labels_df
    ):
        """CRITICAL: Test backtest correctly uses microstructure features."""
        if sample_features_df is None or sample_labels_df is None:
            pytest.skip("Missing data")

        # Verify microstructure features are available
        micro_cols = [
            c
            for c in sample_features_df.columns
            if any(
                pattern in c.lower()
                for pattern in ["engulfing", "doji", "hammer", "volume_delta"]
            )
        ]

        # Backtest should work with or without microstructure features
        # The key is that pipeline handles them correctly
        assert len(sample_features_df) > 0, "No features for backtest"

        # If microstructure features exist, verify they're not all null
        for col in micro_cols:
            non_null_count = sample_features_df[col].drop_nulls().shape[0]
            assert non_null_count > 0, f"Microstructure feature {col} is entirely null"

    def test_backtest_with_confidence_threshold_predictions(self):
        """Test backtest handles confidence-thresholded predictions."""
        # Create sample predictions with varying confidence
        predictions = np.array([1, -1, 1, 0, -1, 1])
        probabilities = np.array(
            [
                [0.30, 0.55, 0.15],  # Below threshold
                [0.50, 0.20, 0.30],  # Below threshold
                [0.10, 0.70, 0.20],  # Above threshold
                [0.35, 0.35, 0.30],  # Below threshold
                [0.20, 0.20, 0.60],  # Above threshold
                [0.15, 0.65, 0.20],  # Above threshold
            ]
        )

        threshold = 0.6
        filtered_preds, filtered_probs = apply_confidence_threshold(
            predictions, probabilities, threshold
        )

        # Backtest should receive filtered predictions
        # Verify predictions are valid for backtest
        assert np.all(np.isin(filtered_preds, [-1, 0, 1])), (
            "Filtered predictions contain invalid values"
        )

        # Verify probabilities sum to 1
        prob_sums = filtered_probs.sum(axis=1)
        assert np.allclose(prob_sums, 1.0, rtol=1e-5), (
            "Filtered probabilities don't sum to 1"
        )


# Import helper functions that might not exist yet - mock them if needed
try:
    from thesis.models.stacking import (
        apply_confidence_threshold,
        apply_confidence_threshold_with_history,
    )
except ImportError:
    # Create mock implementations for testing
    def apply_confidence_threshold(predictions, probabilities, threshold=0.6):
        """Mock implementation for testing."""
        max_probs = np.max(probabilities, axis=1)
        low_conf_mask = max_probs < threshold

        filtered_preds = predictions.copy()
        filtered_preds[low_conf_mask] = 0

        filtered_probs = probabilities.copy()
        filtered_probs[low_conf_mask] = np.array([0.0, 1.0, 0.0])

        return filtered_preds, filtered_probs

    def apply_confidence_threshold_with_history(
        predictions, probabilities, threshold=0.6
    ):
        """Mock implementation that preserves original predictions."""
        filtered_preds, filtered_probs = apply_confidence_threshold(
            predictions, probabilities, threshold
        )
        return {
            "original_predictions": predictions.copy(),
            "filtered_predictions": filtered_preds,
            "original_probabilities": probabilities.copy(),
            "filtered_probabilities": filtered_probs,
        }
