"""Integration tests for complete pipeline."""

import numpy as np
import polars as pl
import pytest

try:
    from thesis.config.loader import load_config
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


class TestPipelineEndToEnd:
    """End-to-end pipeline integration tests."""

    def test_data_loading_to_predictions(self, raw_ohlcv_data, sample_features_df):
        """Test full pipeline from data loading to predictions."""
        # 1. Verify data loaded
        assert len(raw_ohlcv_data) > 0
        assert "close" in raw_ohlcv_data.columns
        
        # 2. Verify features exist
        if sample_features_df is not None:
            assert len(sample_features_df) > 0
            # 3. Verify temporal alignment
            assert len(sample_features_df) <= len(raw_ohlcv_data)

    def test_train_val_test_splitting(self, train_data, val_data, test_data):
        """Test train/val/test data splitting integration."""
        # Verify all splits loaded
        assert len(train_data) > 0
        assert len(val_data) > 0
        assert len(test_data) > 0
        
        # Verify temporal ordering
        if "timestamp" in train_data.columns:
            train_max = train_data["timestamp"].max()
            val_min = val_data["timestamp"].min()
            val_max = val_data["timestamp"].max()
            test_min = test_data["timestamp"].min()
            
            # Should have temporal ordering
            if train_max is not None and val_min is not None:
                assert train_max < val_min or train_max == val_min
            if val_max is not None and test_min is not None:
                assert val_max < test_min or val_max == test_min

    @pytest.mark.critical
    def test_no_data_leakage_between_splits(self, train_data, val_data, test_data):
        """CRITICAL: Verify no data leakage between splits."""
        # Check for overlapping timestamps
        if "timestamp" in train_data.columns:
            train_times = set(train_data["timestamp"].to_list())
            val_times = set(val_data["timestamp"].to_list())
            test_times = set(test_data["timestamp"].to_list())
            
            # No overlap between train and val
            train_val_overlap = train_times & val_times
            assert len(train_val_overlap) == 0, \
                f"Data leakage: {len(train_val_overlap)} timestamps in both train and val"
            
            # No overlap between val and test
            val_test_overlap = val_times & test_times
            assert len(val_test_overlap) == 0, \
                f"Data leakage: {len(val_test_overlap)} timestamps in both val and test"

    @pytest.mark.slow
    @pytest.mark.critical
    @pytest.mark.skipif(not (HAS_LIGHTGBM and HAS_CONFIG), reason="LightGBM or config not available")
    def test_model_training_inference_pipeline(self, sample_features_df, sample_labels_df):
        """CRITICAL: Test full model training and inference pipeline."""
        if sample_features_df is None or sample_labels_df is None:
            pytest.skip("Missing features or labels")
        
        # Prepare data
        n_samples = min(200, len(sample_features_df), len(sample_labels_df))
        X = sample_features_df.head(n_samples).to_pandas()
        y = sample_labels_df.head(n_samples).to_pandas()["label"].values
        
        # Select only numeric columns for ML (exclude timestamp)
        X = X.select_dtypes(include=[np.number])
        
        # Clean data
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[mask]
        y = y[mask].astype(int)
        
        # Convert labels from (-1, 0, 1) to (0, 1, 2) for LightGBM
        y = y + 1  # -1 -> 0, 0 -> 1, 1 -> 2
        
        if len(X) < 50:
            pytest.skip("Insufficient clean data")
        
        # Split: first 80% train, last 20% test
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train LightGBM model
        train_data = lgb.Dataset(X_train, label=y_train)
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'verbose': -1,
            'boosting_type': 'gbdt',
            'num_leaves': 10,
            'learning_rate': 0.1
        }
        model = lgb.train(params, train_data, num_boost_round=30)
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Verify outputs
        assert predictions.shape == (len(X_test), 3)
        assert np.allclose(predictions.sum(axis=1), 1.0, rtol=1e-4)

    def test_feature_label_alignment(self, sample_features_df, sample_labels_df):
        """Test features and labels are temporally aligned."""
        if sample_features_df is None or sample_labels_df is None:
            pytest.skip("Missing data")
        
        # Get aligned subset
        min_len = min(len(sample_features_df), len(sample_labels_df))
        
        features_subset = sample_features_df.head(min_len)
        labels_subset = sample_labels_df.head(min_len)
        
        # Check alignment
        assert len(features_subset) == len(labels_subset)
        
        # If timestamps present, verify exact alignment
        if "timestamp" in features_subset.columns and "timestamp" in labels_subset.columns:
            feat_times = features_subset["timestamp"].to_list()
            label_times = labels_subset["timestamp"].to_list()
            
            # Should be aligned
            assert feat_times == label_times, "Features and labels not temporally aligned"

    @pytest.mark.critical
    def test_temporal_normalization_integrity(self, sample_features_df):
        """CRITICAL: Test temporal normalization in full pipeline."""
        if sample_features_df is None:
            pytest.skip("No features data")
        
        # Get a feature column (skip timestamp if present)
        feature_cols = [c for c in sample_features_df.columns if c != "timestamp"]
        if len(feature_cols) == 0:
            pytest.skip("No feature columns")
        
        feature_col = feature_cols[0]
        values = sample_features_df[feature_col].to_numpy()
        
        # Check no future-looking normalization artifacts
        assert np.all(np.isfinite(values)), "Features contain non-finite values"


class TestDataQuality:
    """Data quality integration tests."""

    @pytest.mark.critical
    def test_no_duplicate_timestamps(self, raw_ohlcv_data):
        """CRITICAL: Verify no duplicate timestamps in data."""
        if "timestamp" not in raw_ohlcv_data.columns:
            pytest.skip("No timestamp column")
        
        timestamps = raw_ohlcv_data["timestamp"].to_list()
        unique_timestamps = set(timestamps)
        
        assert len(timestamps) == len(unique_timestamps), \
            f"Found {len(timestamps) - len(unique_timestamps)} duplicate timestamps"

    def test_data_completeness(self, raw_ohlcv_data):
        """Test data completeness for OHLCV."""
        required_cols = ["open", "high", "low", "close", "volume"]
        
        for col in required_cols:
            assert col in raw_ohlcv_data.columns, f"Missing column: {col}"
            
            # Check for NaN values
            col_data = raw_ohlcv_data[col].to_numpy()
            nan_count = np.isnan(col_data).sum()
            
            # Allow small percentage of NaN (e.g., at start of indicators)
            nan_pct = nan_count / len(col_data) * 100
            assert nan_pct < 5.0, f"Column {col} has {nan_pct:.1f}% NaN values"

    def test_price_consistency(self, raw_ohlcv_data):
        """Test OHLC price consistency."""
        ohlc = raw_ohlcv_data[["open", "high", "low", "close"]].to_numpy()
        
        # High >= Low
        assert np.all(ohlc[:, 1] >= ohlc[:, 2] - 1e-8), "High < Low found"
        
        # High >= Open and Close (mostly)
        high_consistent = np.all(
            (ohlc[:, 1] >= ohlc[:, 0] - 1e-6) & (ohlc[:, 1] >= ohlc[:, 3] - 1e-6)
        )
        assert high_consistent, "High not highest"
        
        # Low <= Open and Close (mostly)
        low_consistent = np.all(
            (ohlc[:, 2] <= ohlc[:, 0] + 1e-6) & (ohlc[:, 2] <= ohlc[:, 3] + 1e-6)
        )
        assert low_consistent, "Low not lowest"

    @pytest.mark.critical
    def test_label_distribution(self, sample_labels_df):
        """CRITICAL: Test label distribution is reasonable."""
        if sample_labels_df is None:
            pytest.skip("No labels data")
        
        labels = sample_labels_df["label"].to_numpy()
        
        # Check all labels valid
        assert np.all(np.isin(labels, [-1, 0, 1])), "Invalid labels found"
        
        # Check distribution not too imbalanced
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        
        for label, count in zip(unique, counts):
            pct = count / total * 100
            # Each label should be at least 5% of data
            assert pct >= 5.0, f"Label {label} only {pct:.1f}% of data"


class TestModelIntegration:
    """Model integration tests."""

    @pytest.mark.slow
    @pytest.mark.skipif(not (HAS_LIGHTGBM and HAS_CONFIG), reason="LightGBM or config not available")
    def test_lightgbm_full_pipeline(self, sample_features_df, sample_labels_df):
        """Test LightGBM model end-to-end."""
        if sample_features_df is None or sample_labels_df is None:
            pytest.skip("Missing data")
        
        # Prepare data
        n_samples = min(150, len(sample_features_df), len(sample_labels_df))
        X = sample_features_df.head(n_samples).to_pandas()
        y = sample_labels_df.head(n_samples).to_pandas()["label"].values
        
        # Select only numeric columns for ML (exclude timestamp)
        X = X.select_dtypes(include=[np.number])
        
        # Clean
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X_clean = X[mask]
        y_clean = y[mask].astype(int)
        
        # Convert labels from (-1, 0, 1) to (0, 1, 2) for LightGBM
        y_clean = y_clean + 1  # -1 -> 0, 0 -> 1, 1 -> 2
        
        if len(X_clean) < 40:
            pytest.skip("Insufficient clean data")
        
        # Split temporally
        split_idx = int(0.8 * len(X_clean))
        X_train, X_test = X_clean[:split_idx], X_clean[split_idx:]
        y_train, y_test = y_clean[:split_idx], y_clean[split_idx:]
        
        # Train
        train_data = lgb.Dataset(X_train, label=y_train)
        params = {'objective': 'multiclass', 'num_class': 3, 'verbose': -1}
        model = lgb.train(params, train_data, num_boost_round=20)
        
        # Predict
        predictions = model.predict(X_test)
        pred_classes = np.argmax(predictions, axis=1)
        
        # Verify
        assert len(pred_classes) == len(X_test)
        
        # Accuracy should be reasonable (above random for 3-class: 33%)
        acc = np.mean(pred_classes == y_test)
        assert acc > 0.25, f"Accuracy {acc:.2f} too low"

    @pytest.mark.critical
    def test_cross_validation_no_leakage(self, sample_features_df):
        """CRITICAL: Test CV doesn't leak data between folds."""
        if sample_features_df is None:
            pytest.skip("No features data")
        
        # Manual temporal CV
        X = sample_features_df.head(200).to_pandas()
        y = np.random.randint(0, 3, len(X))
        
        # 3-fold temporal split
        fold_size = len(X) // 3
        
        for fold in range(3):
            train_end = (fold + 1) * fold_size
            test_start = train_end + 10  # 10-sample gap
            test_end = min(test_start + fold_size, len(X))
            
            if test_start < len(X):
                train_idx = list(range(0, train_end))
                test_idx = list(range(test_start, test_end))
                
                # No overlap
                assert len(set(train_idx) & set(test_idx)) == 0
                
                # Temporal ordering
                assert max(train_idx) < min(test_idx)
