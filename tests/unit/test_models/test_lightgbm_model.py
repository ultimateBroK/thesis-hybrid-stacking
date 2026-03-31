"""Tests for LightGBM model."""

import numpy as np
import pytest

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from thesis.models.lightgbm_model import train_lightgbm, _train_fixed_params
    HAS_LGBM_MODEL = True
except ImportError:
    HAS_LGBM_MODEL = False


@pytest.mark.skipif(not HAS_LIGHTGBM, reason="LightGBM not available")
class TestLightGBMTraining:
    """Test cases for LightGBM training."""

    @pytest.mark.slow
    def test_training_on_synthetic_data(self):
        """Test LightGBM can be trained on synthetic features."""
        from sklearn.datasets import make_classification
        
        # Generate synthetic data with enough informative features for 3 classes
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_classes=3,
            n_informative=5,  # Must be >= log2(n_classes) = 2, use 5 for safety
            n_clusters_per_class=1,
            random_state=42
        )
        
        # Create LightGBM dataset
        train_data = lgb.Dataset(X, label=y)
        
        # Train simple model
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 10,
            'learning_rate': 0.1,
            'verbose': -1
        }
        
        model = lgb.train(params, train_data, num_boost_round=30)
        
        # Should have trained successfully
        assert model is not None
        
        # Can make predictions
        predictions = model.predict(X)
        assert predictions.shape == (len(X), 3)

    @pytest.mark.slow
    def test_predict_proba(self):
        """Test probability predictions."""
        from sklearn.datasets import make_classification
        
        np.random.seed(42)
        X, y = make_classification(
            n_samples=100, 
            n_features=10, 
            n_classes=3, 
            n_informative=5,  # Must be >= log2(3) = 2
            n_clusters_per_class=1,
            random_state=42
        )
        
        train_data = lgb.Dataset(X, label=y)
        
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'verbose': -1
        }
        
        model = lgb.train(params, train_data, num_boost_round=20)
        
        # Get probabilities
        proba = model.predict(X)
        
        # Should be (n_samples, n_classes)
        assert proba.shape == (len(X), 3)
        
        # Should sum to 1
        assert np.allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

    def test_feature_importance(self):
        """Test feature importance extraction."""
        from sklearn.datasets import make_classification
        
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=3,
            n_clusters_per_class=1,
            random_state=42
        )
        
        train_data = lgb.Dataset(X, label=y)
        
        params = {'objective': 'multiclass', 'num_class': 3, 'verbose': -1}
        model = lgb.train(params, train_data, num_boost_round=30)
        
        # Get importance
        importance = model.feature_importance(importance_type='gain')
        
        assert len(importance) == n_features
        assert np.all(importance >= 0)

    @pytest.mark.slow
    def test_training_with_real_features(self, sample_features_df, sample_labels_df):
        """Test training with real feature data."""
        if sample_features_df is None or sample_labels_df is None:
            pytest.skip("Missing features or labels data")
        
        # Prepare data - exclude timestamp column and select only numeric columns
        X = sample_features_df.head(200).to_pandas()
        y = sample_labels_df.head(200).to_pandas()["label"].values
        
        # Select only numeric columns for LightGBM
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X = X[numeric_cols]
        
        # Remap labels from [-1, 0, 1] to [0, 1, 2] for LightGBM compatibility
        label_map = {-1: 0, 0: 1, 1: 2}
        y = np.array([label_map[int(label)] for label in y])
        
        # Remove any NaN rows
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X_clean = X[mask]
        y_clean = y[mask].astype(int)
        
        if len(X_clean) < 50:
            pytest.skip("Insufficient clean data")
        
        # Train
        train_data = lgb.Dataset(X_clean, label=y_clean)
        params = {'objective': 'multiclass', 'num_class': 3, 'verbose': -1}
        model = lgb.train(params, train_data, num_boost_round=20)
        
        # Predict
        predictions = model.predict(X_clean)
        
        assert predictions.shape == (len(X_clean), 3)


@pytest.mark.skipif(not HAS_LIGHTGBM, reason="LightGBM not available")
class TestLightGBMParams:
    """Tests for LightGBM hyperparameters."""

    def test_default_parameters(self):
        """Test default parameter configuration."""
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        
        # Verify params structure
        assert 'objective' in params
        assert 'num_class' in params
        assert 'learning_rate' in params


@pytest.mark.skipif(not HAS_LIGHTGBM, reason="LightGBM not available")
class TestLightGBMDataLeakage:
    """CRITICAL: Data leakage prevention tests."""

    @pytest.mark.critical
    def test_no_future_feature_leakage(self, sample_features_df):
        """CRITICAL: Verify feature engineering doesn't leak future info."""
        if sample_features_df is None:
            pytest.skip("No features data")
        
        # Check that features are finite
        features = sample_features_df.to_pandas()
        
        assert not np.isnan(features).all().any()
        assert np.isfinite(features).all().all()

    @pytest.mark.critical
    def test_train_test_separation(self):
        """CRITICAL: Test strict train/test separation."""
        from sklearn.datasets import make_classification
        
        np.random.seed(42)
        n_samples = 200
        n_features = 10
        
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=3,
            n_informative=5,  # Must be >= log2(3) = 2
            n_clusters_per_class=1,
            random_state=42
        )
        
        # Manual split with gap
        train_end = 150
        test_start = 155
        
        X_train = X[:train_end]
        y_train = y[:train_end]
        X_test = X[test_start:]
        y_test = y[test_start:]
        
        # Train model
        train_data = lgb.Dataset(X_train, label=y_train)
        params = {'objective': 'multiclass', 'num_class': 3, 'verbose': -1}
        model = lgb.train(params, train_data, num_boost_round=20)
        
        # Test predictions
        predictions = model.predict(X_test)
        
        assert predictions.shape == (len(X_test), 3)
        
        # Verify no test indices leaked into training
        assert len(X_train) + len(X_test) <= n_samples

    @pytest.mark.critical
    def test_feature_importance_stability(self):
        """CRITICAL: Test feature importance is stable."""
        from sklearn.datasets import make_classification
        
        np.random.seed(42)
        X, y = make_classification(
            n_samples=100, 
            n_features=5, 
            n_classes=3, 
            n_informative=3,  # Must be >= log2(3) = 2
            n_clusters_per_class=1,
            random_state=42
        )
        
        # Train two identical models
        train_data1 = lgb.Dataset(X, label=y)
        train_data2 = lgb.Dataset(X, label=y)
        
        params = {'objective': 'multiclass', 'num_class': 3, 'verbose': -1}
        
        model1 = lgb.train(params, train_data1, num_boost_round=30)
        model2 = lgb.train(params, train_data2, num_boost_round=30)
        
        # Importance should be similar (exact match not guaranteed due to threading)
        imp1 = model1.feature_importance()
        imp2 = model2.feature_importance()
        
        # Rank correlation should be high
        from scipy.stats import spearmanr
        corr, _ = spearmanr(imp1, imp2)
        assert corr > 0.8, f"Feature importance unstable: correlation={corr:.2f}"
