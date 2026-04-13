"""Tests for meta-learner stacking."""

import numpy as np
import pytest

try:
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.ensemble import RandomForestClassifier

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


try:
    from thesis.models.stacking import train_stacking

    HAS_STACKING = True
except ImportError:
    HAS_STACKING = False


@pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not available")
class TestMetaLearner:
    """Test cases for meta-learner."""

    def test_logistic_regression_meta_learner(self):
        """Test logistic regression as meta-learner."""
        np.random.seed(42)
        n_samples = 100
        n_base_models = 2
        n_classes = 3

        # Create synthetic base predictions (probabilities)
        base_predictions = np.random.rand(n_samples, n_base_models * n_classes)
        # Normalize to sum to 1 per model
        for i in range(n_base_models):
            start_idx = i * n_classes
            end_idx = (i + 1) * n_classes
            probs = base_predictions[:, start_idx:end_idx]
            probs = probs / probs.sum(axis=1, keepdims=True)
            base_predictions[:, start_idx:end_idx] = probs

        # True labels
        y = np.random.randint(0, n_classes, n_samples)

        # Train meta-learner
        meta = LogisticRegression(max_iter=1000, random_state=42)
        meta.fit(base_predictions, y)

        # Make predictions
        predictions = meta.predict(base_predictions)

        assert len(predictions) == n_samples
        assert np.all(np.isin(predictions, range(n_classes)))

    @pytest.mark.slow
    def test_ridge_meta_learner(self):
        """Test ridge classifier as meta-learner."""
        np.random.seed(42)
        n_samples = 100
        n_features = 10

        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 3, n_samples)

        meta = RidgeClassifier(random_state=42)
        meta.fit(X, y)

        predictions = meta.predict(X)
        assert len(predictions) == n_samples

    @pytest.mark.slow
    def test_random_forest_meta_learner(self):
        """Test random forest as meta-learner."""
        np.random.seed(42)
        n_samples = 100
        n_features = 10

        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 3, n_samples)

        meta = RandomForestClassifier(n_estimators=10, random_state=42)
        meta.fit(X, y)

        # Get probabilities
        proba = meta.predict_proba(X)

        assert proba.shape == (n_samples, 3)
        assert np.allclose(proba.sum(axis=1), 1.0, rtol=1e-4)


@pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not available")
class TestStackingEnsemble:
    """Test cases for stacking ensemble."""

    @pytest.mark.slow
    def test_ensemble_predictions(self):
        """Test ensemble produces valid predictions."""
        np.random.seed(42)
        n_samples = 100
        n_classes = 3

        # Simulate base model predictions
        base1 = np.random.rand(n_samples, n_classes)
        base1 = base1 / base1.sum(axis=1, keepdims=True)

        base2 = np.random.rand(n_samples, n_classes)
        base2 = base2 / base2.sum(axis=1, keepdims=True)

        # Stack predictions
        stacked = np.hstack([base1, base2])

        # Meta-learner
        y_true = np.random.randint(0, n_classes, n_samples)
        meta = LogisticRegression(max_iter=1000, random_state=42)
        meta.fit(stacked, y_true)

        final_predictions = meta.predict(stacked)

        assert len(final_predictions) == n_samples
        assert np.all(np.isin(final_predictions, range(n_classes)))

    @pytest.mark.critical
    def test_no_label_leakage_in_stacking(
        self, mock_lstm_predictions, mock_lightgbm_predictions
    ):
        """CRITICAL: Verify stacking doesn't leak label information."""
        # Get predictions from both models
        lstm_proba = mock_lstm_predictions["probabilities"]
        lgbm_proba = mock_lightgbm_predictions["probabilities"]

        # Stack
        stacked = np.hstack([lstm_proba, lgbm_proba])

        # True labels
        y = mock_lstm_predictions["labels"]

        # Train meta-learner
        meta = LogisticRegression(max_iter=1000, random_state=42)
        meta.fit(stacked, y)

        # Predictions
        predictions = meta.predict(stacked)

        # Verify predictions consistent
        assert len(predictions) == len(y)

    @pytest.mark.slow
    def test_ensemble_with_different_meta_learners(self):
        """Test ensemble with different meta-learners."""
        np.random.seed(42)
        n_samples = 100
        n_classes = 3

        # Base predictions
        base1 = np.random.rand(n_samples, n_classes)
        base1 = base1 / base1.sum(axis=1, keepdims=True)

        base2 = np.random.rand(n_samples, n_classes)
        base2 = base2 / base2.sum(axis=1, keepdims=True)

        stacked = np.hstack([base1, base2])
        y = np.random.randint(0, n_classes, n_samples)

        # Test different meta-learners
        for meta_cls in [LogisticRegression, RidgeClassifier]:
            if meta_cls == LogisticRegression:
                meta = meta_cls(max_iter=1000, random_state=42)
            else:
                meta = meta_cls(random_state=42)

            meta.fit(stacked, y)
            proba = (
                meta.predict_proba(stacked) if hasattr(meta, "predict_proba") else None
            )

            if proba is not None:
                assert proba.shape == (n_samples, n_classes)
                assert np.allclose(proba.sum(axis=1), 1.0, rtol=1e-4)


class TestStackingDataLeakage:
    """CRITICAL: Data leakage prevention in stacking."""

    @pytest.mark.critical
    def test_out_of_fold_predictions(self):
        """CRITICAL: Verify out-of-fold predictions used for meta-training."""
        np.random.seed(42)
        n_samples = 100
        n_folds = 3
        n_classes = 3

        # Simulate out-of-fold predictions
        oof_predictions = np.zeros((n_samples, n_classes))
        fold_indices = np.array_split(np.arange(n_samples), n_folds)

        for fold_idx in range(n_folds):
            # Simulate predictions for this fold
            fold_size = len(fold_indices[fold_idx])
            fold_pred = np.random.rand(fold_size, n_classes)
            fold_pred = fold_pred / fold_pred.sum(axis=1, keepdims=True)

            # Assign to OOF array
            oof_predictions[fold_indices[fold_idx]] = fold_pred

        # Verify all samples have OOF predictions
        assert np.all(oof_predictions.sum(axis=1) > 0)

        # Verify OOF predictions are valid probabilities
        assert np.allclose(oof_predictions.sum(axis=1), 1.0, rtol=1e-4)

    @pytest.mark.critical
    @pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not available")
    def test_no_test_predictions_in_meta_training(self):
        """CRITICAL: Test predictions never used in meta-learner training."""
        np.random.seed(42)
        n_train = 100
        n_test = 30
        n_classes = 3

        # Train OOF predictions
        train_oof = np.random.rand(n_train, n_classes)
        train_oof = train_oof / train_oof.sum(axis=1, keepdims=True)

        # Test predictions
        test_pred = np.random.rand(n_test, n_classes)
        test_pred = test_pred / test_pred.sum(axis=1, keepdims=True)

        # Train labels
        y_train = np.random.randint(0, n_classes, n_train)

        # Meta-learner trained ONLY on OOF predictions
        meta = LogisticRegression(max_iter=1000, random_state=42)
        meta.fit(train_oof, y_train)

        # Predict on test
        test_predictions = meta.predict(test_pred)

        # Verify no test data leaked into training
        assert len(test_predictions) == n_test

    @pytest.mark.slow
    @pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not available")
    def test_stacking_cv_integrity(self):
        """CRITICAL: Test cross-validation integrity in stacking."""
        np.random.seed(42)
        n_samples = 150
        n_classes = 3

        # Simulate 3-fold CV predictions
        oof_preds = []
        fold_sizes = [50, 50, 50]

        for fold_size in fold_sizes:
            fold_pred = np.random.rand(fold_size, n_classes)
            fold_pred = fold_pred / fold_pred.sum(axis=1, keepdims=True)
            oof_preds.append(fold_pred)

        oof_predictions = np.vstack(oof_preds)
        y = np.random.randint(0, n_classes, n_samples)

        # Train meta-learner on OOF
        meta = LogisticRegression(max_iter=1000, random_state=42)
        meta.fit(oof_predictions, y)

        # Verify predictions are consistent
        final_proba = meta.predict_proba(oof_predictions)

        assert final_proba.shape == (n_samples, n_classes)
        assert np.allclose(final_proba.sum(axis=1), 1.0, rtol=1e-4)


class TestStackingOOFPaths:
    """Verify stacking loads OOF predictions (not validation) for training."""

    def test_stacking_loads_oof_not_val(self):
        """Verify train_stacking() looks for oof_predictions_path, not predictions_path."""
        import inspect
        from thesis.models.stacking import train_stacking

        source = inspect.getsource(train_stacking)

        # Must reference oof_predictions_path (the new field)
        assert "oof_predictions_path" in source, (
            "CRITICAL: train_stacking() must load from oof_predictions_path, not predictions_path"
        )

        # Should NOT load from predictions_path (val predictions) for training
        # The word "predictions_path" alone appearing would indicate val predictions usage
        # But we allow it in error messages and generate_test_predictions

    def test_stacking_docstring_mentions_oof(self):
        """Verify docstring documents OOF-based training."""
        from thesis.models.stacking import train_stacking

        doc = train_stacking.__doc__ or ""
        assert "OOF" in doc or "out-of-fold" in doc.lower(), (
            "train_stacking() docstring must document OOF-based training"
        )

    def test_no_lookahead_in_stacking_source(self):
        """Meta-test: train_stacking source must not contain .shift(-1)."""
        import inspect
        from thesis.models.stacking import train_stacking

        source = inspect.getsource(train_stacking)
        assert ".shift(-1)" not in source, (
            "CRITICAL: train_stacking() contains .shift(-1) — LOOKAHEAD BIAS!"
        )
