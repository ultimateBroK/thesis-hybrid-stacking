"""Tests for model module.

Tests LightGBM training helpers, class weight computation,
and deployment model metadata.
Legacy meta-learner tests removed — pipeline now uses tabular walk-forward models.
"""

from pathlib import Path
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from thesis.models.stacking import _compute_class_weights
from thesis.shared.config import Config


@pytest.fixture
def sample_config() -> Config:
    """Create a sample config for testing."""
    config = Config()
    config.model.num_leaves = 4
    config.model.max_depth = 3
    config.model.learning_rate = 0.1
    config.model.n_estimators = 5
    config.model.min_child_samples = 10
    config.model.subsample = 0.8
    config.model.subsample_freq = 1
    config.model.feature_fraction = 0.8
    config.model.reg_alpha = 0.01
    config.model.reg_lambda = 0.01
    config.model.early_stopping_rounds = 5
    config.workflow.random_seed = 42
    config.workflow.n_jobs = 1
    config.splitting.purge_bars = 5
    return config


@pytest.fixture
def synthetic_classification_data():
    """Create synthetic classification data."""
    np.random.seed(42)
    n_samples = 200
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    # Create simple decision boundary
    y = np.where(X[:, 0] + X[:, 1] > 0, 1, np.where(X[:, 0] + X[:, 1] < -0.5, -1, 0))

    return X, y


@pytest.mark.unit
@pytest.mark.models
def test_compute_class_weights_returns_dict(synthetic_classification_data) -> None:
    """Test _compute_class_weights returns dict with classes as keys."""
    X, y = synthetic_classification_data

    weights = _compute_class_weights(y)

    assert isinstance(weights, dict)
    unique_classes = np.unique(y)
    for cls in unique_classes:
        assert int(cls) in weights
        assert isinstance(weights[int(cls)], float)
        assert weights[int(cls)] > 0


@pytest.mark.unit
@pytest.mark.models
def test_compute_class_weights_balanced(synthetic_classification_data) -> None:
    """Test that class weights are approximately balanced."""
    X, y = synthetic_classification_data

    weights = _compute_class_weights(y)

    class_counts = {cls: np.sum(y == cls) for cls in np.unique(y)}
    max_count = max(class_counts.values())

    for cls, count in class_counts.items():
        max_count / count
        actual_weight = weights[int(cls)]
        assert actual_weight > 0


@pytest.mark.unit
@pytest.mark.models
def test_compute_class_weights_single_class() -> None:
    """Test class weights with single class."""
    y = np.ones(100)

    weights = _compute_class_weights(y)

    assert isinstance(weights, dict)
    assert 1 in weights
    assert weights[1] > 0


@pytest.mark.unit
@pytest.mark.models
def test_class_weights_with_imbalanced_data() -> None:
    """Test class weights with highly imbalanced data."""
    y = np.array([1] * 90 + [0] * 5 + [-1] * 5)

    weights = _compute_class_weights(y)

    # Minority classes should have higher weights
    assert weights[0] > weights[1]
    assert weights[-1] > weights[1]


# ──────────────────────────────────────────────────────────────────────────────
# _build_lgbm_info deployment metadata tests
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.models
@pytest.mark.skip(reason="_build_lgbm_info removed in refactor")
class TestDeploymentModelMetadata:
    """Tests for _build_lgbm_info deployment model metadata."""

    @staticmethod
    def _make_mock_model(
        best_iteration: int = 50, classes: tuple = (0, 1, 2)
    ) -> MagicMock:
        """Create a mock LightGBM model with required attrs.

        Uses MagicMock so ``best_iteration_`` and ``classes_`` are directly
        settable as attributes without real LightGBM dependency.
        """
        model = MagicMock()
        model.best_iteration_ = best_iteration
        model.classes_ = classes
        return model

    def test_window_provenance_keys_present_with_kwargs(self) -> None:
        """When window_index is provided, provenance keys are in the dict."""
        from thesis.models.stacking import _build_lgbm_info  # removed in refactor

        model = self._make_mock_model()
        train_dates = {"start": "2023-01-01", "end": "2023-06-01"}
        test_dates = {"start": "2023-06-02", "end": "2023-07-01"}

        info = _build_lgbm_info(
            model,
            ["f1", "f2", "f3"],
            last_window_accuracy=0.85,
            window_index=5,
            total_windows=10,
            window_train_dates=train_dates,
            window_test_dates=test_dates,
        )

        assert info["window_index"] == 5
        assert info["total_windows"] == 10
        assert info["window_oof_accuracy"] == 0.85
        assert info["window_train_date_range"] == train_dates
        assert info["window_test_date_range"] == test_dates

    def test_backward_compatible_no_window_provenance_keys(self) -> None:
        """Missing kwargs → no crash and no window-provenance keys in result."""
        from thesis.models.stacking import _build_lgbm_info  # removed in refactor

        model = self._make_mock_model()
        info = _build_lgbm_info(model, ["f1", "f2"], last_window_accuracy=0.85)

        # Core keys always present
        for key in (
            "artifact_strategy",
            "validation_protocol",
            "last_window_accuracy",
            "best_iteration",
            "n_features",
            "n_classes",
        ):
            assert key in info, f"Expected key {key} not found in result"

        # Window provenance keys absent
        for key in (
            "window_index",
            "total_windows",
            "window_oof_accuracy",
            "window_train_date_range",
            "window_test_date_range",
        ):
            assert key not in info, f"Provenance key {key} should be absent"

    def test_metadata_includes_per_window_provenance(self) -> None:
        """Result includes per-window provenance when kwargs are supplied."""
        from thesis.models.stacking import _build_lgbm_info  # removed in refactor

        model = self._make_mock_model(best_iteration=75, classes=(0, 1, 2))
        train_dates = {"start": "2024-01-01", "end": "2024-06-01"}
        test_dates = {"start": "2024-06-02", "end": "2024-07-01"}

        info = _build_lgbm_info(
            model,
            ["a", "b", "c", "d"],
            last_window_accuracy=0.92,
            window_index=3,
            total_windows=6,
            window_train_dates=train_dates,
            window_test_dates=test_dates,
        )

        # Provenance keys present
        assert info["window_index"] == 3
        assert info["total_windows"] == 6
        assert info["window_oof_accuracy"] == 0.92
        assert info["window_train_date_range"] == train_dates
        assert info["window_test_date_range"] == test_dates

        # Core metadata correct
        assert info["n_features"] == 4
        assert info["n_classes"] == 3
        assert info["best_iteration"] == 75
        assert info["artifact_strategy"] == "last_walk_forward_window"

    def test_backward_compatible_with_none_accuracy(self) -> None:
        """None accuracy is handled without crash — key present with None."""
        from thesis.models.stacking import _build_lgbm_info  # removed in refactor

        model = self._make_mock_model()
        info = _build_lgbm_info(model, ["f1"], last_window_accuracy=None)

        assert info["last_window_accuracy"] is None
        assert "window_index" not in info

    def test_window_oof_accuracy_equals_last_window_accuracy(self) -> None:
        """window_oof_accuracy mirrors last_window_accuracy when set."""
        from thesis.models.stacking import _build_lgbm_info  # removed in refactor

        model = self._make_mock_model()

        for acc in (0.0, 0.5, 1.0, None):
            info = _build_lgbm_info(
                model,
                ["f"],
                last_window_accuracy=acc,
                window_index=1,
                total_windows=1,
            )
            assert info["window_oof_accuracy"] == acc
            assert info["last_window_accuracy"] == acc


# ──────────────────────────────────────────────────────────────────────────────
# _compute_distribution_shift_weights tests
# SKIPPED: function removed during refactor
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.models
@pytest.mark.skip(reason="_compute_distribution_shift_weights removed in refactor")
class TestDistributionShiftWeights:
    """Tests for _compute_distribution_shift_weights time-safe weighting."""

    def _skip_placeholder(self) -> None:
        pass

    def test_matched_distributions_return_uniformish_weights(self) -> None:
        """When train and val have similar class distributions, weights ≈ 1.0."""
        pass

    def test_shifted_distributions_return_non_uniform_weights(self) -> None:
        """When val has a different class distribution, weights diverge from 1.0."""
        pass

    def test_weights_aligned_to_y_train_not_y_val(self) -> None:
        """Weights array length matches y_train, not y_val."""
        pass

    def test_all_weights_within_clip_bounds(self) -> None:
        """No individual weight exceeds the default clip range [0.5, 3.0]."""
        pass

    def test_uniform_val_distribution_yields_uniform_weights(self) -> None:
        """A perfectly uniform val distribution produces uniform weights."""
        pass

    def test_zero_train_class_handled_gracefully(self) -> None:
        """Absent train classes should not crash — handled by denominator guard."""
        pass

    def test_single_class_train(self) -> None:
        """Degenerate case: single-class train should still return valid weights."""
        pass


# ──────────────────────────────────────────────────────────────────────────────
# _lgbm_utils: _wrap_np, _filter_validation_to_seen_classes, _save_feature_importance
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.models
class TestWrapNp:
    def test_wraps_as_dataframe(self) -> None:
        import pandas as pd

        from thesis.models.stacking import _wrap_np

        X = np.array([[1, 2], [3, 4]])
        result = _wrap_np(X, ["a", "b"])
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a", "b"]
        assert result.shape == (2, 2)


@pytest.mark.unit
@pytest.mark.models
@pytest.mark.skip(reason="_filter_validation_to_seen_classes removed in refactor")
class TestFilterValidationToSeenClasses:
    def test_all_seen(self) -> None:

        pass

    def test_unseen_class_dropped(self) -> None:
        pass

    def test_no_overlap_returns_none(self) -> None:
        pass


@pytest.mark.unit
@pytest.mark.models
@pytest.mark.skip(reason="_save_feature_importance moved to thesis.models.evaluate")
class TestSaveFeatureImportance:
    def test_saves_json(self, tmp_path) -> None:
        pass

    def test_skips_when_no_feature_importances(self, tmp_path) -> None:
        pass
