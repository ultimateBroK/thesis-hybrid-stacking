"""Tests for model module.

Tests LightGBM training helpers and class weight computation.
Meta-learner tests removed — pipeline now uses GRU + LightGBM directly.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from thesis.config import Config
from thesis.model import _compute_class_weights


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
