"""Unit tests for _baselines — baseline prediction strategies."""

from __future__ import annotations

import numpy as np
import pytest

from thesis.models.baselines import (
    compute_metrics,
    majority_class,
    run_all,
)

# ---------------------------------------------------------------------------
# majority_class_baseline
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMajorityClassBaseline:
    """Majority-class baseline tests."""

    def test_finds_most_common(self) -> None:
        """Most frequent class is predicted."""
        y_true = np.array([0, 0, 0, 1, 1, -1])
        preds, cls = majority_class(y_true)
        assert cls == 0
        assert (preds == 0).all()

    def test_tie_picks_first_sorted(self) -> None:
        """Ties follow numpy sorted class order."""
        y_true = np.array([-1, 1])
        preds, cls = majority_class(y_true)
        # np.unique sorts → first in sorted order is -1
        assert cls == -1

    def test_single_class(self) -> None:
        """Single-class input returns that class."""
        y_true = np.array([1, 1, 1])
        _, cls = majority_class(y_true)
        assert cls == 1


# ---------------------------------------------------------------------------
# compute_baseline_metrics
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeBaselineMetrics:
    """Baseline metric tests."""

    def test_perfect_prediction(self) -> None:
        """Perfect predictions score 1.0."""
        y_true = np.array([-1, 0, 1, -1, 0, 1])
        y_pred = np.array([-1, 0, 1, -1, 0, 1])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0
        assert metrics["directional_accuracy"] == 1.0

    def test_returns_expected_keys(self) -> None:
        """Metrics expose expected report keys."""
        y = np.array([0, 1, -1])
        metrics = compute_metrics(y, y)
        assert set(metrics.keys()) == {
            "accuracy",
            "macro_f1",
            "directional_accuracy",
            "short_f1",
            "long_f1",
        }


# ---------------------------------------------------------------------------
# run_all
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRunAllBaselines:
    """Baseline runner tests."""

    @pytest.fixture()
    def sample_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Small label and return fixture."""
        y_true = np.array([1, -1, 0, 1, -1, 0, 1, 0])
        y_returns = np.array([0.5, -0.3, 0.0, 0.2, -0.1, 0.0, 0.4, -0.2])
        return y_true, y_returns

    def test_returns_expected_keys(self, sample_data: tuple) -> None:
        """Runner returns majority baseline only."""
        y_true, y_returns = sample_data
        results = run_all(y_true, y_returns)
        assert set(results.keys()) == {"majority_class"}

    def test_each_baseline_has_metrics(self, sample_data: tuple) -> None:
        """Baseline output includes core metrics."""
        y_true, y_returns = sample_data
        results = run_all(y_true, y_returns)
        for name, metrics in results.items():
            assert "accuracy" in metrics, f"{name} missing accuracy"
            assert "macro_f1" in metrics, f"{name} missing macro_f1"
            assert "directional_accuracy" in metrics, (
                f"{name} missing directional_accuracy"
            )

    def test_majority_class_includes_label(self, sample_data: tuple) -> None:
        """Majority class label is persisted."""
        y_true, y_returns = sample_data
        results = run_all(y_true, y_returns)
        assert "majority_class_label" in results["majority_class"]

    def test_deterministic(self, sample_data: tuple) -> None:
        """Majority baseline is deterministic."""
        y_true, y_returns = sample_data
        r1 = run_all(y_true, y_returns, seed=42)
        r2 = run_all(y_true, y_returns, seed=42)
        for key in r1:
            for metric in ("accuracy", "macro_f1", "directional_accuracy"):
                assert r1[key][metric] == r2[key][metric]
