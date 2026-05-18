"""Tests for Stage 3 validation and evaluation helpers."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from thesis.models.estimators import align_proba
from thesis.models.evaluate import (
    classification_metrics,
    confusion_matrix,
    model_comparison_table,
    one_hot_proba,
    per_class_metrics,
    proba_columns,
)
from thesis.models.train import choose_model_features
from thesis.models.validation import WalkForwardWindow, generate_windows
from thesis.shared.config import Config


@pytest.mark.unit
class TestChooseModelFeatures:
    """Feature selection from config whitelist tests."""

    def test_uses_config_whitelist(self) -> None:
        """Configured static feature columns win when present."""
        config = Config()
        config.features.static_feature_cols = ["rsi_14", "adx_14"]
        df = pl.DataFrame({"rsi_14": [1.0], "adx_14": [2.0], "extra": [3.0]})

        assert choose_model_features(df, config) == ["rsi_14", "adx_14"]

    def test_raises_when_no_configured_features_found(self) -> None:
        """Raises ValueError when no configured features exist in dataset."""
        config = Config()
        config.features.static_feature_cols = ["nonexistent"]
        df = pl.DataFrame({"a": [1.0], "b": [2.0]})

        with pytest.raises(ValueError, match="No configured model features"):
            choose_model_features(df, config)

    def test_filters_to_available_columns(self) -> None:
        """Only columns present in both config and dataset are kept."""
        config = Config()
        config.features.static_feature_cols = ["a", "b", "nonexistent"]
        df = pl.DataFrame({"a": [1.0], "c": [3.0], "b": [2.0]})

        assert choose_model_features(df, config) == ["a", "b"]


@pytest.mark.unit
class TestWalkForwardWindows:
    """Walk-forward window generation tests."""

    def test_properties_return_slice_lengths(self) -> None:
        """Window length properties reflect slice bounds."""
        window = WalkForwardWindow(10, 110, 120, 150)

        assert window.train_len == 100
        assert window.test_len == 30

    def test_generate_windows_applies_purge_and_embargo(self) -> None:
        """Fixed purge and embargo shift train/test bounds."""
        windows = generate_windows(
            total_bars=300,
            train_window_bars=100,
            test_window_bars=50,
            step_bars=50,
            purge_bars=5,
            embargo_bars=3,
            min_train_bars=40,
        )

        assert windows
        assert windows[0].train_end_idx == 45
        assert windows[0].test_start_idx == 58

    def test_generate_windows_uses_event_purge_when_available(self) -> None:
        """Event purge trims training rows with overlapping labels."""
        event_end = np.arange(300)
        event_end[0:100] = 140
        event_end[0:80] = 90

        windows = generate_windows(
            total_bars=300,
            train_window_bars=100,
            test_window_bars=50,
            step_bars=50,
            purge_bars=5,
            embargo_bars=3,
            min_train_bars=40,
            event_end=event_end,
        )

        assert windows[0].train_end_idx == 80
        assert windows[0].test_start_idx == 103


@pytest.mark.unit
class TestProbabilityHelpers:
    """Probability helper tests."""

    def test_one_hot_proba_uses_canonical_columns(self) -> None:
        """Hard labels convert to canonical probability columns."""
        result = one_hot_proba(np.array([-1, 0, 1]))

        assert "pred_proba_class_minus1" in result
        assert "pred_proba_class_0" in result
        assert "pred_proba_class_1" in result
        np.testing.assert_array_equal(result["pred_proba_class_minus1"], [1, 0, 0])
        np.testing.assert_array_equal(result["pred_proba_class_0"], [0, 1, 0])
        np.testing.assert_array_equal(result["pred_proba_class_1"], [0, 0, 1])

    def test_align_probability_matrix_reorders_columns(self) -> None:
        """Estimator probability columns align to [-1, 0, 1]."""
        proba = np.array([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]])

        result = align_proba(proba, [0, 1, -1])

        assert result.shape == (2, 3)
        np.testing.assert_allclose(result[:, 0], [0.7, 0.3])
        np.testing.assert_allclose(result[:, 1], [0.1, 0.3])

    def test_align_probability_matrix_zero_fills_missing_class(self) -> None:
        """Missing class probabilities become zero-filled columns."""
        proba = np.array([[0.6, 0.4], [0.3, 0.7]])

        result = align_proba(proba, [0, 1])

        assert result.shape == (2, 3)
        np.testing.assert_allclose(result[:, 0], [0.0, 0.0])

    def test_proba_columns_uses_report_column_names(self) -> None:
        """Probability columns use report-compatible names."""
        proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])

        result = proba_columns(proba, [-1, 0, 1])

        assert "pred_proba_class_minus1" in result
        assert "pred_proba_class_0" in result
        assert "pred_proba_class_1" in result
        np.testing.assert_allclose(result["pred_proba_class_minus1"], [0.7, 0.1])


@pytest.mark.unit
class TestCoreEvaluation:
    """Core classification evaluation tests."""

    def test_per_class_metrics_reports_all_classes(self) -> None:
        """Per-class metrics include Short, Hold, and Long."""
        y_true = np.array([-1, 0, 1, -1, 0, 1])
        y_pred = np.array([-1, 0, 1, 0, -1, 1])

        result = per_class_metrics(y_true, y_pred)

        assert set(result) == {"-1", "0", "1"}
        assert result["1"]["f1"] == 1.0

    def test_confusion_matrix_is_label_indexed(self) -> None:
        """Confusion matrix is indexed by canonical class labels."""
        y_true = np.array([-1, -1, 0, 1])
        y_pred = np.array([-1, 0, 0, 1])

        result = confusion_matrix(y_true, y_pred)

        assert result["-1"]["-1"] == 1
        assert result["-1"]["0"] == 1
        assert result["1"]["1"] == 1

    def test_classification_metrics_excludes_confidence_diagnostics(self) -> None:
        """Core metrics exclude confidence diagnostics."""
        y_true = np.array([-1, 0, 1])
        y_pred = np.array([-1, 0, 1])

        result = classification_metrics(y_true, y_pred)

        assert result["accuracy"] == 1.0
        assert result["macro_f1"] == 1.0
        assert "mean_confidence" not in result
        assert "confidence_bin" not in result

    def test_model_comparison_table_keeps_core_columns(self) -> None:
        """Model comparison table keeps only report core columns."""
        table = model_comparison_table(
            {
                "hybrid_stacking": {
                    "accuracy": 0.4,
                    "macro_f1": 0.35,
                    "directional_accuracy": 0.51,
                }
            }
        )

        assert table.columns == [
            "model",
            "accuracy",
            "macro_f1",
            "directional_accuracy",
        ]
        assert table["model"][0] == "hybrid_stacking"
