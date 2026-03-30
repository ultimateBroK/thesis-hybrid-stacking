"""Unit tests for label validation module.

Tests for lookahead bias detection, label consistency, and validation
functions in the thesis.validation.label_validation module.
"""

import numpy as np
import polars as pl
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path


class TestValidateNoLookahead:
    """Tests for the validate_no_lookahead function."""

    def test_empty_dataframe_passes(self):
        """Empty dataframe should pass validation."""
        from thesis.validation.label_validation import validate_no_lookahead

        df = pl.DataFrame({"label": []})
        results = validate_no_lookahead(df)

        assert results["passed"] is True
        assert len(results["errors"]) == 0

    def test_all_neutral_labels_passes(self):
        """All neutral labels (0) should pass validation."""
        from thesis.validation.label_validation import validate_no_lookahead

        df = pl.DataFrame({"label": [0, 0, 0, 0, 0]})
        results = validate_no_lookahead(df)

        assert results["passed"] is True
        assert results["details"]["total_labels"] == 5
        assert results["details"]["hold_pct"] == 1.0

    def test_rapid_oscillations_detected(self):
        """1 -> -1 or -1 -> 1 changes should be flagged."""
        from thesis.validation.label_validation import validate_no_lookahead

        # Create labels with rapid oscillations (1 -> -1)
        labels = [1, -1, 1, -1, 1, -1, 1, -1, 0, 0]
        df = pl.DataFrame({"label": labels})

        # Low threshold to trigger error
        results = validate_no_lookahead(df, max_rapid_change_pct=50.0)

        assert results["details"]["rapid_oscillations"] == 7  # Changes between 1/-1
        assert results["details"]["rapid_oscillation_pct"] == pytest.approx(77.78, abs=0.1)  # 7/9 changes

    def test_rapid_oscillations_under_threshold_passes(self):
        """Rapid oscillations under threshold should pass."""
        from thesis.validation.label_validation import validate_no_lookahead

        # Mix with few oscillations
        labels = [1, 1, 1, -1, -1, -1, 1, 1, 0, 0]
        df = pl.DataFrame({"label": labels})

        results = validate_no_lookahead(df, max_rapid_change_pct=30.0)

        # Only 2 rapid oscillations (1->-1 at pos 2-3, -1->1 at pos 5-6)
        assert results["passed"] is True
        assert results["details"]["rapid_oscillations"] == 2

    def test_high_win_rate_warning(self):
        """Suspiciously high win rate should generate warning."""
        from thesis.validation.label_validation import validate_no_lookahead

        # 90% win rate (9 wins, 1 loss)
        labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, -1]
        df = pl.DataFrame({"label": labels})

        results = validate_no_lookahead(df, max_win_rate=0.60)

        assert any("win rate" in w.lower() for w in results["warnings"])
        assert results["details"]["win_rate"] == 0.9

    def test_normal_win_rate_no_warning(self):
        """Normal 50% win rate should not generate warning."""
        from thesis.validation.label_validation import validate_no_lookahead

        # 50% win rate
        labels = [1, 1, 1, 1, 1, -1, -1, -1, -1, -1]
        df = pl.DataFrame({"label": labels})

        results = validate_no_lookahead(df, max_win_rate=0.70)

        assert not any("win rate" in w.lower() for w in results["warnings"])
        assert results["details"]["win_rate"] == 0.5

    def test_directional_imbalance_warning(self):
        """Severe directional imbalance should generate warning."""
        from thesis.validation.label_validation import validate_no_lookahead

        # 70% long, 30% short
        labels = [1, 1, 1, 1, 1, 1, 1, -1, -1, -1]
        df = pl.DataFrame({"label": labels})

        results = validate_no_lookahead(df)

        assert any("imbalanced" in w.lower() or "bias" in w.lower() for w in results["warnings"])

    def test_last_n_bars_non_neutral_warning(self):
        """Too many non-neutral labels in last N bars should warn."""
        from thesis.validation.label_validation import validate_no_lookahead

        # Last 5 bars have non-neutral labels
        labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        df = pl.DataFrame({"label": labels})

        results = validate_no_lookahead(df, max_horizon=5)

        # Last 5 bars have 100% non-neutral
        assert results["details"]["last_n_non_neutral_pct"] == 100.0
        assert any("last" in w.lower() for w in results["warnings"])

    def test_label_distribution_details(self):
        """Label distribution should be calculated correctly."""
        from thesis.validation.label_validation import validate_no_lookahead

        labels = [1, 1, 1, -1, -1, 0, 0, 0, 0, 0]
        df = pl.DataFrame({"label": labels})

        results = validate_no_lookahead(df)

        assert results["details"]["hold_pct"] == 0.5
        assert results["details"]["long_pct"] == 0.3
        assert results["details"]["short_pct"] == 0.2

    def test_rapid_change_exceeds_threshold_fails(self):
        """Rapid changes exceeding threshold should fail validation."""
        from thesis.validation.label_validation import validate_no_lookahead

        # Alternating 1 and -1 creates maximum rapid oscillations
        labels = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
        df = pl.DataFrame({"label": labels})

        # Set threshold below the oscillation rate
        results = validate_no_lookahead(df, max_rapid_change_pct=80.0)

        assert results["passed"] is False
        assert len(results["errors"]) > 0
        assert any("lookahead" in e.lower() or "oscillation" in e.lower() for e in results["errors"])


class TestRunLabelValidationPipeline:
    """Tests for the run_label_validation_pipeline function."""

    @patch("thesis.validation.label_validation.validate_no_lookahead")
    @patch("thesis.config.loader.Config")
    def test_pipeline_with_valid_labels(self, mock_config_class, mock_validate):
        """Pipeline should succeed with valid labels."""
        from thesis.validation.label_validation import run_label_validation_pipeline

        # Setup mocks
        mock_config = MagicMock()
        mock_config.labels.labels_path = "fake/path/labels.parquet"
        mock_config_class.return_value = mock_config

        # Mock to return passing results
        mock_validate.return_value = {
            "passed": True,
            "errors": [],
            "warnings": [],
            "details": {"win_rate": 0.5, "rapid_oscillation_pct": 5.0},
        }

        with patch("polars.read_parquet") as mock_read:
            mock_read.return_value = pl.DataFrame({"label": [1, -1, 0, 1, -1]})
            with patch("pathlib.Path.exists", return_value=True):
                results = run_label_validation_pipeline("config.toml")

        assert results["passed"] is True
        mock_validate.assert_called_once()

    @patch("thesis.config.loader.Config")
    def test_pipeline_file_not_found(self, mock_config_class):
        """Pipeline should return error if labels file not found."""
        from thesis.validation.label_validation import run_label_validation_pipeline

        mock_config = MagicMock()
        mock_config.labels.labels_path = "fake/nonexistent.parquet"
        mock_config_class.return_value = mock_config

        # Mock Path.exists to return False
        with patch("pathlib.Path.exists", return_value=False):
            results = run_label_validation_pipeline("config.toml")

        assert "error" in results
        assert results["passed"] is False

    @patch("thesis.validation.label_validation.validate_no_lookahead")
    @patch("thesis.config.loader.Config")
    def test_pipeline_failure_raises_error(self, mock_config_class, mock_validate):
        """Pipeline should raise ValueError on validation failure."""
        from thesis.validation.label_validation import run_label_validation_pipeline

        mock_config = MagicMock()
        mock_config.labels.labels_path = "fake/path/labels.parquet"
        mock_config_class.return_value = mock_config

        mock_validate.return_value = {
            "passed": False,
            "errors": ["Lookahead bias detected"],
            "warnings": [],
            "details": {},
        }

        with patch("polars.read_parquet") as mock_read:
            mock_read.return_value = pl.DataFrame({"label": [1, -1, 1, -1, 1]})
            with patch("pathlib.Path.exists", return_value=True):
                with pytest.raises(ValueError, match="Label validation failed"):
                    run_label_validation_pipeline("config.toml")


class TestQuickLabelCheck:
    """Tests for the quick_label_check function."""

    def test_quick_check_valid_labels(self):
        """Quick check should return True for valid labels."""
        from thesis.validation.label_validation import quick_label_check

        with patch("polars.read_parquet") as mock_read:
            mock_read.return_value = pl.DataFrame({"label": [1, -1, 0, 1, 0, -1]})

            result = quick_label_check("fake/path.parquet")
            assert result is True

    def test_quick_check_invalid_labels(self):
        """Quick check should return False for invalid labels."""
        from thesis.validation.label_validation import quick_label_check

        # Create a DataFrame with impossible oscillations
        df = pl.DataFrame({"label": [1, -1, 1, -1, 1, -1]})

        with patch("polars.read_parquet", return_value=df):
            result = quick_label_check("fake/path.parquet")
            # Will fail due to high rapid oscillations
            assert result is False

    def test_quick_check_handles_exceptions(self):
        """Quick check should return False on any exception."""
        from thesis.validation.label_validation import quick_label_check

        with patch("polars.read_parquet", side_effect=Exception("Read error")):
            result = quick_label_check("fake/path.parquet")
            assert result is False


class TestEdgeCases:
    """Edge case tests for label validation."""

    def test_single_row_dataframe(self):
        """Single row dataframe should not cause errors."""
        from thesis.validation.label_validation import validate_no_lookahead

        df = pl.DataFrame({"label": [1]})
        results = validate_no_lookahead(df)

        assert results["passed"] is True
        assert results["details"]["total_labels"] == 1

    def test_two_row_dataframe(self):
        """Two row dataframe should handle diff calculation."""
        from thesis.validation.label_validation import validate_no_lookahead

        df = pl.DataFrame({"label": [1, -1]})
        results = validate_no_lookahead(df)

        assert results["details"]["rapid_oscillations"] == 1

    def test_all_same_directional_labels(self):
        """All same directional labels should not flag as oscillations."""
        from thesis.validation.label_validation import validate_no_lookahead

        df = pl.DataFrame({"label": [1, 1, 1, 1, 1]})
        results = validate_no_lookahead(df)

        assert results["details"]["rapid_oscillations"] == 0
        assert results["details"]["win_rate"] == 1.0  # All "wins"

    def test_no_directional_labels(self):
        """No directional labels should handle win rate calculation."""
        from thesis.validation.label_validation import validate_no_lookahead

        df = pl.DataFrame({"label": [0, 0, 0, 0]})
        results = validate_no_lookahead(df)

        assert results["passed"] is True
        # No win rate in details if no directional labels
        assert "win_rate" not in results["details"] or results["details"].get("wins") == 0

    def test_large_dataframe_performance(self):
        """Large dataframe should complete without timeout."""
        from thesis.validation.label_validation import validate_no_lookahead

        # Create 10000 rows with realistic pattern (mostly 0s, few oscillations)
        np.random.seed(42)
        # Generate data with low oscillation rate - long runs of same label
        labels = []
        current = np.random.choice([1, -1])
        for i in range(10000):
            if i % 100 == 0 and i > 0:  # Change every 100 samples
                current = np.random.choice([1, -1, 0])
            labels.append(current)
        
        df = pl.DataFrame({"label": labels})

        results = validate_no_lookahead(df)

        assert results["details"]["total_labels"] == 10000
