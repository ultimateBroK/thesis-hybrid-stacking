"""Tests for reporting model comparison helpers.

NOTE: pair_windows_by_date, find_architecture_session, parse_date were removed
in the reporting refactor. Tests are skipped pending rewrite.
"""

from __future__ import annotations

import pytest

from thesis.reporting.report import (
    _parse_date,
    _tbl_row,
    build_model_comparison_rows,
    write_model_comparison_artifacts,
)
from thesis.shared.config import Config

# ---------------------------------------------------------------------------
# _tbl_row
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTblRow:
    def test_basic(self) -> None:
        assert _tbl_row("a", "b", "c") == "| a | b | c |"

    def test_single_cell(self) -> None:
        assert _tbl_row("x") == "| x |"

    def test_empty(self) -> None:
        assert _tbl_row() == "|  |"


# ---------------------------------------------------------------------------
# parse_date (now _parse_date)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestParseDate:
    def test_date_only(self) -> None:
        from datetime import datetime

        result = _parse_date("2024-01-15")
        assert result == datetime(2024, 1, 15)

    def test_datetime_with_space(self) -> None:
        from datetime import datetime

        result = _parse_date("2024-01-15 10:30:00")
        assert result == datetime(2024, 1, 15, 10, 30)

    def test_iso_format(self) -> None:
        from datetime import datetime

        result = _parse_date("2024-01-15T10:30:00")
        assert result == datetime(2024, 1, 15, 10, 30)

    def test_empty_string(self) -> None:
        assert _parse_date("") is None

    def test_invalid_format(self) -> None:
        assert _parse_date("not-a-date") is None

    def test_long_iso_string(self) -> None:
        result = _parse_date("2024-01-15T10:30:00+00:00")
        assert result is not None
        assert result.year == 2024


# ---------------------------------------------------------------------------
# pair_windows_by_date — removed in refactor
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.skip(reason="pair_windows_by_date removed in refactor")
class TestPairWindowsByDate:
    pass


# ---------------------------------------------------------------------------
# find_architecture_session — removed in refactor
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.skip(reason="find_architecture_session removed in refactor")
class TestFindArchitectureSession:
    pass


# ---------------------------------------------------------------------------
# build_model_comparison_rows
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildModelComparisonRows:
    def test_without_pred_stats(self) -> None:
        config = Config()
        rows = build_model_comparison_rows(config, None)
        # Should still return pending rows for planned models
        assert len(rows) >= 1
        models = [r["model"] for r in rows]
        assert any("LightGBM" in m for m in models)
        assert "Logistic Regression" in models
        assert "Random Forest" in models
        assert "Hybrid Stacking" in models


# ---------------------------------------------------------------------------
# write_model_comparison_artifacts
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestWriteModelComparisonArtifacts:
    def test_writes_csv_and_md(self, tmp_path) -> None:
        rows = [
            {
                "model": "Test",
                "accuracy": 0.8,
                "macro_f1": 0.7,
                "directional_accuracy": None,
                "long_f1": None,
                "short_f1": None,
                "mae_return": None,
                "rmse_return": None,
                "r2_return": None,
                "source": "test",
            },
        ]
        csv_path = write_model_comparison_artifacts(tmp_path, rows)
        assert csv_path.exists()
