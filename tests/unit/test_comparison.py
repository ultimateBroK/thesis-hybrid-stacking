"""Tests for stage_6_reporting/_comparison.py — model comparison helpers."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from thesis.shared.config import Config
from thesis.stage_6_reporting.comparison import (
    build_model_comparison_rows,
    find_architecture_session,
    pair_windows_by_date,
    parse_date,
    write_model_comparison_artifacts,
)
from thesis.stage_6_reporting.md_format import _tbl_row

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
# parse_date
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestParseDate:
    def test_date_only(self) -> None:
        from datetime import datetime

        result = parse_date("2024-01-15")
        assert result == datetime(2024, 1, 15)

    def test_datetime_with_space(self) -> None:
        from datetime import datetime

        result = parse_date("2024-01-15 10:30:00")
        assert result == datetime(2024, 1, 15, 10, 30)

    def test_iso_format(self) -> None:
        from datetime import datetime

        result = parse_date("2024-01-15T10:30:00")
        assert result == datetime(2024, 1, 15, 10, 30)

    def test_empty_string(self) -> None:
        assert parse_date("") is None

    def test_invalid_format(self) -> None:
        assert parse_date("not-a-date") is None

    def test_long_iso_string(self) -> None:

        result = parse_date("2024-01-15T10:30:00+00:00")
        assert result is not None
        assert result.year == 2024


# ---------------------------------------------------------------------------
# pair_windows_by_date
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPairWindowsByDate:
    def test_basic_pairing(self) -> None:
        current = [
            {
                "accuracy": 0.8,
                "window": 1,
                "test_dates": {"start": "2024-01-01", "end": "2024-02-01"},
            },
            {
                "accuracy": 0.85,
                "window": 2,
                "test_dates": {"start": "2024-02-01", "end": "2024-03-01"},
            },
        ]
        sibling = [
            {
                "accuracy": 0.75,
                "window": 1,
                "test_dates": {"start": "2024-01-01", "end": "2024-02-01"},
            },
            {
                "accuracy": 0.82,
                "window": 2,
                "test_dates": {"start": "2024-02-01", "end": "2024-03-01"},
            },
        ]
        paired = pair_windows_by_date(current, sibling)
        assert len(paired) == 2
        assert paired[0] == (0.8, 0.75)
        assert paired[1] == (0.85, 0.82)

    def test_missing_accuracy_skipped(self) -> None:
        current = [
            {
                "accuracy": None,
                "window": 1,
                "test_dates": {"start": "2024-01-01", "end": "2024-02-01"},
            }
        ]
        sibling = [
            {
                "accuracy": 0.75,
                "window": 1,
                "test_dates": {"start": "2024-01-01", "end": "2024-02-01"},
            }
        ]
        paired = pair_windows_by_date(current, sibling)
        assert len(paired) == 0

    def test_no_overlap_no_pairing(self) -> None:
        current = [
            {
                "accuracy": 0.8,
                "window": 1,
                "test_dates": {"start": "2024-01-01", "end": "2024-02-01"},
            }
        ]
        sibling = [
            {
                "accuracy": 0.75,
                "window": 1,
                "test_dates": {"start": "2025-01-01", "end": "2025-02-01"},
            }
        ]
        paired = pair_windows_by_date(current, sibling)
        assert len(paired) == 0

    def test_partial_overlap_picks_best(self) -> None:
        current = [
            {
                "accuracy": 0.8,
                "window": 1,
                "test_dates": {"start": "2024-01-01", "end": "2024-03-01"},
            }
        ]
        sibling = [
            {
                "accuracy": 0.7,
                "window": 1,
                "test_dates": {"start": "2024-01-01", "end": "2024-02-01"},
            },
            {
                "accuracy": 0.9,
                "window": 2,
                "test_dates": {"start": "2024-02-01", "end": "2024-03-01"},
            },
        ]
        paired = pair_windows_by_date(current, sibling)
        # Both overlap by ~1 month each; picks the best overlapping one
        assert len(paired) == 1

    def test_missing_test_dates_skipped(self) -> None:
        current = [{"accuracy": 0.8, "window": 1}]
        sibling = [
            {
                "accuracy": 0.75,
                "window": 1,
                "test_dates": {"start": "2024-01-01", "end": "2024-02-01"},
            }
        ]
        paired = pair_windows_by_date(current, sibling)
        assert len(paired) == 0


# ---------------------------------------------------------------------------
# find_architecture_session
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFindArchitectureSession:
    def test_nonexistent_dir(self, tmp_path) -> None:
        assert (
            find_architecture_session(tmp_path / "nonexistent", "static", "/foo")
            is None
        )

    def test_finds_matching_session(self, tmp_path) -> None:
        session = tmp_path / "XAUUSD_1H_20240101"
        session.mkdir()
        config_dir = session / "config"
        config_dir.mkdir()
        snapshot = config_dir / "config_snapshot.toml"
        snapshot.write_text('[model]\narchitecture = "static"\n')

        result = find_architecture_session(tmp_path, "static", "/other")
        assert result == session

    def test_finds_lgbm_session(self, tmp_path) -> None:
        session = tmp_path / "XAUUSD_1H_20240102"
        session.mkdir()
        config_dir = session / "config"
        config_dir.mkdir()
        snapshot = config_dir / "config_snapshot.toml"
        snapshot.write_text('[model]\narchitecture = "lgbm"\n')

        result = find_architecture_session(tmp_path, "lgbm", "/other")
        assert result == session

    def test_excludes_current_session(self, tmp_path) -> None:
        session = tmp_path / "XAUUSD_1H_20240101"
        session.mkdir()
        config_dir = session / "config"
        config_dir.mkdir()
        snapshot = config_dir / "config_snapshot.toml"
        snapshot.write_text('[model]\narchitecture = "static"\n')

        result = find_architecture_session(tmp_path, "static", str(session))
        assert result is None

    def test_wrong_architecture_not_found(self, tmp_path) -> None:
        session = tmp_path / "XAUUSD_1H_20240101"
        session.mkdir()
        config_dir = session / "config"
        config_dir.mkdir()
        snapshot = config_dir / "config_snapshot.toml"
        snapshot.write_text('[model]\narchitecture = "stacking"\n')

        result = find_architecture_session(tmp_path, "static", "/other")
        assert result is None


# ---------------------------------------------------------------------------
# build_model_comparison_rows
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildModelComparisonRows:
    def test_with_pred_stats(self) -> None:
        config = Config()
        pred_stats = {
            "directional_accuracy": 0.7,
            "accuracy": 0.65,
            "macro_f1": 0.6,
            "per_class": {"Long": {"f1": 0.7}, "Short": {"f1": 0.5}},
            "regression_auxiliary": {"mae": 0.1, "rmse": 0.2, "r_squared": 0.3},
        }
        rows = build_model_comparison_rows(config, pred_stats)
        assert len(rows) >= 1
        assert rows[0]["accuracy"] == 0.65
        assert rows[0]["directional_accuracy"] == 0.7

    def test_current_pred_stats_use_model_label(self) -> None:
        config = Config()
        config.model.architecture = "stacking"
        pred_stats = {
            "directional_accuracy": 0.7,
            "accuracy": 0.65,
            "macro_f1": 0.6,
            "per_class": {"Long": {"f1": 0.7}, "Short": {"f1": 0.5}},
        }
        rows = build_model_comparison_rows(config, pred_stats)
        assert rows[0]["model"] == "Hybrid Stacking"
        assert rows[0]["source"] == "current_session"

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

    def test_with_predictions_file(self, tmp_path) -> None:
        config = Config()
        config.paths.predictions = str(tmp_path / "preds.parquet")
        config.paths.ohlcv = str(tmp_path / "ohlcv.parquet")

        n = 30
        preds_df = pl.DataFrame(
            {
                "timestamp": pl.datetime_range(
                    pl.datetime(2024, 1, 1),
                    pl.datetime(2024, 1, 1) + pl.duration(hours=n - 1),
                    interval="1h",
                    eager=True,
                ),
                "true_label": np.random.choice([-1, 0, 1], n),
                "pred_label": np.random.choice([-1, 0, 1], n),
            }
        )
        preds_df.write_parquet(tmp_path / "preds.parquet")

        rows = build_model_comparison_rows(config, None)
        assert len(rows) >= 1


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
        csv_path, md_path = write_model_comparison_artifacts(tmp_path, rows)
        assert csv_path.exists()
        assert md_path.exists()
        assert "Test" in md_path.read_text()
