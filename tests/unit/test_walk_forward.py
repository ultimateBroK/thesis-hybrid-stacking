"""Tests for walk-forward validation helpers.

Tests for confidence column enrichment added to OOF predictions (task 13).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from thesis.stage_4_training._walk_forward import _add_confidence_columns


def _make_oof_df(n_rows: int = 20) -> pl.DataFrame:
    """Create a synthetic OOF predictions DataFrame with probability columns."""
    rng = np.random.RandomState(42)
    return pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start=pl.datetime(2024, 1, 1, 0),
                end=pl.datetime(2024, 1, 1, 0) + pl.duration(hours=n_rows - 1),
                interval="1h",
                eager=True,
            ),
            "pred_proba_class_minus1": rng.uniform(0.1, 0.5, n_rows),
            "pred_proba_class_0": rng.uniform(0.1, 0.6, n_rows),
            "pred_proba_class_1": rng.uniform(0.1, 0.5, n_rows),
        }
    )


# ---------------------------------------------------------------------------
# Confidence columns tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_add_confidence_columns_produces_expected_columns() -> None:
    """_add_confidence_columns should add max_confidence and confidence_bin."""
    df = _make_oof_df(20)
    result = _add_confidence_columns(df)

    assert "max_confidence" in result.columns
    assert "confidence_bin" in result.columns
    # Original rows should be preserved
    assert len(result) == len(df)


@pytest.mark.unit
def test_add_confidence_columns_max_is_max_of_probas() -> None:
    """max_confidence should be the row-wise max of the three proba columns."""
    df = _make_oof_df(20)
    result = _add_confidence_columns(df)

    expected_max = df.select(
        pl.max_horizontal(
            [
                pl.col("pred_proba_class_minus1"),
                pl.col("pred_proba_class_0"),
                pl.col("pred_proba_class_1"),
            ]
        )
    ).to_series()

    actual_max = result["max_confidence"]
    for i in range(len(df)):
        assert actual_max[i] == pytest.approx(expected_max[i])


@pytest.mark.unit
def test_add_confidence_columns_bin_logic() -> None:
    """Confidence bins should follow: >= 0.6 → high, >= 0.4 → medium, else low."""
    df = _make_oof_df(20)
    result = _add_confidence_columns(df)

    max_conf = result["max_confidence"].to_numpy()
    bins = result["confidence_bin"].to_list()

    for i in range(len(df)):
        if max_conf[i] >= 0.6:
            assert bins[i] == "high", f"Row {i}: conf={max_conf[i]:.3f}, bin={bins[i]}"
        elif max_conf[i] >= 0.4:
            assert bins[i] == "medium", f"Row {i}: conf={max_conf[i]:.3f}, bin={bins[i]}"
        else:
            assert bins[i] == "low", f"Row {i}: conf={max_conf[i]:.3f}, bin={bins[i]}"


@pytest.mark.unit
def test_add_confidence_columns_missing_proba_noop() -> None:
    """When probability columns are missing, the function returns the DF unchanged."""
    df = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start=pl.datetime(2024, 1, 1, 0),
                end=pl.datetime(2024, 1, 1, 3),
                interval="1h",
                eager=True,
            ),
            "pred_label": [1, -1, 0, 1],
        }
    )
    result = _add_confidence_columns(df)

    # Should be unchanged — no new columns
    assert result.columns == df.columns
    assert len(result) == len(df)


@pytest.mark.unit
def test_add_confidence_columns_partial_proba_noop() -> None:
    """When only some probability columns exist, the function is a no-op."""
    df = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start=pl.datetime(2024, 1, 1, 0),
                end=pl.datetime(2024, 1, 1, 3),
                interval="1h",
                eager=True,
            ),
            "pred_proba_class_minus1": [0.5, 0.1, 0.2, 0.3],
            "pred_proba_class_0": [0.3, 0.7, 0.3, 0.3],
            # pred_proba_class_1 is missing
        }
    )
    result = _add_confidence_columns(df)

    # Should be unchanged
    assert "max_confidence" not in result.columns
    assert "confidence_bin" not in result.columns
    assert len(result) == len(df)


@pytest.mark.unit
def test_add_confidence_columns_bins_categorical() -> None:
    """Confidence bins should be string type (not null)."""
    df = _make_oof_df(20)
    result = _add_confidence_columns(df)

    assert result["confidence_bin"].dtype == pl.Utf8
    assert result["confidence_bin"].null_count() == 0
