"""Tests for walk-forward validation helpers.

Tests for confidence column enrichment added to OOF predictions (task 13).
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from thesis.stage_4_training._walk_forward import (
    _add_confidence_columns,
    _validate_predictions,
    _write_prediction_manifest,
)


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
            "true_label": rng.choice([-1, 0, 1], n_rows),
            "pred_label": rng.choice([-1, 0, 1], n_rows),
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
            assert bins[i] == "medium", (
                f"Row {i}: conf={max_conf[i]:.3f}, bin={bins[i]}"
            )
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


@pytest.mark.unit
def test_validate_predictions_rejects_duplicate_timestamps(tmp_path) -> None:
    """OOF predictions must have unique timestamps (no window overlap)."""
    df = _add_confidence_columns(_make_oof_df(4))
    bad = df.with_columns(pl.lit(df["timestamp"][0]).alias("timestamp"))

    with pytest.raises(ValueError, match="duplicate timestamps"):
        _validate_predictions(bad, tmp_path / "final_predictions.parquet")


@pytest.mark.unit
def test_validate_predictions_rejects_invalid_label(tmp_path) -> None:
    """pred_label must stay in {-1, 0, 1}."""
    df = _add_confidence_columns(_make_oof_df(4)).with_columns(
        pl.when(pl.arange(0, pl.len()) == 0)
        .then(2)
        .otherwise(pl.col("pred_label"))
        .alias("pred_label")
    )

    with pytest.raises(ValueError, match="Invalid pred_label"):
        _validate_predictions(df, tmp_path / "final_predictions.parquet")


@pytest.mark.unit
def test_write_prediction_manifest(tmp_path) -> None:
    """Prediction manifest should summarize final_predictions.parquet."""
    df = _add_confidence_columns(_make_oof_df(5))
    preds_path = tmp_path / "final_predictions.parquet"

    _validate_predictions(df, preds_path)
    _write_prediction_manifest(df, preds_path, windows_count=2)

    with open(tmp_path / "prediction_manifest.json") as f:
        manifest = json.load(f)
    assert manifest["row_count"] == 5
    assert manifest["windows_count"] == 2
    assert manifest["start"] == str(df["timestamp"][0])
    assert manifest["end"] == str(df["timestamp"][-1])
    assert manifest["mean_confidence"] is not None
