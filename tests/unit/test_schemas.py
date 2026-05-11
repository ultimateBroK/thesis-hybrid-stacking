"""Tests for schema validation classes."""

import pytest
import polars as pl

from thesis.shared.config import Config
from thesis.shared.schemas import FeaturesSchema, LabelsSchema, OhlcvSchema
from tests.helpers import (
    create_synthetic_features,
    create_synthetic_labeled_data,
    create_synthetic_ohlcv,
)


# ---------------------------------------------------------------------------
# OhlcvSchema
# ---------------------------------------------------------------------------


class TestOhlcvSchema:
    def test_ohlcv_schema_accepts_valid(self):
        df = create_synthetic_ohlcv(n_rows=10)
        OhlcvSchema.validate(df)

    def test_ohlcv_schema_rejects_missing_columns(self):
        df = create_synthetic_ohlcv(n_rows=10).drop("volume")
        with pytest.raises(ValueError, match="missing columns"):
            OhlcvSchema.validate(df)

    def test_ohlcv_schema_ignores_config(self):
        df = create_synthetic_ohlcv(n_rows=10)
        OhlcvSchema.validate(df, config=Config())


# ---------------------------------------------------------------------------
# FeaturesSchema
# ---------------------------------------------------------------------------


class TestFeaturesSchema:
    def test_features_schema_structural_only(self):
        df = create_synthetic_ohlcv(n_rows=10)
        FeaturesSchema.validate(df, config=None)

    def test_features_schema_full_with_config(self):
        cfg = Config()
        df = create_synthetic_features(cfg, n_rows=10)
        # Add prediction-only cols that the helper does not generate
        df = df.with_columns(
            [
                pl.Series("log_returns", [0.0] * 10),
                pl.Series("open_norm", [0.0] * 10),
                pl.Series("high_norm", [0.0] * 10),
                pl.Series("low_norm", [0.0] * 10),
                pl.Series("close_norm", [0.0] * 10),
            ]
        )
        FeaturesSchema.validate(df, config=cfg)

    def test_features_schema_rejects_missing_feature_cols(self):
        cfg = Config()
        df = create_synthetic_ohlcv(n_rows=10)
        with pytest.raises(ValueError, match="missing columns"):
            FeaturesSchema.validate(df, config=cfg)


# ---------------------------------------------------------------------------
# LabelsSchema
# ---------------------------------------------------------------------------


class TestLabelsSchema:
    def test_labels_schema_structural_only(self):
        df = create_synthetic_ohlcv(n_rows=10).with_columns(
            [
                pl.Series("label", [0] * 10, dtype=pl.Int32),
                pl.Series("upper_barrier", [0.0] * 10),
                pl.Series("lower_barrier", [0.0] * 10),
                pl.Series("touched_bar", [-1] * 10, dtype=pl.Int32),
                pl.Series("event_end", list(range(10)), dtype=pl.Int32),
                pl.Series("sample_weight", [1.0] * 10),
            ]
        )
        LabelsSchema.validate(df, config=None)

    def test_labels_schema_full_with_config(self):
        cfg = Config()
        df = create_synthetic_labeled_data(cfg, n_rows=10)
        # Add prediction-only cols that the helper does not generate
        df = df.with_columns(
            [
                pl.Series("log_returns", [0.0] * 10),
                pl.Series("open_norm", [0.0] * 10),
                pl.Series("high_norm", [0.0] * 10),
                pl.Series("low_norm", [0.0] * 10),
                pl.Series("close_norm", [0.0] * 10),
            ]
        )
        LabelsSchema.validate(df, config=cfg)

    def test_labels_schema_rejects_missing_label(self):
        df = create_synthetic_ohlcv(n_rows=10)
        with pytest.raises(ValueError, match="missing columns"):
            LabelsSchema.validate(df, config=None)
