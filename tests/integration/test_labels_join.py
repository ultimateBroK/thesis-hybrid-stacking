"""Integration tests for generate_labels join-path behavior.

These tests exercise the full generate_labels pipeline with parquet I/O,
so they belong in integration/ rather than unit/.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from unittest.mock import patch

from thesis.shared.config import Config
from thesis.stage_3_labels.labeling import generate_labels


def _make_features_with_ohlc_atr(n: int = 50) -> pl.DataFrame:
    """Build a features DataFrame with OHLC + ATR columns."""
    rng = np.random.default_rng(42)
    close = 1800 + rng.normal(0, 5, n).cumsum()
    return pl.DataFrame(
        {
            "timestamp": np.arange(n, dtype=np.int64),
            "open": close + 0.5,
            "high": close + 3.0,
            "low": close - 3.0,
            "close": close,
            "volume": rng.integers(100, 1000, n).astype(float),
            "atr_14": np.ones(n) * 10.0,
        }
    )


def _make_minimal_config(tmp_path: Path, features_df: pl.DataFrame) -> Config:
    """Build a Config pointing to tmp parquet files for generate_labels tests."""
    feat_path = tmp_path / "features.parquet"
    ohlcv_path = tmp_path / "ohlcv.parquet"
    labels_path = tmp_path / "labels.parquet"

    features_df.write_parquet(feat_path)
    pl.DataFrame(
        {
            "timestamp": [0],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [1.0],
        }
    ).write_parquet(ohlcv_path)

    cfg = Config()
    cfg.paths.features = str(feat_path)
    cfg.paths.ohlcv = str(ohlcv_path)
    cfg.paths.labels = str(labels_path)
    return cfg


_SCHEMA_PATCH = "thesis.stage_3_labels.labeling.LabelsSchema.validate"


@pytest.mark.integration
def test_labels_skip_ohlcv_join_when_features_have_ohlc(
    tmp_path: Path,
) -> None:
    """When features already contain OHLC columns, ohlcv.parquet is NOT loaded."""
    features_df = _make_features_with_ohlc_atr()
    cfg = _make_minimal_config(tmp_path, features_df)

    with (
        patch(_SCHEMA_PATCH),
        patch(
            "thesis.stage_3_labels.labeling.pl.read_parquet",
            wraps=pl.read_parquet,
        ) as mock_read,
    ):
        generate_labels(cfg)

    assert mock_read.call_count == 1, (
        f"Expected exactly 1 parquet read (features only), got {mock_read.call_count}"
    )


@pytest.mark.integration
def test_labels_join_ohlcv_when_features_missing_ohlc(
    tmp_path: Path,
) -> None:
    """When features lack OHLC columns, OHLCV is loaded and joined."""
    n = 50
    rng = np.random.default_rng(42)
    close = 1800 + rng.normal(0, 5, n).cumsum()

    features_df = pl.DataFrame(
        {
            "timestamp": np.arange(n, dtype=np.int64),
            "volume": rng.integers(100, 1000, n).astype(float),
            "atr_14": np.ones(n) * 10.0,
        }
    )
    ohlcv_df = pl.DataFrame(
        {
            "timestamp": np.arange(n, dtype=np.int64),
            "open": close + 0.5,
            "high": close + 3.0,
            "low": close - 3.0,
            "close": close,
        }
    )

    feat_path = tmp_path / "features.parquet"
    ohlcv_path = tmp_path / "ohlcv.parquet"
    labels_path = tmp_path / "labels.parquet"

    features_df.write_parquet(feat_path)
    ohlcv_df.write_parquet(ohlcv_path)

    cfg = Config()
    cfg.paths.features = str(feat_path)
    cfg.paths.ohlcv = str(ohlcv_path)
    cfg.paths.labels = str(labels_path)

    with patch(_SCHEMA_PATCH):
        generate_labels(cfg)

    result = pl.read_parquet(labels_path)
    for col in ("open", "high", "low", "close"):
        assert col in result.columns, f"Missing OHLC column after join: {col}"


@pytest.mark.integration
def test_labels_no_right_columns(tmp_path: Path) -> None:
    """Output DataFrame has zero columns matching the *_right join-artifact pattern."""
    features_df = _make_features_with_ohlc_atr()
    cfg = _make_minimal_config(tmp_path, features_df)

    with patch(_SCHEMA_PATCH):
        generate_labels(cfg)

    result = pl.read_parquet(cfg.paths.labels)
    right_cols = [c for c in result.columns if c.endswith("_right")]
    assert len(right_cols) == 0, f"Found *_right columns in output: {right_cols}"


@pytest.mark.integration
def test_labels_missing_atr_raises(tmp_path: Path) -> None:
    """Missing ATR column raises ValueError before any labeling work."""
    n = 50
    rng = np.random.default_rng(42)
    close = 1800 + rng.normal(0, 5, n).cumsum()
    features_df = pl.DataFrame(
        {
            "timestamp": np.arange(n, dtype=np.int64),
            "open": close + 0.5,
            "high": close + 3.0,
            "low": close - 3.0,
            "close": close,
            "volume": rng.integers(100, 1000, n).astype(float),
        }
    )
    cfg = _make_minimal_config(tmp_path, features_df)

    with patch(_SCHEMA_PATCH):
        with pytest.raises(ValueError, match="atr_14"):
            generate_labels(cfg)
