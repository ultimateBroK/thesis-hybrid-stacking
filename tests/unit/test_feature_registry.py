"""Tests for shared/feature_registry.py — column-set builders."""

from __future__ import annotations

import pytest

from thesis.shared.config import Config
from thesis.shared.constants import (
    CORE_STATIC_FEATURES,
    OHLCV_RAW_COLS,
    build_exclude_cols,
    build_feature_output_cols,
    get_static_feature_cols,
)


@pytest.fixture()
def config() -> Config:
    return Config()


@pytest.mark.unit
class TestBuildFeatureOutputCols:
    def test_build_feature_output_cols_matches_config(self, config: Config) -> None:
        cols = build_feature_output_cols(config)
        static = get_static_feature_cols(config)
        for c in static:
            assert c in cols, f"static feature {c!r} missing from output"
        for c in OHLCV_RAW_COLS:
            assert c in cols, f"OHLCV column {c!r} missing from output"


@pytest.mark.unit
class TestBuildExcludeCols:
    def test_build_exclude_cols_no_right_columns(self, config: Config) -> None:
        excl = build_exclude_cols(config)
        right_cols = [c for c in excl if c.endswith("_right")]
        assert right_cols == [], f"*_right columns in exclude set: {right_cols}"

    def test_build_exclude_cols_no_stale_columns(self, config: Config) -> None:
        excl = build_exclude_cols(config)
        for stale in ("tp_price", "sl_price"):
            assert stale not in excl, (
                f"stale column {stale!r} should not be in exclude set"
            )


@pytest.mark.unit
class TestStaticCols:
    def test_static_cols_match_core_static_features(self, config: Config) -> None:
        cols = get_static_feature_cols(config)
        assert cols == list(CORE_STATIC_FEATURES)
