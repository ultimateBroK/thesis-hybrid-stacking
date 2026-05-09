"""Single source of truth for feature column lists across pipeline stages."""

from __future__ import annotations

OHLCV_RAW_COLS: list[str] = ["timestamp", "open", "high", "low", "close", "volume"]
"""Core OHLCV columns present in the raw data source."""

OHLCV_OPTIONAL_COLS: list[str] = ["tick_count", "avg_spread"]
"""Optional columns that may accompany OHLCV data."""

LABEL_META_COLS: list[str] = [
    "label",
    "upper_barrier",
    "lower_barrier",
    "touched_bar",
    "event_end",
    "sample_weight",
]
"""Metadata columns produced by the triple-barrier labelling stage."""


def get_static_feature_cols(config) -> list[str]:
    """Return the static feature columns from config."""
    return list(config.features.static_feature_cols)


def get_label_helper_cols(config) -> list[str]:
    """Return helper columns used during label construction (e.g. ATR)."""
    return [f"atr_{config.features.atr_period}"]


def build_feature_output_cols(config) -> list[str]:
    """All columns that ``features.parquet`` must contain.

    Combines OHLCV raw columns, label helpers, and static features.
    """
    return sorted(
        set(
            OHLCV_RAW_COLS
            + get_label_helper_cols(config)
            + get_static_feature_cols(config)
        )
    )


def build_label_output_cols(config) -> list[str]:
    """All columns that ``labels.parquet`` must contain.

    Superset of feature output columns plus label metadata columns.
    """
    return sorted(set(build_feature_output_cols(config) + LABEL_META_COLS))


def build_exclude_cols(config) -> frozenset[str]:
    """Columns excluded from model training — the minimal non-feature set."""
    return frozenset(
        OHLCV_RAW_COLS
        + OHLCV_OPTIONAL_COLS
        + get_label_helper_cols(config)
        + LABEL_META_COLS
    )
