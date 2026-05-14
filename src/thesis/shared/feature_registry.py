"""Single source of truth for feature column lists across pipeline stages."""

from __future__ import annotations

# Type note: the *config* parameter is intentionally left untyped throughout this
# module.  Importing ``Config`` from ``thesis.config`` would create a circular
# dependency because that module may transitively import pipeline helpers that
# depend on the column lists defined here.

# ---------------------------------------------------------------------------
# Constants (no config needed)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Regime feature column names (only active when enable_regime_features=True)
# ---------------------------------------------------------------------------

_REGIME_INDICATOR_FEATURES: list[str] = ["volatility_regime", "trend_regime"]
_REGIME_LABEL_PRIOR_FEATURES: list[str] = [
    "label_prior_long_lag1",
    "label_prior_short_lag1",
]
REGIME_FEATURES: list[str] = _REGIME_INDICATOR_FEATURES + _REGIME_LABEL_PRIOR_FEATURES


def get_regime_feature_cols(config) -> list[str]:
    """Return regime feature columns if enabled, empty list otherwise."""
    if not getattr(config.features, "enable_regime_features", False):
        return []
    return list(REGIME_FEATURES)


# ---------------------------------------------------------------------------
# Config-driven helpers
# ---------------------------------------------------------------------------


def get_static_feature_cols(config) -> list[str]:
    """Return the static (non-sequential) feature columns from config."""
    cols = list(config.features.static_feature_cols)
    if getattr(config.features, "enable_regime_features", False):
        for c in REGIME_FEATURES:
            if c not in cols:
                cols.append(c)
    return cols


def get_label_helper_cols(config) -> list[str]:
    """Return helper columns used during label construction (e.g. ATR)."""
    return [f"atr_{config.features.atr_period}"]


def build_feature_output_cols(config) -> list[str]:
    """All columns that ``features.parquet`` must contain.

    Combines OHLCV raw columns, label helpers, and model-facing tabular
    features into a sorted, deduplicated list.

    Note: regime label-prior features (label_prior_long_lag1, etc.) are NOT
    included here because they depend on labels which don't exist at feature
    engineering time.  They are added dynamically in stage_4.
    """
    # Build indicator-only regime features for features.parquet
    regime_indicator_cols = (
        _REGIME_INDICATOR_FEATURES
        if getattr(config.features, "enable_regime_features", False)
        else []
    )
    # Use static_feature_cols from config (without regime additions) as the base
    base_cols = list(config.features.static_feature_cols)
    return sorted(
        set(
            OHLCV_RAW_COLS
            + get_label_helper_cols(config)
            + base_cols
            + regime_indicator_cols
        )
    )


def build_label_output_cols(config) -> list[str]:
    """All columns that ``labels.parquet`` must contain.

    Superset of feature output columns plus label metadata columns.
    """
    return sorted(set(build_feature_output_cols(config) + LABEL_META_COLS))


def build_exclude_cols(config) -> frozenset[str]:
    """Columns excluded from model training — the minimal non-feature set.

    These columns are either identifiers (timestamp), raw price/volume data,
    labelling artefacts, or helper/diagnostic columns that should not be
    used as tabular model inputs.
    """
    return frozenset(
        OHLCV_RAW_COLS
        + OHLCV_OPTIONAL_COLS
        + get_label_helper_cols(config)
        + LABEL_META_COLS
        + ["log_returns"]  # helper return alias, excluded from static model inputs
    )
