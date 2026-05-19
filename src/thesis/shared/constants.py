"""Project-wide constants.

Single source of truth for column exclusion sets and other pipeline-level
constants.
"""

# Columns never used as model features
EXCLUDE_COLS: frozenset[str] = frozenset(
    [
        "timestamp",
        "label",
        "upper_barrier",
        "lower_barrier",
        "touched_bar",
        "event_end",
        "sample_weight",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "avg_spread",
        "tick_count",
        "bid_volume",
        "ask_volume",
        "atr_14",
        "log_returns",
    ]
)

# Hourly bars per year (24h × 5d × 52w)
H1_BARS_PER_YEAR: int = 24 * 5 * 52

# Visualization palette
CHART_COLORS: dict[str, str] = {
    "primary": "#2563EB",
    "secondary": "#7C3AED",
    "success": "#059669",
    "danger": "#DC2626",
    "warning": "#D97706",
    "gray": "#6B7280",
    "long": "#059669",
    "short": "#DC2626",
    "flat": "#6B7280",
}

EXCLUDED_FEATURE_COLS = EXCLUDE_COLS


def timeframe_to_ms(timeframe: str) -> int:
    """Parse timeframe string to milliseconds."""
    tf = timeframe.upper()
    if tf.endswith("H"):
        hours = int(tf[:-1])
        if hours <= 0:
            raise ValueError(f"Invalid timeframe '{tf}': hours must be > 0")
        return hours * 3_600_000
    if tf.endswith("MIN"):
        minutes = int(tf[:-3])
        if minutes <= 0:
            raise ValueError(f"Invalid timeframe '{tf}': minutes must be > 0")
        return minutes * 60_000
    if tf.endswith("M"):
        minutes = int(tf[:-1])
        if minutes <= 0:
            raise ValueError(f"Invalid timeframe '{tf}': minutes must be > 0")
        return minutes * 60_000
    if tf in ("D", "1D"):
        return 86_400_000
    raise ValueError(f"Unsupported timeframe: {timeframe}")


# Labeling constants
SAMPLE_WEIGHT_MIN: float = 0.05  # floor for average-uniqueness computation
ATR_LOW_QUANTILE: float = 0.05  # diagnostic logging quantiles
ATR_HIGH_QUANTILE: float = 0.95
LABEL_PROFITABILITY_WARN_PCT: float = 60.0  # warn if both long/short < this %
ROUNDTRIP_MULT: float = 2.0  # commission → price-unit cost multiplier
CENSORED_LABEL: int = -2  # rows with insufficient horizon

# Distribution-shift weight clipping
DIST_SHIFT_CLIP_MIN: float = 0.5  # min/max per-class weight ratios
DIST_SHIFT_CLIP_MAX: float = 3.0

# Numerical stability
FEATURE_EPS: float = 1e-10  # division safety in feature expressions
STD_EPS: float = (
    1e-8  # z-score denominators (larger to avoid amplifying near-constant series)
)

# Calibration
ECE_N_BINS: int = 10  # confidence bins for Expected Calibration Error
CALIB_LR: float = 0.01  # LBFGS learning rate for temperature-scaling
CALIB_MAX_ITER: int = 100  # LBFGS max iterations

# Default LightGBM tabular features — grouped by category
CORE_STATIC_FEATURES: tuple[str, ...] = (
    # Momentum
    "return_1h",
    "return_4h",
    "rsi_14",
    "macd_hist_atr",
    # Trend
    "ema34_vs_ema89",
    "close_vs_ema_34",
    "adx_14",
    # Volatility
    "atr_pct_close",
    "atr_percentile",
    "high_low_range_20",
    # Position
    "price_position_20",
    "close_vs_vwap_atr",
    # Time/session
    "day_of_week",
    "sess_london",
    "sess_ny_am",
    # Tick-derived
    "spread_pct_close",
    "tick_count_zscore_20",
    "volume_imbalance",
)

# ---------------------------------------------------------------------------
# Feature-registry constants & helpers  (merged from feature_registry.py)
# ---------------------------------------------------------------------------

# Type note: the *config* parameter is intentionally left untyped throughout
# this section.  Importing ``Config`` from ``thesis.config`` would create a
# circular dependency because that module may transitively import pipeline
# helpers that depend on the column lists defined here.

# ── OHLCV / label column sets ──

OHLCV_RAW_COLS: list[str] = ["timestamp", "open", "high", "low", "close", "volume"]
"""Core OHLCV columns present in the raw data source."""

OHLCV_OPTIONAL_COLS: list[str] = [
    "tick_count",
    "avg_spread",
    "bid_volume",
    "ask_volume",
]
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

# ── Regime feature column names (only active when enable_regime_features=True) ──

_REGIME_INDICATOR_FEATURES: list[str] = ["volatility_regime", "trend_regime"]
REGIME_FEATURES: list[str] = list(_REGIME_INDICATOR_FEATURES)





# ── Config-driven helpers ──


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

    Note: Stage 3 only consumes model-facing features produced by Stage 2.
    """
    regime_indicator_cols = (
        _REGIME_INDICATOR_FEATURES
        if getattr(config.features, "enable_regime_features", False)
        else []
    )
    base_cols = list(config.features.static_feature_cols)
    return sorted(
        set(
            OHLCV_RAW_COLS
            + get_label_helper_cols(config)
            + base_cols
            + regime_indicator_cols
        )
    )



