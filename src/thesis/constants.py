"""Project-wide constants shared across pipeline stages.

This module is the single source of truth for column exclusion sets and other
pipeline-level constants. Importing from here prevents the silent drift that
occurs when each stage maintains its own copy.
"""

# ---------------------------------------------------------------------------
# Column exclusion sets
# ---------------------------------------------------------------------------

#: Columns that are *never* model features — excluded from training,
#: correlation filtering, and feature selection everywhere.
#:
#: Rationale per group:
#:  - timestamp          → index / join key, not a feature
#:  - label              → target variable (look-ahead)
#:  - upper_barrier/lower_barrier/touched_bar → label-derived, pure look-ahead
#:  - tp_price/sl_price → legacy label aliases from older cached artifacts
#:  - open_right/high_right/low_right/close_right → label-derived look-ahead
#:  - open/high/low/close/volume → raw OHLCV, excluded to avoid raw price leakage
#:  - avg_spread/tick_count → microstructure columns kept for backtest
#:    but not useful as ML features in their raw form
#:  - log_returns → GRU sequence input; excluded from the *static* LightGBM features
#:    to avoid double-counting the information already encoded in GRU hidden states
#:
#: All 28 engineered features (core indicators, multi-timeframe 4H, trend
#: distances, Bollinger bands, volume z-score, log returns, range, and regime
#: features) are intentionally NOT in this set — they are available as GRU
#: sequence inputs and/or static LightGBM features.
EXCLUDE_COLS: frozenset[str] = frozenset(
    [
        "timestamp",
        "label",
        "upper_barrier",
        "lower_barrier",
        "touched_bar",
        "tp_price",
        "sl_price",
        "open_right",  # Label-derived — pure look-ahead
        "high_right",  # Label-derived — pure look-ahead
        "low_right",  # Label-derived — pure look-ahead
        "close_right",  # Label-derived — pure look-ahead
        "open",
        "high",
        "low",
        "close",
        "volume",
        "avg_spread",
        "tick_count",
        "log_returns",  # GRU sequence input — not a static feature for LightGBM
    ]
)

# Backward-compatible private alias used by internal modules
_EXCLUDE_COLS = EXCLUDE_COLS

# Annualization constant for hourly XAU/USD-style markets.
# Uses 24 hours × 5 trading days × 52 weeks; actual bar counts may vary by
# broker holidays, market closures, and missing data.
H1_BARS_PER_YEAR: int = 24 * 5 * 52

# ---------------------------------------------------------------------------
# Shared visualization palette (matplotlib + pyecharts)
# ---------------------------------------------------------------------------

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

#: Alias for interactive chart modules (`charts/`) — same set as ``EXCLUDE_COLS``.
EXCLUDED_FEATURE_COLS = EXCLUDE_COLS

# Core tabular features — price-action focused with minimal indicators.
# Keep in sync with config.toml [features].static_feature_cols.
CORE_STATIC_FEATURES: tuple[str, ...] = (
    # Trend
    "ema34_vs_ema89",
    "close_vs_ema_34",
    "trend_strength",
    # Momentum
    "return_1h",
    "return_4h",
    "macd_hist",
    "rsi_14",
    # Volatility / Regime
    "atr_14",
    "atr_percentile",
    "high_low_range_20",
    # Position / Location
    "price_dist_ratio",
    "price_position_20",
    "pivot_position",
    # Candle Structure
    "candle_body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    # Session
    "sess_london",
    "sess_overlap",
    # Volume / Activity
    "volume_zscore_20",
)
