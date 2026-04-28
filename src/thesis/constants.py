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
#:  - tp_price/sl_price/touched_bar → label-derived, pure look-ahead
#:  - open_right/high_right/low_right/close_right → label-derived look-ahead
#:  - open/high/low/close/volume → raw OHLCV, excluded to avoid raw price leakage
#:  - avg_spread/tick_count/dead_hour → microstructure columns kept for backtest
#:    but not useful as ML features in their raw form
#:  - log_returns → GRU sequence input; excluded from the *static* LightGBM features
#:    to avoid double-counting the information already encoded in GRU hidden states
EXCLUDE_COLS: frozenset[str] = frozenset(
    [
        "timestamp",
        "label",
        "tp_price",
        "sl_price",
        "touched_bar",
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
        "dead_hour",
        "log_returns",  # GRU sequence input — not a static feature for LightGBM
    ]
)

# Backward-compatible private alias used by internal modules
_EXCLUDE_COLS = EXCLUDE_COLS

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
