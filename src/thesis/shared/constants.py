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
        "atr_14",
        "log_returns",
    ]
)

_EXCLUDE_COLS = EXCLUDE_COLS  # back-compat alias

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

EXCLUDED_FEATURE_COLS = EXCLUDE_COLS  # alias for chart modules


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

# Default LightGBM tabular features
CORE_STATIC_FEATURES: tuple[str, ...] = (
    "ema34_vs_ema89",
    "close_vs_ema_34",
    "adx_14",
    "ema_slope_20",
    "return_1h",
    "return_4h",
    "macd_hist_atr",
    "rsi_14",
    "atr_pct_close",
    "atr_ratio",
    "atr_percentile",
    "high_low_range_20",
    "price_dist_ratio",
    "price_position_20",
    "pivot_position",
    "vwap",
    "candle_body_ratio",
    "sess_asia",
    "sess_london",
    "sess_ny_am",
    "sess_ny_pm",
)
