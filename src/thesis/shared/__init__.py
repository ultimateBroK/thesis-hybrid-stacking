"""Shared modules for the thesis pipeline.

All stable, cross-cutting code lives here so stage implementations can
import from ``thesis.shared`` without coupling to specific pipeline
module locations.
"""

from thesis.shared.config import (
    BacktestConfig,
    Config,
    DataConfig,
    DataRangeConfig,
    FeaturesConfig,
    LabelsConfig,
    LightGBMConfig,
    ModelConfig,
    MultiTimeframeConfig,
    PathsConfig,
    RandomForestConfig,
    ReportFiguresConfig,
    StackingConfig,
    ValidationConfig,
    WorkflowConfig,
    get_config,
    load_config,
    reload_config,
)
from thesis.shared.constants import (
    ATR_HIGH_QUANTILE,
    ATR_LOW_QUANTILE,
    CALIB_LR,
    CALIB_MAX_ITER,
    CENSORED_LABEL,
    CHART_COLORS,
    CORE_STATIC_FEATURES,
    DIST_SHIFT_CLIP_MAX,
    DIST_SHIFT_CLIP_MIN,
    ECE_N_BINS,
    EXCLUDE_COLS,
    EXCLUDED_FEATURE_COLS,
    FEATURE_EPS,
    H1_BARS_PER_YEAR,
    LABEL_PROFITABILITY_WARN_PCT,
    ROUNDTRIP_MULT,
    SAMPLE_WEIGHT_MIN,
    STD_EPS,
)
from thesis.shared.utils import (
    STAGE_LABELS,
    STAGE_STYLES,
    SimpleConsole,
    console,
    stage_header,
    stage_skip,
)
from thesis.shared.zones import ZONE_COLORS, get_metric_zone, is_extreme_value

__all__ = [
    # config
    "Config",
    "load_config",
    "get_config",
    "reload_config",
    "DataConfig",
    "DataRangeConfig",
    "ValidationConfig",
    "MultiTimeframeConfig",
    "FeaturesConfig",
    "LabelsConfig",
    "LightGBMConfig",
    "ModelConfig",
    "RandomForestConfig",
    "StackingConfig",
    "BacktestConfig",
    "WorkflowConfig",
    "PathsConfig",
    "ReportFiguresConfig",
    # constants
    "EXCLUDE_COLS",
    "EXCLUDED_FEATURE_COLS",
    "H1_BARS_PER_YEAR",
    "CHART_COLORS",
    "SAMPLE_WEIGHT_MIN",
    "ATR_LOW_QUANTILE",
    "ATR_HIGH_QUANTILE",
    "LABEL_PROFITABILITY_WARN_PCT",
    "ROUNDTRIP_MULT",
    "CENSORED_LABEL",
    "DIST_SHIFT_CLIP_MIN",
    "DIST_SHIFT_CLIP_MAX",
    "FEATURE_EPS",
    "STD_EPS",
    "ECE_N_BINS",
    "CALIB_LR",
    "CALIB_MAX_ITER",
    "CORE_STATIC_FEATURES",
    # ui (from utils)
    "SimpleConsole",
    "console",
    "STAGE_STYLES",
    "STAGE_LABELS",
    "stage_header",
    "stage_skip",
    # zones
    "get_metric_zone",
    "is_extreme_value",
    "ZONE_COLORS",
]
