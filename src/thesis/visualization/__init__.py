"""Shared visualization layer: artifact loading, summaries, data prep, styling."""

from thesis.visualization.artifacts import load_dashboard_artifacts
from thesis.visualization.chart_data import (
    compute_normalized_confusion_matrix,
    make_kline_series,
    make_volume_series,
    parse_chart_timestamps,
    prepare_candlestick_data,
)
from thesis.visualization.style import CHART_COLORS, COLORS, EXCLUDED_FEATURE_COLS
from thesis.visualization.summaries import compute_prediction_summary

__all__ = [
    "CHART_COLORS",
    "COLORS",
    "EXCLUDED_FEATURE_COLS",
    "compute_normalized_confusion_matrix",
    "compute_prediction_summary",
    "load_dashboard_artifacts",
    "make_kline_series",
    "make_volume_series",
    "parse_chart_timestamps",
    "prepare_candlestick_data",
]
