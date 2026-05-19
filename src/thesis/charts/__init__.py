"""Interactive ECharts chart builders.

Split by domain:
- data: OHLCV, features, labels charts
- model: prediction, confusion, importance charts
- backtest: equity, PnL, rolling Sharpe charts

Artifact loading and data preparation live in thesis.visualization.
"""

from thesis.charts.backtest import build_equity_drawdown_chart
from thesis.charts.data import (
    build_candlestick_chart,
    build_correlation_heatmap,
    build_feature_distribution_chart,
    build_label_distribution_chart,
)
from thesis.charts.model import (
    build_confidence_distribution_chart,
    build_confusion_matrix_chart,
    build_feature_importance_chart,
    build_model_comparison_chart,
)
from thesis.visualization.artifacts import load_dashboard_artifacts as load_session_data
from thesis.visualization.style import COLORS, EXCLUDED_FEATURE_COLS

__all__ = [
    "COLORS",
    "EXCLUDED_FEATURE_COLS",
    "load_session_data",
    "build_candlestick_chart",
    "build_correlation_heatmap",
    "build_label_distribution_chart",
    "build_feature_distribution_chart",
    "build_confusion_matrix_chart",
    "build_confidence_distribution_chart",
    "build_feature_importance_chart",
    "build_model_comparison_chart",
    "build_equity_drawdown_chart",
]
