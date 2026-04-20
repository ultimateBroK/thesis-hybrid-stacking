"""Interactive ECharts chart builders for thesis visualization.

Each function builds a pyecharts chart object that can be:
- Rendered via st_pyecharts() in Streamlit (pyecharts charts)
- Exported as HTML via chart.render("path.html")

Usage:
    from thesis.charts import build_candlestick_chart
    chart = build_candlestick_chart(ohlcv_df, config)
    chart.render("candlestick.html")
"""

from .backtest_charts import (
    build_duration_pnl_scatter,
    build_equity_drawdown_chart,
    build_monthly_returns_heatmap,
    build_pnl_histogram_chart,
    build_rolling_sharpe_chart,
    _compute_monthly_returns,
)
from .data import (
    COLORS,
    EXCLUDED_FEATURE_COLS,
    _get_feature_cols,
    load_session_data,
)
from .data_charts import (
    build_candlestick_chart,
    build_correlation_heatmap,
    build_feature_distributions_chart,
    build_label_distribution_chart,
)
from .model_charts import (
    build_confidence_distribution_chart,
    build_confusion_matrix_chart,
    build_feature_importance_chart,
    build_shap_chart,
)

__all__ = [
    # Constants
    "COLORS",
    "EXCLUDED_FEATURE_COLS",
    # Data loading
    "load_session_data",
    # Data charts
    "build_candlestick_chart",
    "build_correlation_heatmap",
    "build_label_distribution_chart",
    "build_feature_distributions_chart",
    # Model charts
    "build_confusion_matrix_chart",
    "build_confidence_distribution_chart",
    "build_feature_importance_chart",
    "build_shap_chart",
    # Backtest charts
    "build_equity_drawdown_chart",
    "build_pnl_histogram_chart",
    "build_monthly_returns_heatmap",
    "build_rolling_sharpe_chart",
    "build_duration_pnl_scatter",
    # Private (for backward compat)
    "_compute_monthly_returns",
    "_get_feature_cols",
]
