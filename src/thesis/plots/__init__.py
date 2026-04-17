"""Comprehensive thesis visualization — data exploration, model performance, backtest.

Generates all charts needed for the thesis report in one pass.

Usage:
    from thesis.plots import generate_all_charts
    generate_all_charts(config)
"""

import logging

from thesis.config import Config

from .data import _generate_data_charts
from .model import _generate_model_charts
from .backtest import _generate_backtest_charts

logger = logging.getLogger("thesis.visualize")


def generate_all_charts(config: Config) -> None:
    """Generate all thesis visualization charts.

    Args:
        config: Loaded application configuration.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        }
    )

    logger.info("Generating all thesis charts...")

    _generate_data_charts(config)
    _generate_model_charts(config)
    # Backtest charts are now handled by backtesting.py Bokeh HTML output.
    # See: {session_dir}/backtest/backtest_chart.html

    logger.info("All charts generated.")


__all__ = [
    "generate_all_charts",
    "_generate_data_charts",
    "_generate_model_charts",
    "_generate_backtest_charts",
]
