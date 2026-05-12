"""CFD backtest simulation package."""

from .simulation import (
    BacktestResult,
    compute_backtest,
    run_backtest,
    run_backtest_from_data,
    run_backtest_manual,
)
from .strategy import MLSignalStrategy

__all__ = [
    "BacktestResult",
    "MLSignalStrategy",
    "compute_backtest",
    "run_backtest",
    "run_backtest_from_data",
    "run_backtest_manual",
]
