"""CFD backtest simulation package."""

from .simulation import (
    run_backtest,
    run_backtest_from_data,
    run_backtest_manual,
)
from .strategy import MLSignalStrategy

__all__ = [
    "MLSignalStrategy",
    "run_backtest",
    "run_backtest_from_data",
    "run_backtest_manual",
]
