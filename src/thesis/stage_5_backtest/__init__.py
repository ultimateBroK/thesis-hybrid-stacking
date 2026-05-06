"""CFD backtest simulation package."""

from ._impl import (
    run_backtest,
    run_backtest_from_data,
    run_backtest_manual,
)
from ._strategy import HybridGRUStrategy

__all__ = [
    "HybridGRUStrategy",
    "run_backtest",
    "run_backtest_from_data",
    "run_backtest_manual",
]
