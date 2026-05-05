"""CFD backtest simulation package."""

from ._impl import (
    HybridGRUStrategy,
    run_backtest,
    run_backtest_from_data,
    run_backtest_manual,
)

__all__ = [
    "HybridGRUStrategy",
    "run_backtest",
    "run_backtest_from_data",
    "run_backtest_manual",
]
