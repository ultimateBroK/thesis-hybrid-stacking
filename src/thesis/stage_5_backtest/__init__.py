"""Stage 5: CFD backtest simulation.

Public API:
    run_backtest         — full pipeline from Parquet files.
    run_backtest_from_data — from in-memory DataFrames.
    run_backtest_manual  — with explicit keyword parameters.
    HybridGRUStrategy    — strategy class for backtesting.py.
"""

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
