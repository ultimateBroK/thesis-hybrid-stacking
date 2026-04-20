"""Stage 5: CFD backtest simulation via backtesting.py.

Public API:
    run_backtest         — full pipeline from Parquet files.
    run_backtest_from_data — from in-memory DataFrames.
    run_backtest_manual  — with explicit keyword parameters.
    HybridGRUStrategy    — strategy class for backtesting.py.
"""

from .runners import (  # noqa: F401
    run_backtest,
    run_backtest_from_data,
    run_backtest_manual,
    _prepare_df,
    _run_bt,
)
from .stats import _normalize_stats  # noqa: F401
from .strategy import HybridGRUStrategy  # noqa: F401
from .persistence import _trades_to_list  # noqa: F401

__all__ = [
    "run_backtest",
    "run_backtest_from_data",
    "run_backtest_manual",
    "HybridGRUStrategy",
]
