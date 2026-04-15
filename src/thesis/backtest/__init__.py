"""Stage 5: CFD backtest simulation via backtesting.py."""

from .strategy import (  # noqa: F401
    HybridGRUStrategy,
    run_backtest,
    run_backtest_from_data,
    _normalize_stats,
    _trades_to_list,
    _prepare_df,
    _run_bt,
)

__all__ = ["run_backtest", "run_backtest_from_data", "HybridGRUStrategy"]
