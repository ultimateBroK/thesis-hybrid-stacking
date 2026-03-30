"""Regression tests for CFD simulator behavior."""

from datetime import datetime, timedelta
from types import SimpleNamespace

import polars as pl
import pytest

from thesis.backtest.cfd_simulator import _simulate_trades


def _build_config(tmp_path):
    backtest = SimpleNamespace(
        initial_capital=10000.0,
        margin_call_level=0.5,
        stop_out_level=0.2,
        spread_pips=0.0,
        slippage_pips=0.0,
        risk_per_trade=0.001,
        leverage=100,
        backtest_results_path=str(tmp_path / "backtest_results.json"),
    )
    return SimpleNamespace(backtest=backtest)


def test_trade_exit_uses_entry_position_size(tmp_path):
    """Exit PnL must use the size locked at trade entry."""
    start = datetime(2024, 1, 1, 0, 0, 0)
    df = pl.DataFrame(
        {
            "timestamp": [start, start + timedelta(hours=1), start + timedelta(hours=2)],
            "close": [100.0, 101.0, 100.5],
            "atr_14": [0.5, 5.0, 5.0],
            "pred_proba_class_minus1": [0.1, 0.9, 0.2],
            "pred_proba_class_0": [0.2, 0.05, 0.2],
            "pred_proba_class_1": [0.9, 0.05, 0.8],
        }
    )

    results = _simulate_trades(df, _build_config(tmp_path))

    first_trade = results["trades"][0]

    assert first_trade["position"] == "long"
    assert first_trade["position_size"] == pytest.approx(0.2)
    assert first_trade["pnl_dollar"] == pytest.approx(20.0)
