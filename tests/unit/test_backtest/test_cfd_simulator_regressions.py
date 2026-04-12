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
    labels = SimpleNamespace(
        atr_multiplier_tp=1.5,
        atr_multiplier_sl=1.5,
    )
    return SimpleNamespace(backtest=backtest, labels=labels)


def test_trade_exit_uses_entry_position_size(tmp_path):
    """Exit PnL must use the size locked at trade entry."""
    start = datetime(2024, 1, 1, 0, 0, 0)
    df = pl.DataFrame(
        {
            "timestamp": [
                start,
                start + timedelta(hours=1),
                start + timedelta(hours=2),
            ],
            "close": [100.0, 101.0, 100.5],
            "high": [100.5, 102.0, 101.0],
            "low": [99.5, 99.0, 99.5],
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
    # SL is checked first (conservative bias): entry=100, atr=0.5
    # SL = 100 - 0.5*1.5 = 99.25; bar 1 low=99.0 < 99.25 → stop_loss
    assert first_trade["exit_reason"] == "stop_loss"
