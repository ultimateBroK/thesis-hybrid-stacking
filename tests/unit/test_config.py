"""Tests for configuration consistency between sections.

Verifies that labels and backtest barrier multipliers stay aligned,
since the backtest SL/TP must match the label barriers used to
generate the signals being traded.
"""

import pytest

from thesis.config import Config


@pytest.mark.unit
def test_label_backtest_barrier_consistency() -> None:
    """Default config must have matching label and backtest multipliers."""
    cfg = Config()

    assert cfg.labels.atr_tp_multiplier == cfg.backtest.atr_tp_multiplier, (
        f"labels.atr_tp_multiplier ({cfg.labels.atr_tp_multiplier}) "
        f"!= backtest.atr_tp_multiplier ({cfg.backtest.atr_tp_multiplier})"
    )

    assert cfg.labels.atr_sl_multiplier == cfg.backtest.atr_stop_multiplier, (
        f"labels.atr_sl_multiplier ({cfg.labels.atr_sl_multiplier}) "
        f"!= backtest.atr_stop_multiplier ({cfg.backtest.atr_stop_multiplier})"
    )
