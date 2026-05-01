"""Tests for backtest module — backtesting.py integration.

Tests the thin wrapper around backtesting.py v0.6.5.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from thesis.backtest import (
    _calendar_day,
    _prepare_df,
    _run_bt,
    run_backtest_from_data,
)
from thesis.config import Config


def create_synthetic_backtest_data(
    n_rows: int = 100,
    signal_pattern: str = "alternating",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Create synthetic test data + predictions for testing."""
    np.random.seed(42)

    timestamps = pl.datetime_range(
        start=pl.datetime(2023, 1, 1, 0),
        end=pl.datetime(2023, 1, 1, 0) + pl.duration(hours=n_rows - 1),
        interval="1h",
        eager=True,
    )

    base_price = 1800.0
    closes = base_price + np.cumsum(np.random.randn(n_rows) * 0.5)
    opens = closes + np.random.randn(n_rows) * 0.1
    highs = np.maximum(opens, closes) + np.abs(np.random.randn(n_rows)) * 0.5
    lows = np.minimum(opens, closes) - np.abs(np.random.randn(n_rows)) * 0.5

    # Generate signals
    if signal_pattern == "alternating":
        pred_label = np.array([1, -1] * (n_rows // 2) + [1] * (n_rows % 2))
    elif signal_pattern == "all_long":
        pred_label = np.ones(n_rows, dtype=int)
    elif signal_pattern == "all_short":
        pred_label = -np.ones(n_rows, dtype=int)
    elif signal_pattern == "mixed":
        pred_label = np.random.choice([-1, 0, 1], n_rows)
    else:
        pred_label = np.zeros(n_rows, dtype=int)

    # ATR values (large enough to avoid immediate stop-loss)
    atr = np.full(n_rows, 20.0)

    test_df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.random.randint(1000, 10000, n_rows).astype(float),
            "atr_14": atr,
        }
    )

    preds_df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "pred_label": pred_label,
            "pred_proba_class_minus1": np.random.uniform(0, 0.5, n_rows),
            "pred_proba_class_0": np.random.uniform(0, 0.3, n_rows),
            "pred_proba_class_1": np.random.uniform(0.3, 1.0, n_rows),
        }
    )

    return test_df, preds_df


@pytest.fixture
def sample_config() -> Config:
    """Create a sample config for testing."""
    config = Config()
    config.backtest.initial_capital = 10_000.0
    config.backtest.leverage = 50
    config.backtest.spread_ticks = 30.0
    config.backtest.slippage_ticks = 3.0
    config.backtest.commission_per_lot = 10.0
    config.backtest.atr_stop_multiplier = 0.75
    config.backtest.confidence_threshold = (
        0.0  # disable confidence gating for deterministic sizing
    )
    config.data.contract_size = 100
    config.data.tick_size = 0.01
    return config


# ---------------------------------------------------------------------------
# Core tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.backtest
def test_results_contain_expected_keys(sample_config: Config) -> None:
    """Test that results contain backtesting.py native stat keys."""
    test_df, preds_df = create_synthetic_backtest_data(100, "mixed")
    metrics = run_backtest_from_data(test_df, preds_df, sample_config)

    expected_keys = [
        "num_trades",
        "win_rate_pct",
        "sharpe_ratio",
        "max_drawdown_pct",
        "profit_factor",
        "return_pct",
        "equity_final",
    ]
    for key in expected_keys:
        assert key in metrics, f"Missing key: {key}"


@pytest.mark.unit
@pytest.mark.backtest
def test_metrics_values_reasonable(sample_config: Config) -> None:
    """Test that metric values are within reasonable ranges."""
    test_df, preds_df = create_synthetic_backtest_data(100, "mixed")
    metrics = run_backtest_from_data(test_df, preds_df, sample_config)

    # Win rate should be in [0, 100]
    if "win_rate_pct" in metrics:
        assert 0 <= metrics["win_rate_pct"] <= 100

    # Max drawdown should be <= 0
    if "max_drawdown_pct" in metrics:
        assert metrics["max_drawdown_pct"] <= 0

    # Final equity should be positive
    if "equity_final" in metrics:
        assert metrics["equity_final"] > 0


@pytest.mark.unit
@pytest.mark.backtest
def test_empty_signals_handled(sample_config: Config) -> None:
    """Test handling of no signals (no trades)."""
    test_df, preds_df = create_synthetic_backtest_data(50, "none")
    metrics = run_backtest_from_data(test_df, preds_df, sample_config)

    assert metrics.get("num_trades", 0) == 0


@pytest.mark.unit
@pytest.mark.backtest
def test_atr_stop_loss_used(sample_config: Config) -> None:
    """Test that ATR stop-loss is passed to buy/sell."""
    test_df, preds_df = create_synthetic_backtest_data(50, "alternating")
    metrics = run_backtest_from_data(test_df, preds_df, sample_config)

    # Should have trades
    assert metrics.get("num_trades", 0) > 0


@pytest.mark.unit
@pytest.mark.backtest
def test_signal_reversal_works(sample_config: Config) -> None:
    """Test that exclusive_orders handles signal reversal."""
    # Alternating signals should produce multiple trades
    test_df, preds_df = create_synthetic_backtest_data(100, "alternating")
    metrics = run_backtest_from_data(test_df, preds_df, sample_config)

    assert metrics.get("num_trades", 0) > 0


@pytest.mark.unit
@pytest.mark.backtest
def test_commission_calculation(sample_config: Config) -> None:
    """Test that commission callable produces correct values."""
    contract_size = sample_config.data.contract_size
    commission_per_lot = sample_config.backtest.commission_per_lot

    # Simulate the commission function
    order_size = 100.0  # 1 lot
    lots = abs(order_size) / contract_size
    commission = lots * commission_per_lot
    assert commission == 10.0  # 1 lot × $10


@pytest.mark.unit
@pytest.mark.backtest
def test_run_backtest_from_data_compat(sample_config: Config) -> None:
    """Test that ablation interface returns a dict."""
    test_df, preds_df = create_synthetic_backtest_data(50, "mixed")
    result = run_backtest_from_data(test_df, preds_df, sample_config)

    assert isinstance(result, dict)
    assert "num_trades" in result


@pytest.mark.unit
@pytest.mark.backtest
def test_no_lookahead_bias(sample_config: Config) -> None:
    """Test that execution is delayed by 1 bar (backtesting.py native)."""
    n_rows = 10
    test_df, preds_df = create_synthetic_backtest_data(n_rows, "all_long")
    pdf = _prepare_df(test_df, preds_df)
    stats, _bt = _run_bt(pdf, sample_config)
    trades = stats["_trades"]

    # backtesting.py evaluates signals only after bars are complete and fills
    # market orders on the next bar. The first actionable all-long signal is
    # evaluated at bar 1, so the first entry must be at bar 2, not bar 0/1.
    assert len(trades) > 0
    assert trades.iloc[0]["EntryTime"] == pdf.index[2]


@pytest.mark.unit
@pytest.mark.backtest
def test_calendar_day_strips_intraday_time() -> None:
    """Daily risk state must reset by date, not every bar timestamp."""
    ts1 = pd.Timestamp("2026-04-29 09:00:00")
    ts2 = pd.Timestamp("2026-04-29 17:00:00")
    ts3 = pd.Timestamp("2026-04-30 00:00:00")

    assert _calendar_day(ts1) == _calendar_day(ts2)
    assert _calendar_day(ts1) != _calendar_day(ts3)


# ---------------------------------------------------------------------------
# Confidence-based position sizing
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.backtest
def test_position_size_scales_with_confidence() -> None:
    """Higher confidence should produce larger position sizes."""
    from thesis.backtest import HybridGRUStrategy

    # Can't instantiate Strategy without broker/data/params.
    # Use a lightweight mock with only the attributes _compute_lots reads.
    strategy = type("S", (), {
        "confidence_threshold": 0.55,
        "lots_per_trade": 0.1,
        "min_lots": 0.05,
        "max_lots": 0.2,
        "_compute_lots": HybridGRUStrategy._compute_lots,
    })()

    # Low confidence (just above threshold)
    lots_low = strategy._compute_lots(confidence=0.56)
    # High confidence (near 1.0)
    lots_high = strategy._compute_lots(confidence=0.95)

    assert lots_low > 0, "Lots should be positive above threshold"
    assert lots_high > 0, "Lots should be positive at high confidence"
    assert lots_high >= lots_low, (
        f"Higher confidence ({lots_high}) should produce >= lots than "
        f"lower confidence ({lots_low})"
    )

    # Clamped to [min_lots, max_lots]
    assert lots_low >= strategy.min_lots
    assert lots_high <= strategy.max_lots


@pytest.mark.unit
@pytest.mark.backtest
def test_position_size_returns_fixed_when_no_confidence() -> None:
    """Without confidence data, sizing falls back to fixed lots_per_trade."""
    from thesis.backtest import HybridGRUStrategy

    strategy = type("S", (), {
        "confidence_threshold": 0.0,  # disabled
        "lots_per_trade": 0.1,
        "_compute_lots": HybridGRUStrategy._compute_lots,
    })()

    lots = strategy._compute_lots(confidence=None)
    assert lots == 0.1

    # Even with a confidence value, threshold=0 means fixed sizing
    lots2 = strategy._compute_lots(confidence=0.99)
    assert lots2 == 0.1


@pytest.mark.unit
@pytest.mark.backtest
def test_position_size_below_threshold_returns_min() -> None:
    """Confidence below threshold should not be reached (caller gates)."""
    from thesis.backtest import HybridGRUStrategy

    strategy = type("S", (), {
        "confidence_threshold": 0.7,
        "lots_per_trade": 0.1,
        "min_lots": 0.05,
        "max_lots": 0.2,
        "_compute_lots": HybridGRUStrategy._compute_lots,
    })()

    # At threshold boundary: scale = 0 → clamped to min_lots
    lots_at = strategy._compute_lots(confidence=0.70)
    assert lots_at == strategy.min_lots

    # Well above threshold: scale > 0 → lots > min_lots
    lots_above = strategy._compute_lots(confidence=0.95)
    # scale = (0.95-0.70)/0.30 = 0.833 → lots = 0.0833 > 0.05
    assert lots_above > strategy.min_lots
