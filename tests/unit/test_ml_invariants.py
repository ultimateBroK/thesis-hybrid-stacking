"""ML invariants: calibration and reliability tests for the backtest engine.

Tests that verify fundamental ML/trading invariants hold:
1. Perfect predictions on trending data → positive Sharpe (edge detection)
2. Random predictions on random data → zero Sharpe (no spurious edge)
3. Always-LONG strategy → return direction matches market direction
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from thesis.stage_5_backtest import run_backtest_manual


def _make_synthetic_data(
    n_rows: int = 300,
    base_price: float = 2000.0,
    drift_per_bar: float = 0.0,
    noise_std: float = 0.5,
    pred_label: np.ndarray | None = None,
    atr_val: float = 40.0,
    seed: int = 42,
    *,
    freq: str = "1d",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Create synthetic OHLCV data and predictions for invariant tests.

    Args:
        n_rows: Number of bars.
        base_price: Starting price level.
        drift_per_bar: Deterministic drift added to each bar's close.
        noise_std: Standard deviation of Gaussian price noise.
        pred_label: Array of prediction labels (-1, 0, 1).  Defaults to
            all-zero (hold) if None.
        atr_val: Constant ATR(14) value for all bars.
        seed: NumPy random seed for reproducibility.
        freq: Polars duration string for bar interval (``"1d"`` or ``"1h"``).

    Returns:
        Tuple of (test_df, preds_df) — polar DataFrames suitable for
        ``run_backtest_manual``.
    """
    np.random.seed(seed)

    timestamps = pl.datetime_range(
        start=pl.datetime(2024, 1, 1, 0),
        end=pl.datetime(2024, 1, 1, 0)
        + pl.duration(**{("days" if freq == "1d" else "hours"): n_rows - 1}),
        interval=freq,
        eager=True,
    )

    closes = base_price + np.cumsum(
        np.full(n_rows, drift_per_bar) + np.random.randn(n_rows) * noise_std
    )
    opens = closes + np.random.randn(n_rows) * (0.05 if freq == "1h" else 0.5)
    highs = np.maximum(opens, closes) + np.abs(np.random.randn(n_rows)) * (
        0.2 if freq == "1h" else 2.0
    )
    lows = np.minimum(opens, closes) - np.abs(np.random.randn(n_rows)) * (
        0.2 if freq == "1h" else 2.0
    )

    atr = np.full(n_rows, atr_val)

    test_df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.ones(n_rows) * 5000.0,
            "atr_14": atr,
        }
    )

    if pred_label is None:
        pred_label = np.zeros(n_rows, dtype=int)

    preds_df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "pred_label": pred_label,
            "pred_proba_class_minus1": np.zeros(n_rows),
            "pred_proba_class_0": np.zeros(n_rows),
            "pred_proba_class_1": np.ones(n_rows) * 0.9,
        }
    )

    return test_df, preds_df


# ---------------------------------------------------------------------------
# Invariant 1: Perfect predictions → positive Sharpe
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_perfect_predictions_produce_positive_sharpe() -> None:
    """Perfect predictions on a strong uptrend with zero costs → Sharpe > 2.0.

    Uses daily bars (~1 year) with a consistent $5/bar upward drift and
    all-long predictions.  Zero costs + wide ATR stops isolate the directional
    signal quality.  The daily frequency aligns with backtesting.py's Sharpe
    annualization so the value is interpretable.
    """
    n_rows = 300
    test_df, preds_df = _make_synthetic_data(
        n_rows=n_rows,
        base_price=2000.0,
        drift_per_bar=5.0,
        noise_std=0.2,
        pred_label=np.ones(n_rows, dtype=int),
        atr_val=200.0,
        freq="1d",
    )

    metrics, _trades = run_backtest_manual(
        test_df,
        preds_df,
        leverage=50,
        lots_per_trade=0.3,
        min_lots=0.1,
        max_lots=0.5,
        confidence_threshold=0.0,
        spread_ticks=0,
        slippage_ticks=0,
        commission_per_lot=0,
        atr_stop_multiplier=1.0,
        atr_tp_multiplier=0,
        horizon_bars=0,
        initial_capital=10_000.0,
    )

    sharpe = metrics.get("sharpe_ratio", 0)
    assert sharpe > 2.0, (
        f"Perfect predictions on uptrend should produce Sharpe > 2.0, got {sharpe:.3f}"
    )
    assert metrics.get("num_trades", 0) >= 1, "Expected at least 1 trade"


# ---------------------------------------------------------------------------
# Invariant 2: Random predictions → zero Sharpe (no spurious edge)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_random_predictions_produce_zero_sharpe() -> None:
    """Random predictions on zero-drift random walk → Sharpe ≈ 0.

    Uses daily bars with ~1 year of data.  A short horizon forces frequent
    position turnover to exercise the random signal across many independent
    trades over a full calendar year, giving the annualize Sharpe enough
    statistical power.
    """
    n_rows = 300
    np.random.seed(99)
    pred_label = np.random.choice([-1, 1], n_rows)

    test_df, preds_df = _make_synthetic_data(
        n_rows=n_rows,
        base_price=2000.0,
        drift_per_bar=0.0,
        noise_std=5.0,
        pred_label=pred_label,
        atr_val=200.0,
        seed=99,
        freq="1d",
    )

    metrics, _trades = run_backtest_manual(
        test_df,
        preds_df,
        leverage=50,
        lots_per_trade=0.3,
        min_lots=0.1,
        max_lots=0.5,
        confidence_threshold=0.0,
        spread_ticks=0,
        slippage_ticks=0,
        commission_per_lot=0,
        atr_stop_multiplier=1.0,
        atr_tp_multiplier=0,
        horizon_bars=5,
        initial_capital=10_000.0,
    )

    sharpe = metrics.get("sharpe_ratio", 0)
    assert -0.5 < sharpe < 0.5, (
        f"Random predictions should produce Sharpe ≈ 0 (±0.5), got {sharpe:.3f}"
    )


# ---------------------------------------------------------------------------
# Invariant 3: Always-LONG return matches market direction
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_always_long_matches_market_return() -> None:
    """An always-LONG strategy on an uptrending market → positive return.

    The strategy enters one long position early and holds to the end (wide
    ATR stops prevent premature exit).  Under zero costs the return_pct
    must be positive, matching the buy-and-hold return direction.
    """
    n_rows = 300
    test_df, preds_df = _make_synthetic_data(
        n_rows=n_rows,
        base_price=2000.0,
        drift_per_bar=3.0,
        noise_std=1.0,
        pred_label=np.ones(n_rows, dtype=int),
        atr_val=200.0,
        freq="1d",
    )

    metrics, _trades = run_backtest_manual(
        test_df,
        preds_df,
        leverage=10,
        lots_per_trade=0.2,
        min_lots=0.1,
        max_lots=0.5,
        confidence_threshold=0.0,
        spread_ticks=0,
        slippage_ticks=0,
        commission_per_lot=0,
        atr_stop_multiplier=3.0,
        atr_tp_multiplier=0,
        horizon_bars=0,
        initial_capital=10_000.0,
    )

    return_pct = metrics.get("return_pct", 0)
    assert return_pct > 0, (
        f"Always-LONG on uptrending market must have positive return, "
        f"got {return_pct:.2f}%"
    )
    assert metrics.get("num_trades", 0) >= 1, "Expected at least 1 trade"
