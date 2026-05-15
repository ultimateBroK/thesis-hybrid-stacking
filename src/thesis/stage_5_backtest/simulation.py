"""CFD backtest via backtesting.py FractionalBacktest.

ATR SL/TP multipliers must match label barriers — same ATR multiple.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import polars as pl

from thesis.shared.config import Config
from thesis.shared.schemas import BacktestMetrics, TradeRecord
from thesis.shared.ui import console
from thesis.stage_5_backtest.persistence import (
    _log_core_backtest_metrics,
    _normalize_stats,
    _save_bokeh_chart,
    _save_equity_curve_csv,
    _save_json_results,
    _save_trade_details_csv,
    _trades_to_list,
)
from thesis.stage_5_backtest.runners import (
    _prepare_df,
    _run_fractional_backtest,
)

logger = logging.getLogger("thesis.backtest")

BacktestResult = tuple[BacktestMetrics, list[TradeRecord], object]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_backtest_data(config: Config) -> tuple[pd.DataFrame, str]:
    """Load predictions and price data for backtesting."""
    preds_path = Path(config.paths.predictions)
    if not preds_path.exists():
        raise FileNotFoundError(f"Predictions not found: {preds_path}")
    with console.status(f"[cyan]Loading predictions[/] {preds_path}"):
        preds_df = pl.read_csv(preds_path)

    test_path = Path(config.paths.test_data)
    is_static = config.validation.method == "static"
    labels_path = Path(config.paths.labels)
    if test_path.exists() and is_static:
        source = str(test_path)
        with console.status(f"[cyan]Loading static test data[/] {test_path}"):
            test_df = pl.read_parquet(test_path)
    else:
        if test_path.exists() and not is_static:
            logger.warning(
                "Static test file found (%s) but walk-forward method='%s' — ignoring",
                test_path,
                config.validation.method,
            )
        if not labels_path.exists():
            raise FileNotFoundError(
                f"Neither test data ({test_path}) nor labels ({labels_path}) found"
            )
        source = str(labels_path)
        logger.info("Walk-forward mode: OOF predictions + labeled data")
        with console.status(f"[cyan]Loading labels for backtest[/] {labels_path}"):
            test_df = pl.read_parquet(labels_path)

    pdf = _prepare_df(
        test_df, preds_df, test_source=source, preds_source=str(preds_path)
    )
    return _apply_oos_date_filter(pdf, config), source


def _apply_oos_date_filter(pdf: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Filter backtest bars to OOS date range."""
    bc = config.backtest
    if bc.oob_start_date:
        start_ts = pd.Timestamp(bc.oob_start_date)
        pdf = pdf[pdf.index >= start_ts]
        logger.info("OOS start filter: %s → %d bars", bc.oob_start_date, len(pdf))
    if bc.oob_end_date:
        end_ts = pd.Timestamp(bc.oob_end_date)
        pdf = pdf[pdf.index <= end_ts]
        logger.info("OOS end filter: %s → %d bars", bc.oob_end_date, len(pdf))
    if bc.oob_start_date or bc.oob_end_date:
        logger.info(
            "OOS date range: %s to %s (%d bars)",
            bc.oob_start_date or "start",
            bc.oob_end_date or "end",
            len(pdf),
        )
    return pdf


# ---------------------------------------------------------------------------
# Pure computation
# ---------------------------------------------------------------------------


def compute_backtest(config: Config) -> BacktestResult:
    """Load data, run FractionalBacktest, return normalized results.

    No files written. Returns (metrics dict, trades list, bt_engine).
    """
    pdf, _ = _load_backtest_data(config)
    logger.info("Confidence threshold: %.2f", config.backtest.confidence_threshold)
    with console.status("[cyan]Running CFD backtest[/]"):
        stats, bt = _run_fractional_backtest(pdf, config)

    metrics = _normalize_stats(stats)
    trades = _trades_to_list(
        stats.get("_trades", pd.DataFrame()),
        commission_per_lot=config.backtest.commission_per_lot,
        contract_size=config.data.contract_size,
    )
    return metrics, trades, bt


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _persist_backtest_results(
    metrics: BacktestMetrics,
    trades: list[TradeRecord],
    bt_engine: object,
    config: Config,
) -> None:
    """Write metrics JSON, trade-detail CSV, equity-curve CSV, Bokeh HTML."""
    out_path = Path(config.paths.backtest_results)
    _save_json_results(metrics, trades, out_path)
    if trades:
        _save_trade_details_csv(trades, out_path.parent)
        _save_equity_curve_csv(trades, out_path.parent, config.backtest.initial_capital)
    _log_core_backtest_metrics(metrics, config.backtest.initial_capital)
    session_dir = Path(config.paths.session_dir) if config.paths.session_dir else None
    _save_bokeh_chart(bt_engine, metrics, session_dir)


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def run_backtest(config: Config) -> None:
    """Run full CFD backtest from files in config.

    Walk-forward: OOF predictions + labeled dataset.
    Static: traditional test split file.
    Writes metrics JSON, trade-detail CSV, equity-curve CSV, Bokeh HTML.
    """
    metrics, trades, bt_engine = compute_backtest(config)
    _persist_backtest_results(metrics, trades, bt_engine, config)


# ---------------------------------------------------------------------------
# In-memory entry points
# ---------------------------------------------------------------------------


def run_backtest_from_data(
    test_df: pl.DataFrame,
    preds_df: pl.DataFrame,
    config: Config,
) -> BacktestResult:
    """Run backtest using in-memory Polars DataFrames."""
    pdf = _prepare_df(test_df, preds_df)
    stats, bt = _run_fractional_backtest(pdf, config)
    metrics = _normalize_stats(stats)
    trades = _trades_to_list(
        stats.get("_trades", pd.DataFrame()),
        commission_per_lot=config.backtest.commission_per_lot,
        contract_size=config.data.contract_size,
    )
    return metrics, trades, bt


def run_backtest_manual(
    test_df: pl.DataFrame,
    preds_df: pl.DataFrame,
    *,
    leverage: int = 100,
    lots_per_trade: float = 0.2,
    min_lots: float = 0.1,
    max_lots: float = 0.5,
    confidence_threshold: float = 0.0,
    spread_ticks: int = 35,
    slippage_ticks: int = 5,
    commission_per_lot: float = 10.0,
    atr_stop_multiplier: float = 1.0,
    atr_tp_multiplier: float = 2.0,
    horizon_bars: int = 10,
    contract_size: int = 100,
    tick_size: float = 0.01,
    initial_capital: float = 10_000.0,
    max_drawdown_cutoff: float = 0.50,
    dd_cooldown_bars: int = 12,
    max_open_positions: int = 1,
    daily_loss_limit: float = 0.03,
    min_bars_between_trades: int = 6,
) -> tuple[BacktestMetrics, list[TradeRecord]]:
    """Run backtest with manual params (no Config required)."""
    from thesis.stage_5_backtest.runners import (
        _compute_spread_rate,
        _create_fractional_backtest,
        _make_commission_fn,
    )

    pdf = _prepare_df(test_df, preds_df)

    median_price = float(pdf["Close"].median())
    spread_total = _compute_spread_rate(
        spread_ticks, slippage_ticks, tick_size, median_price
    )

    commission_fn = _make_commission_fn(commission_per_lot, contract_size)
    bt = _create_fractional_backtest(
        pdf,
        cash=initial_capital,
        spread=spread_total,
        commission_fn=commission_fn,
        leverage=leverage,
    )

    stats = bt.run(
        atr_stop_mult=atr_stop_multiplier,
        atr_tp_mult=atr_tp_multiplier,
        lots_per_trade=lots_per_trade,
        min_lots=min_lots,
        max_lots=max_lots,
        confidence_threshold=confidence_threshold,
        contract_size=contract_size,
        horizon_bars=horizon_bars,
        max_drawdown_cutoff=max_drawdown_cutoff,
        dd_cooldown_bars=dd_cooldown_bars,
        max_open_positions=max_open_positions,
        daily_loss_limit=daily_loss_limit,
        min_bars_between_trades=min_bars_between_trades,
    )

    metrics = _normalize_stats(stats)
    trades = _trades_to_list(
        stats.get("_trades", pd.DataFrame()),
        commission_per_lot=commission_per_lot,
        contract_size=contract_size,
    )

    return metrics, trades
