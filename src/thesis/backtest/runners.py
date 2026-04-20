"""Backtest initialization, execution, and public API.

Provides the pipeline for preparing DataFrames, configuring backtesting.py,
running the HybridGRUStrategy, and exposing three public entry points:

- ``run_backtest`` — full pipeline from Parquet files (used by thesis pipeline).
- ``run_backtest_from_data`` — from in-memory DataFrames (used by ablation).
- ``run_backtest_manual`` — with explicit parameters (used by dashboard).
"""

import logging
from pathlib import Path
from typing import Callable

import pandas as pd
import polars as pl
from backtesting.lib import FractionalBacktest

from thesis.backtest.persistence import (
    _save_bokeh_chart,
    _save_equity_curve_csv,
    _save_json_results,
    _save_trade_details_csv,
    _trades_to_list,
)
from thesis.backtest.stats import _normalize_stats
from thesis.backtest.strategy import HybridGRUStrategy
from thesis.config import Config

logger = logging.getLogger("thesis.backtest")


# Data Preparation


def _prepare_df(test_df: pl.DataFrame, preds_df: pl.DataFrame) -> pd.DataFrame:
    """Prepare a pandas DataFrame merging market data and predictions.

    Renames price columns to backtesting.py's expected PascalCase format
    (Open, High, Low, Close, Volume) and merges prediction columns from
    the model output.

    Args:
        test_df: Market data with timestamp, OHLCV, and atr_14 columns.
        preds_df: Predictions with timestamp, pred_label, and optional
            pred_proba_class_* columns.

    Returns:
        Pandas DataFrame indexed by timestamp (DatetimeIndex) with renamed
        price columns and merged prediction columns.

    Raises:
        ValueError: If pred_label is missing from preds_df or atr_14 is
            missing from the merged result.
    """
    test = test_df.with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
    preds = preds_df.with_columns(pl.col("timestamp").cast(pl.Datetime("us")))

    if "pred_label" not in preds.columns:
        raise ValueError("Predictions must contain 'pred_label' column")

    pred_cols = ["timestamp", "pred_label"]
    for col in [
        "pred_proba_class_minus1",
        "pred_proba_class_0",
        "pred_proba_class_1",
    ]:
        if col in preds.columns:
            pred_cols.append(col)

    merged = test.join(preds.select(pred_cols), on="timestamp", how="inner")

    if "atr_14" not in merged.columns:
        raise ValueError(
            "atr_14 column not found in test data. "
            "Ensure feature engineering includes ATR before backtest."
        )

    logger.info("Backtest bars: %d", len(merged))

    pdf = merged.to_pandas()
    pdf = pdf.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    if "Volume" not in pdf.columns:
        pdf["Volume"] = 0

    pdf = pdf.set_index("timestamp")
    pdf.index = pd.DatetimeIndex(pdf.index)

    return pdf


# Backtest Configuration Helpers


def _compute_spread_rate(
    bc: "Config.BacktestConfig",
    dc: "Config.DataConfig",
    median_price: float,
) -> float:
    """Convert tick-based spread to relative rate for backtesting.py.

    Args:
        bc: Backtest configuration with spread_ticks and slippage_ticks.
        dc: Data configuration with tick_size.
        median_price: Median close price used as normalization denominator.

    Returns:
        Relative spread rate as a fraction (e.g., 0.0003 for 3 pips on XAUUSD).
    """
    total_ticks = bc.spread_ticks + bc.slippage_ticks
    return total_ticks * dc.tick_size / median_price


def _build_commission_fn(
    bc: "Config.BacktestConfig",
    dc: "Config.DataConfig",
) -> "Callable[[float, float], float]":
    """Build a commission function closure for backtesting.py.

    Args:
        bc: Backtest configuration with commission_per_lot.
        dc: Data configuration with contract_size.

    Returns:
        A commission function that takes (order_size, price) and returns
        commission in dollars.
    """

    def commission_fn(order_size: float, price: float) -> float:
        lots = abs(order_size) / dc.contract_size
        return lots * bc.commission_per_lot

    return commission_fn


def _init_backtest(
    pdf: pd.DataFrame,
    bc: "Config.BacktestConfig",
    dc: "Config.DataConfig",
    spread: float,
    commission_fn: "Callable[[float, float], float]",
) -> FractionalBacktest:
    """Construct a FractionalBacktest instance without running it.

    Args:
        pdf: Prepared pandas DataFrame with price and prediction columns.
        bc: Backtest configuration.
        dc: Data configuration.
        spread: Pre-computed relative spread rate.
        commission_fn: Commission function from _build_commission_fn.

    Returns:
        Configured FractionalBacktest instance ready for .run().
    """
    margin = 1.0 / bc.leverage
    return FractionalBacktest(
        pdf,
        HybridGRUStrategy,
        cash=bc.initial_capital,
        spread=spread,
        commission=commission_fn,
        margin=margin,
        exclusive_orders=True,
        finalize_trades=True,
        fractional_unit=1.0,
    )


def _run_bt(pdf: pd.DataFrame, config: Config) -> tuple[pd.Series, FractionalBacktest]:
    """Run a backtest using HybridGRUStrategy with extracted helpers.

    Args:
        pdf: Prepared DataFrame with market data and predictions.
        config: Application configuration with backtest and data sections.

    Returns:
        Tuple of (backtest statistics Series, Backtest instance).
    """
    bc = config.backtest
    dc = config.data

    median_price = float(pdf["Close"].median())
    spread = _compute_spread_rate(bc, dc, median_price)
    commission_fn = _build_commission_fn(bc, dc)
    bt = _init_backtest(pdf, bc, dc, spread, commission_fn)

    stats = bt.run(
        atr_stop_mult=bc.atr_stop_multiplier,
        lots_per_trade=bc.lots_per_trade,
        confidence_threshold=bc.confidence_threshold,
        contract_size=dc.contract_size,
        horizon_bars=config.labels.horizon_bars,
        auto_lot_sizing=bc.auto_lot_sizing,
        risk_per_trade_pct=bc.risk_per_trade_pct,
        min_lot_size=bc.min_lot_size,
        max_lot_size=bc.max_lot_size,
        enable_performance_adjustment=bc.enable_performance_adjustment,
        enable_volatility_adjustment=bc.enable_volatility_adjustment,
        max_capital_risk_pct=bc.max_capital_risk_pct,
        performance_multiplier=bc.performance_multiplier,
        performance_reduction=bc.performance_reduction,
    )
    return stats, bt


# Public API


def run_backtest(config: Config) -> None:
    """Run a full CFD backtest from files specified in config and persist results.

    Loads test market data and model predictions from configured Parquet paths,
    validates required inputs, prepares merged market/prediction data, executes
    the backtest using the configured strategy and risk/cost settings, and writes
    outputs to disk.

    Written outputs include:
        - JSON file with normalized metrics and trade records.
        - Optional trades detail CSV and equity-curve CSV when trades are present.
        - Optional Bokeh HTML chart under the configured session directory.

    Args:
        config: Application configuration object containing paths and
            backtest/data settings (must provide paths.test_data,
            paths.predictions, paths.backtest_results; optional
            paths.session_dir).
    """
    test_path = Path(config.paths.test_data)
    preds_path = Path(config.paths.predictions)
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_path}")
    if not preds_path.exists():
        raise FileNotFoundError(f"Predictions not found: {preds_path}")

    test_df = pl.read_parquet(test_path)
    preds_df = pl.read_parquet(preds_path)

    pdf = _prepare_df(test_df, preds_df)
    logger.info("Confidence threshold: %.2f", config.backtest.confidence_threshold)
    stats, bt = _run_bt(pdf, config)

    metrics = _normalize_stats(stats)
    trades = _trades_to_list(
        stats["_trades"],
        commission_per_lot=config.backtest.commission_per_lot,
        contract_size=config.data.contract_size,
    )

    out_path = Path(config.paths.backtest_results)
    _save_json_results(metrics, trades, out_path)

    if trades:
        _save_trade_details_csv(trades, out_path.parent)
        initial_capital = (
            config.backtest.initial_capital
            if hasattr(config.backtest, "initial_capital")
            else 10_000.0
        )
        _save_equity_curve_csv(trades, out_path.parent, initial_capital)

    logger.info("=== BACKTEST RESULTS ===")
    for k, v in metrics.items():
        logger.info("  %s: %s", k, v)

    session_dir = Path(config.paths.session_dir) if config.paths.session_dir else None
    _save_bokeh_chart(bt, stats, session_dir)


def run_backtest_from_data(
    test_df: pl.DataFrame,
    preds_df: pl.DataFrame,
    config: Config,
) -> dict:
    """Run the full backtest pipeline using in-memory Polars DataFrames.

    Args:
        test_df: Market/test data containing price columns and atr_14.
        preds_df: Predictions data containing timestamp and pred_label
            (optional pred_proba_* columns allowed).
        config: Configuration object with backtest, data, and paths sections.

    Returns:
        Normalized metrics dictionary extracted from the backtest results.
    """
    pdf = _prepare_df(test_df, preds_df)
    stats, _ = _run_bt(pdf, config)
    return _normalize_stats(stats)


def run_backtest_manual(
    test_df: pl.DataFrame,
    preds_df: pl.DataFrame,
    *,
    leverage: int = 100,
    lots_per_trade: float = 1.0,
    confidence_threshold: float = 0.0,
    spread_ticks: int = 35,
    slippage_ticks: int = 5,
    commission_per_lot: float = 10.0,
    atr_stop_multiplier: float = 0.75,
    auto_lot_sizing: bool = False,
    risk_per_trade_pct: float = 1.0,
    min_lot_size: float = 0.1,
    max_lot_size: float = 10.0,
    enable_performance_adjustment: bool = True,
    enable_volatility_adjustment: bool = True,
    max_capital_risk_pct: float = 10.0,
    performance_multiplier: float = 1.2,
    performance_reduction: float = 0.8,
    horizon_bars: int = 10,
    contract_size: int = 100,
    tick_size: float = 0.01,
    initial_capital: float = 10_000.0,
) -> tuple[dict, list[dict]]:
    """Run a backtest with manually specified parameters (no Config required).

    Designed for interactive use in dashboards where parameters can be tuned
    without modifying the config file.

    Args:
        test_df: Market/test data with OHLCV columns and atr_14.
        preds_df: Predictions with timestamp and pred_label (optionally
            pred_proba_* columns).
        leverage: CFD leverage ratio (default 100).
        lots_per_trade: Fixed lot size per trade when auto_lot_sizing=False.
        confidence_threshold: Minimum prediction probability to trade (0 = disabled).
        spread_ticks: Spread in ticks (default 35 = $0.35 for XAUUSD).
        slippage_ticks: Slippage in ticks (default 5 = $0.05).
        commission_per_lot: Commission per lot in dollars (default $10).
        atr_stop_multiplier: ATR multiplier for stop-loss distance (default 0.75).
        auto_lot_sizing: If True, calculate lot size based on risk parameters.
        risk_per_trade_pct: Risk per trade as percentage of equity (default 1%).
        min_lot_size: Minimum lot size when auto_lot_sizing=True.
        max_lot_size: Maximum lot size when auto_lot_sizing=True.
        enable_performance_adjustment: Adjust position size based on equity performance.
        enable_volatility_adjustment: Reduce size during high volatility periods.
        max_capital_risk_pct: Maximum % of initial capital to risk per trade.
        performance_multiplier: Max position size increase when performing well.
        performance_reduction: Min position size when underperforming.
        horizon_bars: Time-based exit after N bars (default 10).
        contract_size: Units per lot (default 100 oz for XAUUSD).
        tick_size: Price tick size in dollars (default 0.01).
        initial_capital: Starting capital for the backtest.

    Returns:
        Tuple of (metrics dict, trades list). Metrics contains normalized
        performance metrics; trades is a list of per-trade records.
    """
    pdf = _prepare_df(test_df, preds_df)

    median_price = float(pdf["Close"].median())
    spread_total = (spread_ticks + slippage_ticks) * tick_size / median_price

    def commission_fn(order_size: float, price: float) -> float:
        lots = abs(order_size) / contract_size
        return lots * commission_per_lot

    margin = 1.0 / leverage

    bt = FractionalBacktest(
        pdf,
        HybridGRUStrategy,
        cash=initial_capital,
        spread=spread_total,
        commission=commission_fn,
        margin=margin,
        exclusive_orders=True,
        finalize_trades=True,
        fractional_unit=1.0,
    )

    stats = bt.run(
        atr_stop_mult=atr_stop_multiplier,
        lots_per_trade=lots_per_trade,
        confidence_threshold=confidence_threshold,
        contract_size=contract_size,
        horizon_bars=horizon_bars,
        auto_lot_sizing=auto_lot_sizing,
        risk_per_trade_pct=risk_per_trade_pct,
        min_lot_size=min_lot_size,
        max_lot_size=max_lot_size,
        enable_performance_adjustment=enable_performance_adjustment,
        enable_volatility_adjustment=enable_volatility_adjustment,
        max_capital_risk_pct=max_capital_risk_pct,
        performance_multiplier=performance_multiplier,
        performance_reduction=performance_reduction,
    )

    metrics = _normalize_stats(stats)
    trades = _trades_to_list(
        stats["_trades"],
        commission_per_lot=commission_per_lot,
        contract_size=contract_size,
    )

    return metrics, trades
