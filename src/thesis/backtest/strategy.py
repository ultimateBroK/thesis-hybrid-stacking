"""Backtest runner using backtesting.py — thin wrapper.

Replaces the 721-line custom CFD simulator with backtesting.py v0.6.5.
Strategy logic: ML signals + ATR stop-loss. All risk management is native.
"""

import csv
import json
import logging
from pathlib import Path

import pandas as pd
import polars as pl
from backtesting import Backtest, Strategy

from thesis.config import Config

logger = logging.getLogger("thesis.backtest")


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class HybridGRUStrategy(Strategy):
    """Trade on ML signals with ATR stop-loss.

    No manual signal shift — backtesting.py natively delays execution
    by 1 bar (evaluates at close[i], executes at open[i+1]).

    Position sizing: fixed lot size per trade (avoids runaway sizing
    that occurs when buy()/sell() are called without explicit size
    and backtesting.py defaults to "max affordable" with leverage).

    Confidence filtering: when confidence_threshold > 0, only trade
    when the predicted class probability exceeds the threshold.
    """

    atr_stop_mult = 0.75
    lots_per_trade = 1.0
    confidence_threshold = 0.0  # 0 = disabled, trade all signals

    def init(self) -> None:
        self.signals = self.I(lambda: self.data.pred_label, name="signals", plot=False)
        self.atr = self.I(lambda: self.data.atr_14, name="ATR", plot=True)
        # Probabilities — may not exist if predictions lack proba columns
        self._has_proba = hasattr(self.data, "pred_proba_class_minus1")
        if self._has_proba:
            self.proba_short = self.I(
                lambda: self.data.pred_proba_class_minus1,
                name="proba_short",
                plot=False,
            )
            self.proba_hold = self.I(
                lambda: self.data.pred_proba_class_0, name="proba_hold", plot=False
            )
            self.proba_long = self.I(
                lambda: self.data.pred_proba_class_1, name="proba_long", plot=False
            )

    def next(self) -> None:
        signal = int(self.signals[-1])
        price = self.data.Close[-1]
        atr = self.atr[-1]

        # Confidence gate: skip low-confidence signals
        if self.confidence_threshold > 0 and self._has_proba:
            if signal == 1:
                confidence = float(self.proba_long[-1])
            elif signal == -1:
                confidence = float(self.proba_short[-1])
            else:
                return  # Hold — do nothing
            if confidence < self.confidence_threshold:
                return  # Below threshold — skip trade

        # Fixed lot sizing: lots × contract_size = units (e.g. 1 lot = 100 oz)
        size = self.lots_per_trade * 100  # contract_size baked in

        # is_long/is_short allow signal reversal via exclusive_orders
        if signal == 1 and not self.position.is_long:
            sl = price - atr * self.atr_stop_mult
            self.buy(size=size, sl=sl)
        elif signal == -1 and not self.position.is_short:
            sl = price + atr * self.atr_stop_mult
            self.sell(size=size, sl=sl)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_stats(stats: pd.Series) -> dict:
    """Convert backtesting.py stats to snake_case dict for downstream use."""
    raw = stats.to_dict()
    out: dict = {}
    for k, v in raw.items():
        if k.startswith("_"):
            continue
        # Convert display names to snake_case
        key = (
            k.lower()
            .replace(" ", "_")
            .replace(".", "")
            .replace("[", "")
            .replace("]", "")
            .replace("(", "")
            .replace(")", "")
            .replace("$", "")
            .replace("%", "pct")
            .replace("#", "num")
            .replace("__", "_")
            .rstrip("_")
        )
        out[key] = v
    return out


def _trades_to_list(
    trades_df: pd.DataFrame,
    commission_per_lot: float = 20.0,
    contract_size: float = 100.0,
) -> list[dict]:
    """Convert _trades DataFrame to list of dicts for JSON serialization.

    backtesting.py stores ReturnPct as a decimal fraction (e.g. 0.002 = 0.2%).
    We multiply by 100 so the CSV shows an actual percentage.

    Args:
        trades_df: Raw trades DataFrame from backtesting.py stats.
        commission_per_lot: Commission charged per lot (for per-trade breakdown).
        contract_size: Units per lot (e.g. 100 oz for XAU/USD).
    """
    if trades_df.empty:
        return []
    records = trades_df.reset_index(drop=True)
    result: list[dict] = []
    for _, row in records.iterrows():
        size = float(row.get("Size", 0))
        lots = abs(size) / contract_size
        commission = lots * commission_per_lot
        result.append(
            {
                "entry_time": str(row.get("EntryTime", "")),
                "exit_time": str(row.get("ExitTime", "")),
                "direction": "long" if size > 0 else "short",
                "entry_price": float(row.get("EntryPrice", 0)),
                "exit_price": float(row.get("ExitPrice", 0)),
                "size": size,
                "pnl": float(row.get("PnL", 0)),
                "return_pct": float(row.get("ReturnPct", 0)) * 100,
                "commission": round(commission, 2),
                "duration": str(row.get("Duration", "")),
            }
        )
    return result


def _prepare_df(test_df: pl.DataFrame, preds_df: pl.DataFrame) -> pd.DataFrame:
    """Merge test data + predictions and convert to pandas for backtesting.py."""
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

    # Convert to pandas with backtesting.py required columns
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
    # Ensure required columns
    if "Volume" not in pdf.columns:
        pdf["Volume"] = 0

    pdf = pdf.set_index("timestamp")
    pdf.index = pd.DatetimeIndex(pdf.index)

    return pdf


def _run_bt(pdf: pd.DataFrame, config: Config) -> tuple[pd.Series, Backtest]:
    """Create and run the Backtest with native cost params."""
    bc = config.backtest
    dc = config.data

    # Spread: absolute ticks → relative rate
    median_price = float(pdf["Close"].median())
    spread_total = (bc.spread_ticks + bc.slippage_ticks) * dc.tick_size / median_price

    # Commission: callable for per-lot model
    def commission_fn(order_size: float, price: float) -> float:
        lots = abs(order_size) / dc.contract_size
        return lots * bc.commission_per_lot

    # Margin: 1/leverage
    margin = 1.0 / bc.leverage

    bt = Backtest(
        pdf,
        HybridGRUStrategy,
        cash=bc.initial_capital,
        spread=spread_total,
        commission=commission_fn,
        margin=margin,
        exclusive_orders=True,
        finalize_trades=True,
    )

    stats = bt.run(
        atr_stop_mult=bc.atr_stop_multiplier,
        lots_per_trade=bc.lots_per_trade,
        confidence_threshold=bc.confidence_threshold,
    )
    return stats, bt


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_backtest(config: Config) -> None:
    """Run CFD backtest and save results.

    Args:
        config: Loaded application configuration.
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

    # Extract results
    metrics = _normalize_stats(stats)
    trades = _trades_to_list(
        stats["_trades"],
        commission_per_lot=config.backtest.commission_per_lot,
        contract_size=config.data.contract_size,
    )

    # Save JSON
    out_path = Path(config.paths.backtest_results)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"metrics": metrics, "trades": trades}, f, indent=2, default=str)
    logger.info("Backtest results saved: %s", out_path)

    # Save trade details CSV
    if trades:
        csv_path = out_path.parent / "trades_detail.csv"
        fieldnames = list(trades[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(trades)
        logger.info("Trade details CSV saved: %s (%d trades)", csv_path, len(trades))

        # Save equity curve CSV
        eq_path = out_path.parent / "equity_curve.csv"
        initial_capital = (
            config.backtest.initial_capital
            if hasattr(config.backtest, "initial_capital")
            else 10_000.0
        )
        equity = initial_capital
        with open(eq_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["trade_num", "exit_time", "pnl", "equity", "drawdown_pct"])
            peak = initial_capital
            for i, t in enumerate(trades, 1):
                equity += t["pnl"]
                peak = max(peak, equity)
                dd_pct = (equity - peak) / peak * 100 if peak > 0 else 0.0
                writer.writerow(
                    [
                        i,
                        t.get("exit_time", ""),
                        round(t["pnl"], 2),
                        round(equity, 2),
                        round(dd_pct, 4),
                    ]
                )
        logger.info("Equity curve CSV saved: %s", eq_path)

    # Log key metrics
    logger.info("=== BACKTEST RESULTS ===")
    for k, v in metrics.items():
        logger.info("  %s: %s", k, v)

    # Save Bokeh HTML chart
    session_dir = Path(config.paths.session_dir) if config.paths.session_dir else None
    if session_dir:
        chart_dir = session_dir / "backtest"
        chart_dir.mkdir(parents=True, exist_ok=True)
        chart_path = chart_dir / "backtest_chart.html"
        bt.plot(
            filename=str(chart_path),
            open_browser=False,
            plot_equity=True,
            plot_drawdown=True,
            plot_trades=True,
            resample="2h",
        )
        logger.info("Bokeh chart saved: %s", chart_path)


def run_backtest_from_data(
    test_df: pl.DataFrame,
    preds_df: pl.DataFrame,
    config: Config,
) -> dict:
    """Run backtest from pre-loaded DataFrames (for ablation).

    Args:
        test_df: Test data (Polars).
        preds_df: Predictions (Polars).
        config: Configuration.

    Returns:
        Normalized metrics dict.
    """
    pdf = _prepare_df(test_df, preds_df)
    stats, _ = _run_bt(pdf, config)
    return _normalize_stats(stats)
