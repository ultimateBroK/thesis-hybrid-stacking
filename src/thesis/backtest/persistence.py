"""Backtest result persistence — JSON, CSV, and Bokeh chart outputs.

Handles saving backtest metrics, trade details, equity curves, and
interactive Bokeh charts to the session directory.
"""

import csv
import json
import logging
from pathlib import Path

import pandas as pd
from backtesting.lib import FractionalBacktest

logger = logging.getLogger("thesis.backtest")


def _trades_to_list(
    trades_df: pd.DataFrame,
    commission_per_lot: float = 20.0,
    contract_size: float = 100.0,
) -> list[dict]:
    """Convert a backtesting.py trades DataFrame to a JSON-serializable list.

    Each record contains entry/exit timestamps, direction ("long" or "short"),
    entry/exit prices, lot size, PnL, return percentage, commission, and
    duration.

    Args:
        trades_df: Raw trades DataFrame from backtesting.py stats.
        commission_per_lot: Commission charged per lot to compute per-trade
            commission.
        contract_size: Units per lot used to convert raw Size into lot counts.

    Returns:
        List of trade dictionaries with keys: entry_time, exit_time,
        direction, entry_price, exit_price, lot_size, pnl, return_pct,
        commission, duration.
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
                "lot_size": lots,
                "pnl": float(row.get("PnL", 0)),
                "return_pct": float(row.get("ReturnPct", 0)) * 100,
                "commission": round(commission, 2),
                "duration": str(row.get("Duration", "")),
            }
        )
    return result


def _save_json_results(
    metrics: dict,
    trades: list[dict],
    out_path: Path,
) -> None:
    """Save backtest results as JSON.

    Args:
        metrics: Normalized metrics from _normalize_stats.
        trades: List of trade records from _trades_to_list.
        out_path: Destination path for JSON file.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"metrics": metrics, "trades": trades}, f, indent=2, default=str)
    logger.info("Backtest results saved: %s", out_path)


def _save_trade_details_csv(trades: list[dict], out_dir: Path) -> None:
    """Save per-trade records as CSV.

    Args:
        trades: List of trade dictionaries.
        out_dir: Parent directory for output CSV.
    """
    if not trades:
        return
    csv_path = out_dir / "trades_detail.csv"
    fieldnames = list(trades[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trades)
    logger.info("Trade details CSV saved: %s (%d trades)", csv_path, len(trades))


def _save_equity_curve_csv(
    trades: list[dict],
    out_dir: Path,
    initial_capital: float = 10_000.0,
) -> None:
    """Save equity curve as CSV with running peak and drawdown.

    Each row represents a closed trade with the running equity, peak equity,
    and drawdown percentage.

    Args:
        trades: List of trade dictionaries with pnl and exit_time.
        out_dir: Parent directory for output CSV.
        initial_capital: Starting capital for equity calculation.
    """
    if not trades:
        return
    eq_path = out_dir / "equity_curve.csv"
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


def _save_bokeh_chart(
    bt: FractionalBacktest,
    stats: pd.Series,
    session_dir: Path | None,
) -> None:
    """Save Bokeh HTML chart for the backtest.

    Args:
        bt: Backtest instance with .plot() method.
        stats: Backtest statistics Series (checked for trade count).
        session_dir: Session directory for chart output; if None, skips chart.
    """
    if not session_dir:
        return
    if len(stats["_trades"]) == 0:
        logger.info("No trades — skipping Bokeh chart")
        return
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
