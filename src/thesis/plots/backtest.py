"""Static matplotlib/seaborn backtest charts."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from thesis.config import Config

from .data import _COLORS, _output_dir

logger = logging.getLogger("thesis.visualize")


def _compute_monthly_returns(
    trades: list[dict], initial_capital: float
) -> dict[tuple[int, int], float]:
    """
    Compute monthly percentage returns from trade list.

    Args:
        trades: List of trade dictionaries containing 'exit_time' and 'pnl' keys.
        initial_capital: Starting capital used to compute cumulative equity for accurate monthly returns.

    Returns:
        Dictionary mapping (year, month) tuples to percentage returns.
    """
    monthly_pnl: dict[tuple[int, int], float] = {}

    for t in trades:
        try:
            # Use pd.to_datetime to avoid ISO format parsing errors
            exit_time = pd.to_datetime(t["exit_time"])
            key = (exit_time.year, exit_time.month)
            monthly_pnl[key] = monthly_pnl.get(key, 0.0) + t["pnl"]
        except Exception:
            continue

    # Use cumulative equity for more accurate monthly returns
    equity = float(initial_capital)
    equity_by_month: dict[tuple[int, int], tuple[float, float]] = {}

    for t in trades:
        try:
            exit_time = pd.to_datetime(t["exit_time"])
            key = (exit_time.year, exit_time.month)
            start_eq = equity
            equity += t["pnl"]
            end_eq = equity
            if key not in equity_by_month:
                equity_by_month[key] = (start_eq, end_eq)
            else:
                old_start, old_end = equity_by_month[key]
                equity_by_month[key] = (old_start, end_eq)
        except Exception:
            continue

    monthly_returns = {}
    for key, (start, end) in equity_by_month.items():
        if start > 0:
            monthly_returns[key] = (end - start) / start * 100

    return monthly_returns


def _generate_backtest_charts(config: Config) -> None:
    """Generate backtest analysis charts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = _output_dir(config, "backtest")

    bt_path = Path(config.paths.backtest_results)
    if not bt_path.exists():
        logger.warning("Backtest results not found: %s", bt_path)
        return

    with open(bt_path) as f:
        bt = json.load(f)

    trades = bt.get("trades", [])
    metrics = bt.get("metrics", {})

    if not trades:
        logger.warning("No trades found in backtest results")
        return

    # --- Extract times for X-axis plotting ---
    try:
        times = [pd.to_datetime(trades[0]["entry_time"])]
        for t in trades:
            times.append(pd.to_datetime(t["exit_time"]))
    except Exception as e:
        logger.debug(f"Time parsing failed, falling back to indices: {e}")
        times = list(range(len(trades) + 1))

    # --- 1. Equity Curve with Drawdown ---
    pnls = [t["pnl"] for t in trades]
    equity = [config.backtest.initial_capital]
    for p in pnls:
        equity.append(equity[-1] + p)

    equity_arr = np.array(equity)
    peak = np.maximum.accumulate(equity_arr)
    drawdown_pct = (equity_arr - peak) / peak * 100

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), height_ratios=[3, 1], sharex=True
    )

    # Equity curve
    ax1.plot(times, equity, color=_COLORS["primary"], linewidth=1)
    ax1.fill_between(times, peak, equity_arr, alpha=0.2, color=_COLORS["danger"])
    # Align metric keys with normalize_stats output
    ax1.set_title(
        f"Equity Curve — {metrics.get('num_trades', 0)} trades, "
        f"{metrics.get('return_pct', 0):.1f}% return"
    )
    ax1.set_ylabel("Equity (USD)")
    ax1.grid(True, alpha=0.3)

    # Drawdown
    ax2.fill_between(times, drawdown_pct, 0, color=_COLORS["danger"], alpha=0.5)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    fig.autofmt_xdate()

    fig.savefig(out / "equity_drawdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart: equity_drawdown.png")

    # --- 2. Trade PnL Histogram ---
    fig, ax = plt.subplots(figsize=(10, 5))
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    ax.hist(
        wins, bins=50, alpha=0.7, color=_COLORS["success"], label=f"Wins ({len(wins)})"
    )
    ax.hist(
        losses,
        bins=50,
        alpha=0.7,
        color=_COLORS["danger"],
        label=f"Losses ({len(losses)})",
    )
    ax.axvline(x=0, color="black", linewidth=1, linestyle="-")
    ax.set_title(
        f"Trade PnL Distribution — Avg Win: ${metrics.get('avg_win', 0):.0f}, "
        f"Avg Loss: ${abs(metrics.get('avg_loss', 0)):.0f}"
    )
    ax.set_xlabel("PnL (USD)")
    ax.set_ylabel("Count")
    ax.legend()
    fig.savefig(out / "pnl_histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart: pnl_histogram.png")

    # --- 3. Monthly Returns Heatmap ---
    try:
        monthly = _compute_monthly_returns(trades, config.backtest.initial_capital)
        if monthly:
            years = sorted(set(k[0] for k in monthly.keys()))
            month_names = [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ]

            data = np.full((len(years), 12), np.nan)
            for (yr, mo), ret in monthly.items():
                yi = years.index(yr)
                data[yi, mo - 1] = ret

            fig, ax = plt.subplots(figsize=(12, max(3, len(years) * 0.8)))
            im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-5, vmax=10)

            ax.set_xticks(range(12))
            ax.set_xticklabels(month_names)
            ax.set_yticks(range(len(years)))
            ax.set_yticklabels([str(y) for y in years])

            # Annotate cells
            for i in range(len(years)):
                for j in range(12):
                    if not np.isnan(data[i, j]):
                        ax.text(
                            j,
                            i,
                            f"{data[i, j]:.1f}%",
                            ha="center",
                            va="center",
                            fontsize=8,
                            fontweight="bold",
                        )

            fig.colorbar(im, ax=ax, label="Return %")
            ax.set_title("Monthly Returns Heatmap")
            fig.savefig(out / "monthly_returns.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("Chart: monthly_returns.png")
    except Exception as e:
        logger.warning("Monthly returns chart failed: %s", e)

    # --- 4. Rolling Sharpe Ratio ---
    if len(pnls) > 30:
        pnls_arr = np.array(pnls)
        window = 30
        rolling_mean = np.convolve(pnls_arr, np.ones(window) / window, mode="valid")
        rolling_std = np.array(
            [pnls_arr[i : i + window].std() for i in range(len(pnls_arr) - window + 1)]
        )

        # Calculate average annual trade count for proper annualization
        try:
            days_total = (times[-1] - times[0]).days
            trades_per_year = len(pnls) / max(days_total / 365.25, 1)
        except Exception:
            trades_per_year = 100  # Fallback default value

        annualization_factor = np.sqrt(trades_per_year)
        rolling_sharpe = rolling_mean / (rolling_std + 1e-10) * annualization_factor

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(rolling_sharpe, color=_COLORS["secondary"], linewidth=0.8)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_title(
            f"Rolling Sharpe Ratio (window={window} trades, ann_factor=√{trades_per_year:.0f})"
        )
        ax.set_xlabel("Trade Window Index")
        ax.set_ylabel("Annualized Sharpe")
        ax.grid(True, alpha=0.3)
        fig.savefig(out / "rolling_sharpe.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Chart: rolling_sharpe.png")

    # --- 5. Trade Duration vs PnL Scatter ---
    durations = []
    pnls_for_scatter = []
    for t in trades:
        try:
            entry = pd.to_datetime(t["entry_time"])
            exit_ = pd.to_datetime(t["exit_time"])
            dur_hours = (exit_ - entry).total_seconds() / 3600
            durations.append(dur_hours)
            pnls_for_scatter.append(t["pnl"])
        except Exception:
            continue

    if durations:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = [
            _COLORS["success"] if p > 0 else _COLORS["danger"] for p in pnls_for_scatter
        ]
        ax.scatter(durations, pnls_for_scatter, c=colors, alpha=0.5, s=20)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_title("Trade Duration vs PnL")
        ax.set_xlabel("Duration (hours)")
        ax.set_ylabel("PnL (USD)")
        ax.grid(True, alpha=0.3)
        fig.savefig(out / "duration_vs_pnl.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Chart: duration_vs_pnl.png")
