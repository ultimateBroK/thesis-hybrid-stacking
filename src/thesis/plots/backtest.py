"""Static matplotlib/seaborn backtest charts."""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from thesis.config import Config

from .data import _COLORS, _output_dir

logger = logging.getLogger("thesis.visualize")


def _compute_monthly_returns(trades: list[dict]) -> dict[tuple[int, int], float]:
    """Compute monthly percentage returns from trade list."""
    monthly_pnl: dict[tuple[int, int], float] = {}

    for t in trades:
        try:
            exit_time = datetime.fromisoformat(
                str(t["exit_time"]).replace("Z", "+00:00")
            )
            key = (exit_time.year, exit_time.month)
            monthly_pnl[key] = monthly_pnl.get(key, 0.0) + t["pnl"]
        except (ValueError, TypeError):
            continue

    # Convert to percentage returns (assume starting equity = 100k)
    # Use cumulative equity for more accurate monthly returns
    equity = 100_000.0
    equity_by_month: dict[tuple[int, int], tuple[float, float]] = {}

    for t in trades:
        try:
            exit_time = datetime.fromisoformat(
                str(t["exit_time"]).replace("Z", "+00:00")
            )
            key = (exit_time.year, exit_time.month)
            start_eq = equity
            equity += t["pnl"]
            end_eq = equity
            if key not in equity_by_month:
                equity_by_month[key] = (start_eq, end_eq)
            else:
                old_start, old_end = equity_by_month[key]
                equity_by_month[key] = (old_start, end_eq)
        except (ValueError, TypeError):
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
    ax1.plot(equity, color=_COLORS["primary"], linewidth=1)
    ax1.fill_between(
        range(len(equity)), peak, equity_arr, alpha=0.2, color=_COLORS["danger"]
    )
    ax1.set_title(
        f"Equity Curve — {metrics.get('total_trades', 0)} trades, "
        f"{metrics.get('total_return_pct', 0):.1f}% return"
    )
    ax1.set_ylabel("Equity (USD)")
    ax1.grid(True, alpha=0.3)

    # Drawdown
    ax2.fill_between(
        range(len(drawdown_pct)), drawdown_pct, 0, color=_COLORS["danger"], alpha=0.5
    )
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Trade #")
    ax2.grid(True, alpha=0.3)

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
        f"Avg Loss: ${metrics.get('avg_loss', 0):.0f}"
    )
    ax.set_xlabel("PnL (USD)")
    ax.set_ylabel("Count")
    ax.legend()
    fig.savefig(out / "pnl_histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart: pnl_histogram.png")

    # --- 3. Monthly Returns Heatmap ---
    try:
        monthly = _compute_monthly_returns(trades)
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
        rolling_sharpe = rolling_mean / (rolling_std + 1e-10) * np.sqrt(252)

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(rolling_sharpe, color=_COLORS["secondary"], linewidth=0.8)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.axhline(
            y=2,
            color=_COLORS["success"],
            linewidth=1,
            linestyle="--",
            alpha=0.5,
            label="Sharpe = 2",
        )
        ax.set_title(f"Rolling Sharpe Ratio (window={window} trades)")
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Annualized Sharpe")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(out / "rolling_sharpe.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Chart: rolling_sharpe.png")

    # --- 5. Trade Duration vs PnL Scatter ---
    durations = []
    pnls_for_scatter = []
    for t in trades:
        try:
            entry = datetime.fromisoformat(str(t["entry_time"]).replace("Z", "+00:00"))
            exit_ = datetime.fromisoformat(str(t["exit_time"]).replace("Z", "+00:00"))
            dur_hours = (exit_ - entry).total_seconds() / 3600
            durations.append(dur_hours)
            pnls_for_scatter.append(t["pnl"])
        except (ValueError, TypeError):
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
