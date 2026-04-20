"""Backtest statistics normalization and metric computation.

Converts backtesting.py's raw statistics Series into snake_case dictionaries
with additional computed metrics (recovery factor, avg win/loss).
"""

import pandas as pd


def _extract_recovery_factor(
    equity_final: float,
    equity_peak: float,
    max_dd_pct: float,
    initial_capital: float = 10_000.0,
) -> float:
    """Compute recovery factor = net_profit / max_drawdown_dollars.

    Args:
        equity_final: Final equity value from backtest.
        equity_peak: Peak equity reached during backtest.
        max_dd_pct: Maximum drawdown as percentage of peak equity.
        initial_capital: Starting capital (default 10_000).

    Returns:
        Recovery factor (0.0 if max_drawdown is zero or negative).
    """
    net_profit = equity_final - initial_capital
    max_dd_dollars = abs(max_dd_pct / 100) * equity_peak
    if max_dd_dollars > 0:
        return net_profit / max_dd_dollars
    return 0.0


def _compute_avg_win_loss(trades_df: pd.DataFrame) -> tuple[float, float]:
    """Compute average win and average loss from trades DataFrame.

    Args:
        trades_df: DataFrame with PnL column from backtesting.py stats.

    Returns:
        Tuple of (avg_win, avg_loss). Returns (0.0, 0.0) if DataFrame is empty
        or PnL column is missing.
    """
    if trades_df.empty or "PnL" not in trades_df.columns:
        return 0.0, 0.0
    wins = trades_df[trades_df["PnL"] > 0]["PnL"]
    losses = trades_df[trades_df["PnL"] <= 0]["PnL"]
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(losses.mean()) if not losses.empty else 0.0
    return avg_win, avg_loss


def _normalize_stats(stats: pd.Series) -> dict:
    """Convert a Backtesting.py statistics Series into a snake_case dict.

    Omits keys that begin with an underscore and normalizes display-style keys
    by lowercasing, replacing spaces and punctuation with underscores, and
    mapping ``%`` to ``pct`` and ``#`` to ``num``.

    Adds computed fields:
        - ``recovery_factor``: net_profit / max_drawdown_dollars.
        - ``avg_win``: mean PnL of winning trades.
        - ``avg_loss``: mean PnL of losing trades.

    Args:
        stats: Series-like statistics object produced by Backtesting.py.

    Returns:
        Dictionary of normalized metric names to their original values.
    """
    raw = stats.to_dict()
    out: dict = {}
    for k, v in raw.items():
        if k.startswith("_"):
            continue
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

    equity_final = out.get("equity_final", 0)
    equity_peak = out.get("equity_peak", equity_final)
    max_dd_pct = out.get("max_drawdown_pct", 0)
    out["recovery_factor"] = _extract_recovery_factor(
        equity_final, equity_peak, max_dd_pct
    )

    trades_df = stats.get("_trades", pd.DataFrame())
    avg_win, avg_loss = _compute_avg_win_loss(trades_df)
    out["avg_win"] = avg_win
    out["avg_loss"] = avg_loss

    return out
