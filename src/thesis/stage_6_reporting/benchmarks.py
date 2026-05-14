"""Benchmark comparison: naive strategies vs model."""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path

import numpy as np
import polars as pl
from polars.exceptions import ComputeError

from thesis.shared.config import Config
from thesis.shared.constants import H1_BARS_PER_YEAR

logger = logging.getLogger("thesis.report")

BARS_PER_YEAR = H1_BARS_PER_YEAR


def model_label(config: Config) -> str:
    architecture = config.model.architecture
    if architecture in ("static", "lgbm"):
        return "LightGBM"
    if architecture == "stacking":
        return "Hybrid Stacking"
    return f"{architecture.title()} Model"


def annualized_sharpe(returns: np.ndarray, bars_per_year: int = BARS_PER_YEAR) -> float:
    std = float(np.std(returns, ddof=1))
    if std == 0 or np.isnan(std):
        return 0.0
    return float(np.mean(returns) / std * np.sqrt(bars_per_year))


def max_drawdown_pct(equity: np.ndarray) -> float:
    if len(equity) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak * 100
    return float(abs(dd.min()))


def equity_curve_from_bar_returns(
    returns: np.ndarray, initial_capital: float
) -> np.ndarray:
    equity = np.empty(len(returns) + 1)
    equity[0] = initial_capital
    for i, r in enumerate(returns):
        equity[i + 1] = equity[i] * (1.0 + r)
    return equity


def compute_random_strategy(
    returns: np.ndarray, initial_capital: float, leverage: int, seed: int
) -> dict:
    rng = np.default_rng(seed)
    signals = rng.choice([-1, 1], size=len(returns))
    leveraged = returns * signals * leverage

    equity = equity_curve_from_bar_returns(leveraged, initial_capital)
    ret = (equity[-1] / initial_capital - 1) * 100
    sharpe = annualized_sharpe(leveraged)
    max_dd = max_drawdown_pct(equity)

    active = leveraged[signals != 0]
    win_rate = float((active > 0).sum() / len(active) * 100) if len(active) > 0 else 0.0

    return {
        "return_pct": ret,
        "sharpe": sharpe,
        "max_dd_pct": max_dd,
        "win_rate_pct": win_rate,
        "num_trades": int(np.abs(np.diff(signals)).sum() / 2 + 1),
    }


def load_close_prices_for_benchmark(
    test_data_path: Path, hybrid_metrics: dict, config: Config
) -> np.ndarray | None:
    is_static = config.validation.method == "static"
    if test_data_path.exists() and is_static:
        try:
            df = pl.read_parquet(test_data_path, columns=["close"])
            return df["close"].to_numpy()
        except (ComputeError, OSError):
            logger.warning(
                "Failed to load static test data: %s", test_data_path, exc_info=True
            )
    elif test_data_path.exists() and not is_static:
        logger.warning(
            "Static test file found but workflow is walk-forward — ignoring stale test_data"
        )

    ohlcv_path = Path(config.paths.ohlcv)
    if not ohlcv_path.exists():
        logger.warning("No OHLCV available for benchmark fallback: %s", ohlcv_path)
        return None

    try:
        df = pl.read_parquet(ohlcv_path)
    except (ComputeError, OSError):
        logger.warning(
            "Failed to load OHLCV for benchmarks: %s", ohlcv_path, exc_info=True
        )
        return None

    ts_expr = pl.col("timestamp")
    ts_dtype = df.schema.get("timestamp")
    if ts_dtype == pl.Utf8:
        ts_expr = ts_expr.str.to_datetime()
        ts_dtype = df.select(ts_expr.alias("timestamp")).schema["timestamp"]
    if getattr(ts_dtype, "time_zone", None):
        ts_expr = ts_expr.dt.replace_time_zone(None)

    bt_start = hybrid_metrics.get("start")
    bt_end = hybrid_metrics.get("end")
    if bt_start and bt_end:
        start_dt = datetime.fromisoformat(str(bt_start)[:19])
        end_dt = datetime.fromisoformat(str(bt_end)[:19])
        df = df.filter((ts_expr >= start_dt) & (ts_expr <= end_dt))

    if len(df) < 2:
        logger.warning("OHLCV fallback: insufficient bars (%d)", len(df))
        return None

    logger.info("Benchmark using OHLCV fallback: %d bars", len(df))
    return df["close"].to_numpy()


def compute_benchmark_comparison(
    test_data_path: Path, hybrid_metrics: dict, config: Config
) -> list[dict]:
    close = load_close_prices_for_benchmark(test_data_path, hybrid_metrics, config)
    if close is None or len(close) < 2:
        return []

    initial = config.backtest.initial_capital
    leverage = config.backtest.leverage
    seed = config.workflow.random_seed
    bar_returns = np.diff(close) / close[:-1]

    bh_equity = equity_curve_from_bar_returns(bar_returns, initial)
    bh_return = (bh_equity[-1] / initial - 1) * 100

    al_returns = bar_returns * leverage
    al_equity = equity_curve_from_bar_returns(al_returns, initial)
    al_return = (al_equity[-1] / initial - 1) * 100

    random_result = compute_random_strategy(bar_returns, initial, leverage, seed)

    return [
        {
            "strategy": "Buy & Hold",
            "return_pct": bh_return,
            "sharpe": annualized_sharpe(bar_returns),
            "max_dd_pct": max_drawdown_pct(bh_equity),
            "win_rate_pct": float((bar_returns > 0).sum() / len(bar_returns) * 100)
            if len(bar_returns) > 0
            else 0.0,
            "num_trades": 1,
        },
        {
            "strategy": "Always Long",
            "return_pct": al_return,
            "sharpe": annualized_sharpe(al_returns),
            "max_dd_pct": max_drawdown_pct(al_equity),
            "win_rate_pct": float((al_returns > 0).sum() / len(al_returns) * 100)
            if len(al_returns) > 0
            else 0.0,
            "num_trades": 1,
        },
        {"strategy": "Random Signal", **random_result},
        {
            "strategy": model_label(config),
            "return_pct": hybrid_metrics.get("return_pct", 0),
            "sharpe": hybrid_metrics.get("sharpe_ratio", 0),
            "max_dd_pct": abs(hybrid_metrics.get("max_drawdown_pct", 0)),
            "win_rate_pct": hybrid_metrics.get("win_rate_pct", 0),
            "num_trades": int(hybrid_metrics.get("num_trades", 0)),
        },
    ]
