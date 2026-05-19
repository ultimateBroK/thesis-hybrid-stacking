"""Demo backtest: compressed single-file version of stage_5_backtest.

Illustration-only. Runs CFD backtest via FractionalBacktest + MLSignalStrategy.
Exports: ``run_backtest_demo``, ``compute_backtest``, ``BacktestResult``.
"""

from __future__ import annotations

from collections.abc import Callable
import json
import logging
from pathlib import Path

from backtesting import Strategy
from backtesting.lib import FractionalBacktest
import pandas as pd
import polars as pl

from thesis.shared.config import Config

logger = logging.getLogger("thesis.demo.backtest")

BacktestResult = tuple[dict, list[dict], object]

_MIN_ATR_FLOOR = 0.0001
_DEFAULT_CONTRACT_SIZE = 100.0
_MIN_BARS_FOR_SIGNAL = 2
_MIN_ORDER_SIZE = 1


class MLSignalStrategy(Strategy):
    """Trade ML signals with ATR stops/TPs, confidence gate."""

    atr_stop_mult = 1.0
    atr_tp_mult = 2.0
    lots_per_trade = 0.2
    min_lots = 0.1
    max_lots = 0.5
    confidence_threshold = 0.0
    min_atr = _MIN_ATR_FLOOR
    contract_size = 100
    horizon_bars = 0
    max_drawdown_cutoff = 0.50
    dd_cooldown_bars = 12
    max_open_positions = 1
    daily_loss_limit = 0.03
    min_bars_between_trades = 0

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _calendar_day(value: object) -> object:
        """Map bar timestamp to calendar date (NY+7h for XAU/USD session)."""
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        ts_ny = ts.tz_convert("America/New_York")
        return (ts_ny + pd.Timedelta(hours=7)).date()

    def _normalize_signal(self) -> int:
        """Map raw float signal to {-1, 0, 1}."""
        raw = float(self.signals[-2])
        if raw in (-1, 0, 1):
            return int(raw)
        return 1 if raw > 0 else (-1 if raw < 0 else 0)

    def _init_risk_state(self) -> None:
        """Seed drawdown / daily-loss tracking before first bar."""
        self._initial_capital = self.equity
        self._entry_bar: dict[str, int] = {}
        self._last_exit_bar: int = 0
        self._position_was_open: bool = False
        self._peak_equity: float = self.equity
        self._dd_cooldown_left: int = 0
        self._dd_cutoff_breached: bool = False
        self._daily_start_equity: float = self.equity
        self._current_date: object = None

    def _check_drawdown_breaker(self) -> None:
        """Halt trading when drawdown exceeds cutoff, start cooldown."""
        if self.max_drawdown_cutoff <= 0 or self._peak_equity <= 0:
            return
        dd = (self._peak_equity - self.equity) / self._peak_equity
        if dd >= self.max_drawdown_cutoff and not self._dd_cutoff_breached:
            self._dd_cutoff_breached = True
            self._dd_cooldown_left = self.dd_cooldown_bars
            logger.warning("Drawdown breaker: %.1f%%", dd * 100)

    def _track_daily_equity(self) -> None:
        """Reset daily start equity on calendar-day boundary."""
        bar_date = self._calendar_day(self.data.index[-1])
        if self._current_date != bar_date:
            self._current_date = bar_date
            self._daily_start_equity = self.equity

    def _place_order(self, direction: int, atr: float) -> None:
        """Execute buy (direction=+1) or sell (direction=-1) with ATR SL/TP."""
        lots = max(self.min_lots, min(self.lots_per_trade, self.max_lots))
        size = max(_MIN_ORDER_SIZE, round(lots * self.contract_size))
        close = self.data.Close[-1]
        sl_sign = -1 if direction == 1 else 1
        tp_sign = 1 if direction == 1 else -1
        sl = close + sl_sign * atr * self.atr_stop_mult
        tp = close + tp_sign * atr * self.atr_tp_mult if self.atr_tp_mult > 0 else None
        tag = "long" if direction == 1 else "short"
        self._entry_bar[tag] = len(self.data)
        (self.buy if direction == 1 else self.sell)(size=size, sl=sl, tp=tp)
        self._position_was_open = True

    # ── lifecycle ────────────────────────────────────────────────────

    def init(self) -> None:
        """Register backtesting.py indicators."""
        self.signals = self.I(lambda: self.data.pred_label, name="signals", plot=False)
        self.atr = self.I(lambda: self.data.atr_14, name="ATR", plot=True)
        self._has_proba = hasattr(self.data, "pred_proba_class_minus1")
        if self._has_proba:
            self.proba_short = self.I(
                lambda: self.data.pred_proba_class_minus1,
                name="proba_short",
                plot=False,
            )
            self.proba_hold = self.I(
                lambda: self.data.pred_proba_class_0,
                name="proba_hold",
                plot=False,
            )
            self.proba_long = self.I(
                lambda: self.data.pred_proba_class_1,
                name="proba_long",
                plot=False,
            )
        self._init_risk_state()

    def record_recent_exit(self) -> None:
        """Snapshot bar index so min_bars_between_trades can enforce a gap."""
        if self._position_was_open and not self.position:
            self._last_exit_bar = len(self.data)
            self._position_was_open = False
            self._entry_bar.pop("long", None)
            self._entry_bar.pop("short", None)

    def _update_risk_state(self) -> None:
        self._peak_equity = max(self._peak_equity, self.equity)
        if self._dd_cooldown_left > 0:
            self._dd_cooldown_left -= 1
        self._check_drawdown_breaker()
        self._track_daily_equity()

    def has_enough_history(self) -> bool:
        """Wait until at least 2 bars are buffered for lookback."""
        return len(self.signals) >= _MIN_BARS_FOR_SIGNAL

    def read_trade_signal(self) -> tuple[int, float]:
        """Discretize signal and floor ATR so SL/TP are never zero."""
        signal = self._normalize_signal()
        atr = max(float(self.atr[-1]), self.min_atr)
        return signal, atr

    def close_expired_position(self) -> None:
        """Force-close when the holding period exceeds horizon_bars."""
        if self.horizon_bars <= 0 or not self.position:
            return
        entry_bar = self._entry_bar.get("long") or self._entry_bar.get("short")
        if entry_bar is not None and (len(self.data) - entry_bar) >= self.horizon_bars:
            self.position.close()

    def _is_trading_allowed(self) -> bool:
        if len(self.orders) > 0:
            return False
        if self.position and self.max_open_positions <= 1:
            return False
        if (
            self.min_bars_between_trades > 0
            and self._last_exit_bar > 0
            and (len(self.data) - self._last_exit_bar) <= self.min_bars_between_trades
        ):
            return False
        if self._dd_cutoff_breached:
            return False
        if self._dd_cooldown_left > 0:
            return False
        if self.daily_loss_limit > 0 and self._daily_start_equity > 0:
            daily_pnl = (
                self.equity - self._daily_start_equity
            ) / self._daily_start_equity
            if daily_pnl <= -self.daily_loss_limit:
                return False
        return True

    def can_open_trade(self, signal: int) -> bool:
        """Gate on risk limits *and* model confidence (if available)."""
        if not self._is_trading_allowed():
            return False
        if self.confidence_threshold > 0 and self._has_proba:
            if signal == 1:
                confidence = float(self.proba_long[-2])
            elif signal == -1:
                confidence = float(self.proba_short[-2])
            else:
                return False
            if confidence < self.confidence_threshold:
                return False
        return True

    def open_trade(self, signal: int, atr: float) -> None:
        """Delegate to _place_order once position slot is free."""
        if signal in (1, -1) and not self.position:
            self._place_order(signal, atr)

    def next(self) -> None:
        """Bar-by-bar entry point called by backtesting.py."""
        self.record_recent_exit()
        self._update_risk_state()
        if not self.has_enough_history():
            return
        signal, atr = self.read_trade_signal()
        self.close_expired_position()
        if not self.can_open_trade(signal):
            return
        self.open_trade(signal, atr)


_BT_KEY_MAP: dict[str, str] = {
    "Return [%]": "return_pct",
    "Max. Drawdown [%]": "max_drawdown_pct",
    "Profit Factor": "profit_factor",
    "Sharpe Ratio": "sharpe_ratio",
    "Sortino Ratio": "sortino_ratio",
    "Calmar Ratio": "calmar_ratio",
    "Win Rate [%]": "win_rate_pct",
    "Expectancy [%]": "expectancy_pct",
    "Avg. Trade [%]": "avg_trade_pct",
    "# Trades": "num_trades",
    "Equity Final [$]": "equity_final",
    "Start": "start",
    "End": "end",
}


def _prepare_df(
    test_df: pl.DataFrame,
    preds_df: pl.DataFrame,
    *,
    test_source: str = "<in-memory>",
    preds_source: str = "<in-memory>",
) -> pd.DataFrame:
    """Join OHLCV + predictions on timestamp, validate coverage."""
    test = test_df.with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
    preds = preds_df.with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
    if "pred_label" not in preds.columns:
        raise ValueError("Predictions must contain 'pred_label' column")
    pred_cols = ["timestamp", "pred_label"]
    for col in (
        "pred_proba_class_minus1",
        "pred_proba_class_0",
        "pred_proba_class_1",
    ):
        if col in preds.columns:
            pred_cols.append(col)
    merged = test.join(preds.select(pred_cols), on="timestamp", how="inner")
    coverage = len(merged) / len(preds) if len(preds) else 0.0
    logger.info(
        "Backtest merge: %d + %d -> %d (%.1f%%)",
        len(test),
        len(preds),
        len(merged),
        coverage * 100,
    )
    if coverage < 0.99:
        raise ValueError(
            f"Merge coverage {coverage * 100:.1f}% < 99%. Check timestamps."
        )
    if "atr_14" not in merged.columns:
        raise ValueError("atr_14 missing from test data.")
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


def _normalize_stats(stats: pd.Series) -> dict:
    raw = stats.to_dict()
    return {v: raw[k] for k, v in _BT_KEY_MAP.items() if k in raw}


def _trades_to_list(
    trades_df: pd.DataFrame,
    commission_per_lot: float,
    contract_size: float = _DEFAULT_CONTRACT_SIZE,
) -> list[dict]:
    if trades_df.empty:
        return []
    result: list[dict] = []
    for _, row in trades_df.reset_index(drop=True).iterrows():
        size = float(row.get("Size", 0))
        lots = abs(size) / contract_size
        dur = row.get("Duration")
        dur_sec = pd.Timedelta(dur).total_seconds() if dur is not None else None
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
                "commission": round(lots * commission_per_lot, 2),
                "duration_sec": dur_sec,
            }
        )
    return result


def _compute_spread_rate(
    spread_ticks: float,
    slippage_ticks: float,
    tick_size: float,
    median_price: float,
) -> float:
    return (spread_ticks + slippage_ticks) * tick_size / median_price


def _make_commission_fn(
    commission_per_lot: float, contract_size: float
) -> Callable[[float, float], float]:
    def commission_fn(order_size: float, _price: float) -> float:
        return abs(order_size) / contract_size * commission_per_lot

    return commission_fn


def _run_fractional_backtest(
    pdf: pd.DataFrame, config: Config
) -> tuple[pd.Series, FractionalBacktest]:
    bc, dc = config.backtest, config.data
    median_price = float(pdf["Close"].median())
    spread = _compute_spread_rate(
        bc.spread_ticks, bc.slippage_ticks, dc.tick_size, median_price
    )
    commission_fn = _make_commission_fn(bc.commission_per_lot, dc.contract_size)
    bt = FractionalBacktest(
        pdf,
        MLSignalStrategy,
        cash=bc.initial_capital,
        spread=spread,
        commission=commission_fn,
        margin=1.0 / float(bc.leverage),
        exclusive_orders=True,
        finalize_trades=True,
        fractional_unit=1.0,
    )
    stats = bt.run(
        atr_stop_mult=bc.atr_stop_multiplier,
        atr_tp_mult=bc.atr_tp_multiplier,
        lots_per_trade=bc.lots_per_trade,
        min_lots=bc.min_lots,
        max_lots=bc.max_lots,
        confidence_threshold=bc.confidence_threshold,
        contract_size=dc.contract_size,
        horizon_bars=config.labels.horizon_bars,
        max_drawdown_cutoff=bc.max_drawdown_cutoff,
        dd_cooldown_bars=bc.dd_cooldown_bars,
        max_open_positions=bc.max_open_positions,
        daily_loss_limit=bc.daily_loss_limit,
        min_bars_between_trades=bc.min_bars_between_trades,
    )
    return stats, bt


def _load_backtest_data(config: Config) -> tuple[pd.DataFrame, str]:
    preds_path = Path(config.paths.predictions)
    if not preds_path.exists():
        raise FileNotFoundError(f"Predictions not found: {preds_path}")
    preds_df = pl.read_csv(preds_path)

    test_path = (
        Path(config.paths.test_data) if hasattr(config.paths, "test_data") else None
    )
    labels_path = Path(config.paths.labels)
    if test_path and test_path.exists() and config.validation.method == "static":
        source = str(test_path)
        test_df = pl.read_parquet(test_path)
    else:
        if not labels_path.exists():
            raise FileNotFoundError("Neither test data nor labels found")
        source = str(labels_path)
        test_df = pl.read_parquet(labels_path)

    pdf = _prepare_df(
        test_df,
        preds_df,
        test_source=source,
        preds_source=str(preds_path),
    )
    # OOS date filter
    bc = config.backtest
    if bc.oob_start_date:
        pdf = pdf[pdf.index >= pd.Timestamp(bc.oob_start_date)]
    if bc.oob_end_date:
        pdf = pdf[pdf.index <= pd.Timestamp(bc.oob_end_date)]
    return pdf, source


def compute_backtest(config: Config) -> BacktestResult:
    """End-to-end: load → merge → run FractionalBacktest → (metrics, trades, bt)."""
    pdf, _ = _load_backtest_data(config)
    logger.info("Confidence threshold: %.2f", config.backtest.confidence_threshold)
    stats, bt = _run_fractional_backtest(pdf, config)
    metrics = _normalize_stats(stats)
    trades = _trades_to_list(
        stats.get("_trades", pd.DataFrame()),
        config.backtest.commission_per_lot,
        config.data.contract_size,
    )
    return metrics, trades, bt


def run_backtest_demo(config: Config) -> BacktestResult:
    """Run CFD backtest from config, save JSON results."""
    metrics, trades, bt = compute_backtest(config)

    out_path = Path(config.paths.backtest_results)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"metrics": metrics, "trades": trades}, f, indent=2, default=str)
    logger.info("Backtest results saved: %s", out_path)

    n = int(metrics.get("num_trades", 0))
    logger.info(
        "Trades: %d | Return: %.2f%% | DD: %.2f%% | PF: %.2f | Win: %.1f%%",
        n,
        metrics.get("return_pct", 0),
        metrics.get("max_drawdown_pct", 0),
        metrics.get("profit_factor", 0),
        metrics.get("win_rate_pct", 0),
    )

    return metrics, trades, bt
