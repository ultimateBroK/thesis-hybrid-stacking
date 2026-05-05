"""CFD backtest simulation via backtesting.py.

Combines strategy, runners, persistence, and stats into a single module.

Barrier alignment requirement: The backtest SL/TP ATR multipliers
(``BacktestConfig.atr_stop_multiplier``, ``atr_tp_multiplier``) MUST
match the label barrier multipliers (``LabelsConfig.atr_sl_multiplier``,
``atr_tp_multiplier``).  Signals are generated from labels whose
barriers are placed at ±2×ATR; setting backtest SL/TP to any other
multiple would create a mismatch between the model's training target
and the execution risk envelope, degrading out-of-sample performance.
Both config sections should equal 2.0 for the thesis evaluation.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import polars as pl
from backtesting import Strategy
from backtesting.lib import FractionalBacktest

from thesis._shared.config import Config
from thesis._shared.ui import console

logger = logging.getLogger("thesis.backtest")


# Module-level fallback defaults

#: Floor to prevent microscopic stops in low-volatility regimes.
_MIN_ATR_FLOOR: float = 0.0001

#: Default initial capital used as fallback when BacktestConfig is unavailable.
_DEFAULT_INITIAL_CAPITAL: float = 10_000.0

#: Default commission per lot — fallback for _trades_to_list when config absent.
_DEFAULT_COMMISSION_PER_LOT: float = 20.0

#: Default contract size (units per lot) — fallback for _trades_to_list.
_DEFAULT_CONTRACT_SIZE: float = 100.0

#: Minimum bars required before the shifted-signal logic can activate.
_MIN_BARS_FOR_SIGNAL: int = 2

#: Minimum order size in units (backtesting.py requires whole-number sizes).
_MIN_ORDER_SIZE: int = 1


CORE_BACKTEST_METRICS: tuple[tuple[str, str, str], ...] = (
    ("return_pct", "Total Return", "{:.2f}%"),
    ("max_drawdown_pct", "Max Drawdown", "{:.2f}%"),
    ("profit_factor", "Profit Factor", "{:.2f}"),
    ("sharpe_ratio", "Sharpe Ratio", "{:.2f}"),
    ("win_rate_pct", "Win Rate", "{:.2f}%"),
    ("num_trades", "Trades", "{:,.0f}"),
)

CORE_BACKTEST_METRIC_KEYS = {
    "return_pct",
    "max_drawdown_pct",
    "profit_factor",
    "sharpe_ratio",
    "win_rate_pct",
    "num_trades",
    "equity_final",
    "start",
    "end",
    "sortino_ratio",
    "calmar_ratio",
    "expectancy_pct",
    "avg_trade_pct",
}


def _log_core_backtest_metrics(
    metrics: dict[str, Any], initial_capital: float = _DEFAULT_INITIAL_CAPITAL
) -> None:
    """Log only the finance metrics that matter for CLI readability.

    Args:
        metrics: Normalized backtest statistics from ``_normalize_stats``.
        initial_capital: Starting capital displayed in the log header.
    """
    logger.info("=== BACKTEST CORE METRICS ===")
    logger.info("  Initial Balance: %s", f"${initial_capital:,.0f}")
    for key, label, fmt in CORE_BACKTEST_METRICS:
        value = metrics.get(key)
        if value is None:
            continue
        logger.info("  %s: %s", label, fmt.format(value))

    equity_final = metrics.get("equity_final")
    if equity_final is not None:
        logger.info("  Final Equity: %s", f"${equity_final:,.0f}")


def _calendar_day(value: object) -> object:
    """Return the calendar date for a timestamp-like value.

    Args:
        value: A timestamp object (Pandas Timestamp, datetime, or
            string parseable by ``pd.Timestamp``).

    Returns:
        The calendar date portion as a ``datetime.date`` object.
    """
    return pd.Timestamp(value).date()


# Strategy


class HybridGRUStrategy(Strategy):
    """Trade on ML signals with ATR stop-loss and equity risk management.

    Signal shift: the strategy reads ``self.signals[-2]`` instead of
    ``-1`` so that the trade decision at bar ``i`` is based on the
    prediction made at bar ``i-1``.  This aligns the label anchor
    (``close[i-1]``) with the approximate entry price (``open[i]``),
    since the label for bar ``i-1`` uses barriers centred on
    ``close[i-1]``.  Without this shift the prediction at bar ``i``
    (anchored at ``close[i]``) would be executed at ``open[i+1]`` — a
    one-bar gap that breaks the anchor-entry correspondence.

    Position sizing: fixed-risk after confidence filtering. Confidence decides
    whether a trade is allowed; lot size stays at ``lots_per_trade`` clamped to
    ``[min_lots, max_lots]``.

    Confidence filtering: when confidence_threshold > 0, only trade
    when the predicted class probability exceeds the threshold.

    Stop-loss: set via backtesting.py's native ``sl=`` parameter on
    buy()/sell() calls. The stop price is computed as entry_price ±
    (ATR × atr_stop_mult), floored by min_atr to prevent unrealistic
    stops in low-ATR regimes. A manual stop-check fallback also closes
    positions when the open/low/high crosses the tracked stop level,
    providing conservative detection against brief pierces.

    Take-profit: when ``atr_tp_mult > 0``, a TP price is set at
    entry_price ± (ATR × atr_tp_mult), creating an asymmetric
    risk-reward profile (e.g. 1:2 with SL=1×ATR, TP=2×ATR).

    Risk management includes a max-drawdown circuit breaker, a maximum open
    position limit, and a daily loss limit based on the day's starting equity.

    Attributes:
        atr_stop_mult: ATR multiplier for stop-loss distance.
        atr_tp_mult: ATR multiplier for take-profit distance (0 = disabled).
        lots_per_trade: Fixed lot size after confidence filtering.
        min_lots: Minimum lot safety bound.
        max_lots: Maximum lot safety bound.
        confidence_threshold: Minimum class probability to trade (0 = disabled).
        min_atr: Floor to prevent microscopic stops in low-vol regimes.
        contract_size: Units per lot.
        horizon_bars: Max bars to hold (0 = hold until opposite signal/stop).
        max_drawdown_cutoff: Fraction of peak equity — breach triggers cooldown.
        dd_cooldown_bars: Bars to pause trading after drawdown breach.
        max_open_positions: Max simultaneous open positions.
        daily_loss_limit: Max fraction of daily equity loss before pause.
        min_bars_between_trades: Minimum bars after position exit before re-entry.
    """

    # ── Strategy fallback defaults ──
    # Runtime configuration is passed through bt.run() keyword arguments.
    # These class attributes provide safety defaults for direct Strategy use
    # and for parameters omitted by a caller. Short Strategy attribute names
    # are mapped from configuration fields in _run_bt() and run_backtest_manual().
    # ───────────────────────────────────────────────────────────────────────

    atr_stop_mult = 1.0  # cf. BacktestConfig.atr_stop_multiplier = 2.0
    atr_tp_mult = 2.0  # 0 = disabled (no take-profit); matches BacktestConfig
    lots_per_trade = 0.2  # cf. BacktestConfig.lots_per_trade = 0.1
    min_lots = 0.1  # cf. BacktestConfig.min_lots = 0.01
    max_lots = 0.5  # matches BacktestConfig
    confidence_threshold = 0.0  # 0 = disabled (trade all); cf. BacktestConfig = 0.50
    min_atr = (
        _MIN_ATR_FLOOR  # floor to prevent microscopic stops (module-level constant)
    )
    contract_size = 100  # units per lot; overridden via DataConfig.contract_size
    horizon_bars = 0  # 0 = disabled (hold until opposite signal or stop); overridden via LabelsConfig
    max_drawdown_cutoff = 0.50  # circuit breaker threshold; cf. BacktestConfig = 0.30
    dd_cooldown_bars = (
        12  # pause duration after drawdown breach; matches BacktestConfig
    )
    max_open_positions = 1  # max simultaneous positions; matches BacktestConfig
    daily_loss_limit = 0.03  # daily loss fraction limit; matches BacktestConfig
    min_bars_between_trades = 0  # 0 = disabled; min bars after exit before re-entry

    def init(self) -> None:
        """Register indicators and initialise risk-management state.

        Registers ``signals`` from ``data.pred_label`` and ``ATR`` from
        ``data.atr_14`` with the backtesting indicator system, and sets an
        internal ``_has_proba`` flag indicating whether per-class probability
        columns exist.

        Initializes peak-equity tracking, drawdown cooldown, daily start
        equity, and current-date state for risk management.
        """
        self._initial_capital = self.equity

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
                lambda: self.data.pred_proba_class_0, name="proba_hold", plot=False
            )
            self.proba_long = self.I(
                lambda: self.data.pred_proba_class_1, name="proba_long", plot=False
            )

        self._entry_bar: dict[str, int] = {}

        # Cooldown state — prevent overtrading after position closure
        self._last_exit_bar: int = 0  # bar index of last position exit (0 = none yet)
        self._position_was_open: bool = False  # was a position open at prior bar?

        # Risk-management state
        self._peak_equity: float = self.equity
        self._dd_cooldown_left: int = 0
        self._dd_cutoff_breached: bool = False
        self._daily_start_equity: float = self.equity
        self._current_date: object = None  # track calendar day for daily reset

    def _floor_atr(self, atr: float) -> float:
        """Floor ATR to prevent unrealistic stops in low-volatility regimes.

        Args:
            atr: Current Average True Range value.

        Returns:
            ``max(atr, self.min_atr)`` — guaranteed above the module-level
            ``_MIN_ATR_FLOOR``.
        """
        return max(atr, self.min_atr)

    def _update_risk_state(self) -> None:
        """Update peak equity, drawdown tracking, and daily loss tracking.

        Called every bar in ``next()`` before any trading decisions.

        Maintains peak equity, drawdown cooldown, daily start equity, and the
        calendar-day tracker for daily reset logic. The drawdown circuit
        breaker is a permanent shutdown: once triggered, no new positions are
        opened for the rest of the backtest.
        """
        eq = self.equity
        self._peak_equity = max(self._peak_equity, eq)

        # Drawdown circuit breaker — decrement cooldown each bar
        if self._dd_cooldown_left > 0:
            self._dd_cooldown_left -= 1

        # Check if drawdown exceeds cutoff. For thesis evaluation, stop opening
        # new positions after a catastrophic drawdown instead of repeatedly
        # logging cooldown events for the rest of the backtest.
        if self.max_drawdown_cutoff > 0 and self._peak_equity > 0:
            dd = (self._peak_equity - eq) / self._peak_equity
            if dd >= self.max_drawdown_cutoff and not self._dd_cutoff_breached:
                self._dd_cutoff_breached = True
                self._dd_cooldown_left = self.dd_cooldown_bars
                logger.warning(
                    "Drawdown circuit breaker triggered: %.1f%% drawdown "
                    "exceeds %.1f%% cutoff — blocking new trades",
                    dd * 100,
                    self.max_drawdown_cutoff * 100,
                )

        # Daily loss tracking — reset at start of each new calendar day
        bar_date = _calendar_day(self.data.index[-1])
        if self._current_date != bar_date:
            self._current_date = bar_date
            self._daily_start_equity = eq

    def _is_trading_allowed(self) -> bool:
        """Check all risk gates before opening a new position.

        Returns:
            True if trading is permitted, False if any gate blocks it.
        """
        # Gate 1: max open positions
        if len(self.orders) > 0:
            return False

        if self.position and self.max_open_positions <= 1:
            return False

        # Gate 2: trade cooldown — enforce minimum bars between exits and re-entries
        if (
            self.min_bars_between_trades > 0
            and self._last_exit_bar > 0
            and (len(self.data) - self._last_exit_bar) < self.min_bars_between_trades
        ):
            return False

        # Gate 3: drawdown circuit breaker
        if self._dd_cutoff_breached:
            return False

        if self._dd_cooldown_left > 0:
            return False

        # Gate 4: daily loss limit
        if self.daily_loss_limit > 0 and self._daily_start_equity > 0:
            daily_pnl = (
                self.equity - self._daily_start_equity
            ) / self._daily_start_equity
            if daily_pnl <= -self.daily_loss_limit:
                return False

        return True

    def _compute_lots(self, confidence: float | None) -> float:
        """Return fixed position size after confidence filtering.

        Confidence already gates whether a trade is allowed. Scaling lots by
        confidence amplified wrong high-confidence predictions in the latest
        OOS run, causing drawdown to grow far faster than signal quality. Keep
        sizing fixed until the model is profitable at the base risk level.

        Args:
            confidence: Predicted class probability, accepted for API stability.

        Returns:
            Fixed lot size clamped to configured safety bounds.
        """
        return max(self.min_lots, min(self.lots_per_trade, self.max_lots))

    def next(self) -> None:
        """Evaluate the latest model signal and place orders if appropriate.

        Processes cooldown tracking, risk-state updates, time-based exits,
        risk gates, confidence gates, position sizing, and ATR-based market
        orders. The backtesting engine fills orders on the next bar, so signal
        bar ``i`` cannot trade at the same bar's close.
        """
        # Step 0: cooldown tracking — detect auto-closure from framework SL/TP
        if self._position_was_open and not self.position:
            self._last_exit_bar = len(self.data)
            self._position_was_open = False

        # Step 1: update risk state every bar
        self._update_risk_state()

        # Signal shift: use signals[-2] so the trade decision at bar i
        # is based on the prediction made at bar i-1.  The label for
        # bar i-1 is anchored at close[i-1]; backtesting.py fills
        # orders at the next open, so the approximate entry price is
        # open[i] ≈ close[i-1].  Without the shift, pred_label[i]
        # (anchored at close[i]) would be executed at open[i+1], a
        # one-bar misalignment between label anchor and entry price.
        if len(self.signals) < _MIN_BARS_FOR_SIGNAL:
            return
        raw_signal = float(self.signals[-2])
        # Threshold continuous predictions at 0 for direction:
        #   pred > 0  → Long  (1)
        #   pred < 0  → Short (-1)
        #   pred == 0 → Hold  (0)
        if raw_signal not in (-1, 0, 1):
            signal = 1 if raw_signal > 0 else (-1 if raw_signal < 0 else 0)
        else:
            signal = int(raw_signal)
        atr = self._floor_atr(self.atr[-1])

        # Step 2: time-based exit
        if self.horizon_bars > 0 and self.position:
            entry_bar = self._entry_bar.get("long") or self._entry_bar.get("short")
            if entry_bar is not None:
                bars_held = len(self.data) - entry_bar
                if bars_held >= self.horizon_bars:
                    self.position.close()
                    direction = "long" if self.position.is_long else "short"
                    self._entry_bar.pop(direction, None)
                    self._last_exit_bar = len(self.data)
                    self._position_was_open = False

        # Step 3: risk gate — no new trades if blocked
        if not self._is_trading_allowed():
            return

        # Step 4: confidence gate
        confidence: float | None = None
        if self.confidence_threshold > 0 and self._has_proba:
            # Confidence must use the same bar as the shifted signal
            if signal == 1:
                confidence = float(self.proba_long[-2])
            elif signal == -1:
                confidence = float(self.proba_short[-2])
            else:
                return

            if confidence < self.confidence_threshold:
                return
        elif self.confidence_threshold > 0 and not self._has_proba:
            # Regression mode: no probability columns — skip confidence gate
            pass

        # Step 5: position sizing
        proxy_entry_price = self.data.Close[-1]
        lots = self._compute_lots(confidence)
        size = lots * self.contract_size
        # backtesting.py requires whole-number units (or equity fraction <1)
        size = max(_MIN_ORDER_SIZE, round(size))

        # Step 6: execute trades
        if signal == 1 and not self.position:
            self._entry_bar["long"] = len(self.data)
            sl_price = proxy_entry_price - (atr * self.atr_stop_mult)
            tp_price = (
                proxy_entry_price + (atr * self.atr_tp_mult)
                if self.atr_tp_mult > 0
                else None
            )
            self.buy(size=size, sl=sl_price, tp=tp_price)
            self._position_was_open = True

        elif signal == -1 and not self.position:
            self._entry_bar["short"] = len(self.data)
            sl_price = proxy_entry_price + (atr * self.atr_stop_mult)
            tp_price = (
                proxy_entry_price - (atr * self.atr_tp_mult)
                if self.atr_tp_mult > 0
                else None
            )
            self.sell(size=size, sl=sl_price, tp=tp_price)
            self._position_was_open = True


# Statistics


def _normalize_stats(stats: pd.Series) -> dict:
    """Convert Backtesting.py statistics into the curated core metric dict.

    Only export metrics that are shown in dashboard/CLI.  This prevents
    downstream artifacts from becoming a noisy dump of technical finance
    parameters (Sortino, Calmar, SQN, Kelly, recovery factor, avg win/loss,
    etc.).  Backtesting.py still computes its internal stats while running;
    this function decides what the thesis workflow keeps and saves.

    Args:
        stats: Raw ``pd.Series`` from ``backtesting.py`` stats output.

    Returns:
        Dictionary containing only the keys listed in
        ``CORE_BACKTEST_METRIC_KEYS``, with keys normalized to
        snake_case.
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
        if key in CORE_BACKTEST_METRIC_KEYS:
            out[key] = v

    return out


# Persistence


def _trades_to_list(
    trades_df: pd.DataFrame,
    commission_per_lot: float = _DEFAULT_COMMISSION_PER_LOT,
    contract_size: float = _DEFAULT_CONTRACT_SIZE,
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
    initial_capital: float = _DEFAULT_INITIAL_CAPITAL,
) -> None:
    """Save equity curve as CSV with running peak and drawdown.

    Each row represents a closed trade with the running equity, peak equity,
    and drawdown percentage.

    The equity curve is trade-by-trade closed PnL, not mark-to-market, so
    intra-trade drawdowns are not visible.

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


# Runners — Data Preparation


def _validate_backtest_merge(
    *,
    feature_rows: int,
    prediction_rows: int,
    merged_rows: int,
    test_source: str = "<in-memory test/features>",
    preds_source: str = "<in-memory predictions>",
) -> None:
    """Guard against silent timestamp loss in the backtest inner join."""
    coverage = merged_rows / prediction_rows if prediction_rows else 0.0
    dropped = prediction_rows - merged_rows
    logger.info(
        "Backtest merge: features_rows=%d predictions_rows=%d merged_rows=%d "
        "coverage=%.2f%% dropped_predictions=%d",
        feature_rows,
        prediction_rows,
        merged_rows,
        coverage * 100.0,
        dropped,
    )
    if coverage < 0.99:
        raise ValueError(
            "Backtest merge coverage below 99%: "
            f"expected>=99.00%, actual={coverage * 100.0:.2f}%, "
            f"features_rows={feature_rows}, predictions_rows={prediction_rows}, "
            f"merged_rows={merged_rows}, dropped_predictions={dropped}, "
            f"features_path={test_source}, predictions_path={preds_source}. "
            "Check timestamp alignment before backtesting."
        )


def _prepare_df(
    test_df: pl.DataFrame,
    preds_df: pl.DataFrame,
    *,
    test_source: str = "<in-memory test/features>",
    preds_source: str = "<in-memory predictions>",
) -> pd.DataFrame:
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
    _validate_backtest_merge(
        feature_rows=len(test),
        prediction_rows=len(preds),
        merged_rows=len(merged),
        test_source=test_source,
        preds_source=preds_source,
    )

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


# Runners — Configuration Helpers


def _compute_spread_rate(
    bc: Config.BacktestConfig,
    dc: Config.DataConfig,
    median_price: float,
) -> float:
    """Convert tick-based spread to relative rate for backtesting.py.

    Args:
        bc: Backtest configuration with spread_ticks and slippage_ticks.
        dc: Data configuration with tick_size.
        median_price: Median close price used as normalization denominator.

    Returns:
        Relative spread rate as a fraction.
    """
    total_ticks = bc.spread_ticks + bc.slippage_ticks
    return total_ticks * dc.tick_size / median_price


def _build_commission_fn(
    bc: Config.BacktestConfig,
    dc: Config.DataConfig,
) -> Callable[[float, float], float]:
    """Build a commission function closure for backtesting.py.

    Args:
        bc: Backtest configuration with commission_per_lot.
        dc: Data configuration with contract_size.

    Returns:
        A commission function that takes (order_size, price) and returns
        commission in dollars.
    """

    def commission_fn(order_size: float, price: float) -> float:  # noqa: ARG001
        lots = abs(order_size) / dc.contract_size
        return lots * bc.commission_per_lot

    return commission_fn


def _init_backtest(
    pdf: pd.DataFrame,
    bc: Config.BacktestConfig,
    dc: Config.DataConfig,
    spread: float,
    commission_fn: Callable[[float, float], float],
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


# Public API


def run_backtest(config: Config) -> None:
    """Run a full CFD backtest from files specified in config.

    For walk-forward (sliding) validation, joins OOF predictions with the
    full labeled dataset (which contains OHLCV + features). For static
    validation, uses the traditional test split file.

    Writes normalized metrics and trade records as JSON, optional trade-detail
    and equity-curve CSV files, and an optional Bokeh HTML chart.

    Args:
        config: Application configuration object containing paths and
            backtest/data settings.
    """
    preds_path = Path(config.paths.predictions)
    if not preds_path.exists():
        raise FileNotFoundError(f"Predictions not found: {preds_path}")
    with console.status(f"[cyan]Loading predictions[/] {preds_path}"):
        preds_df = pl.read_parquet(preds_path)

    # Walk-forward: predictions are OOF across all windows — need OHLCV from labels
    # Static: predictions are for the test split — need OHLCV from test split
    test_path = Path(config.paths.test_data)
    is_static = config.validation.method == "static"

    if test_path.exists() and is_static:
        with console.status(f"[cyan]Loading static test data[/] {test_path}"):
            test_df = pl.read_parquet(test_path)
    elif test_path.exists() and not is_static:
        logger.warning(
            "Static test file found (%s) but workflow is walk-forward "
            "(method='%s') — ignoring stale test_data in favor of OOF predictions",
            test_path,
            config.validation.method,
        )
        labels_path = Path(config.paths.labels)
        if not labels_path.exists():
            raise FileNotFoundError(
                f"Labels file not found ({labels_path}) — needed for walk-forward backtest"
            )
        with console.status(f"[cyan]Loading labels for backtest[/] {labels_path}"):
            test_df = pl.read_parquet(labels_path)
    else:
        labels_path = Path(config.paths.labels)
        if not labels_path.exists():
            raise FileNotFoundError(
                f"Neither test data ({test_path}) nor labels ({labels_path}) found"
            )
        logger.info("Walk-forward mode: joining OOF predictions with labeled data")
        with console.status(f"[cyan]Loading labels for backtest[/] {labels_path}"):
            test_df = pl.read_parquet(labels_path)

    pdf = _prepare_df(
        test_df,
        preds_df,
        test_source=str(test_path if test_path.exists() and is_static else labels_path),
        preds_source=str(preds_path),
    )

    # ── OOS date-range filter ──
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

    logger.info("Confidence threshold: %.2f", config.backtest.confidence_threshold)
    with console.status("[cyan]Running CFD backtest[/]"):
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
            else _DEFAULT_INITIAL_CAPITAL
        )
        _save_equity_curve_csv(trades, out_path.parent, initial_capital)

    _log_core_backtest_metrics(metrics, config.backtest.initial_capital)

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
) -> tuple[dict, list[dict]]:
    """Run a backtest with manually specified parameters (no Config required).

    Designed for interactive use in dashboards where parameters can be tuned
    without modifying the config file.

    Args:
        test_df: Market/test data with OHLCV columns and atr_14.
        preds_df: Predictions with timestamp and pred_label (optionally
            pred_proba_* columns).
        leverage: CFD leverage ratio (default 100).
        lots_per_trade: Fixed lot size after confidence filtering.
        min_lots: Minimum lot safety bound.
        max_lots: Maximum lot safety bound.
        confidence_threshold: Minimum prediction probability to trade (0 = disabled).
        spread_ticks: Spread in ticks.
        slippage_ticks: Slippage in ticks.
        commission_per_lot: Commission per lot.
        atr_stop_multiplier: ATR multiplier for stop-loss distance (default 1.0).
        atr_tp_multiplier: ATR multiplier for take-profit distance (default 2.0, 0 = disabled).
        horizon_bars: Time-based exit after N bars (default 10).
        contract_size: Units per lot.
        tick_size: Price tick size in dollars (default 0.01).
        initial_capital: Starting capital for the backtest.
        max_drawdown_cutoff: Circuit breaker drawdown fraction (0.5 = 50%).
        dd_cooldown_bars: Bars to pause after drawdown breach.
        max_open_positions: Max simultaneous open positions.
        daily_loss_limit: Daily equity loss fraction before pause.
        min_bars_between_trades: Minimum bars between position exit and next entry (0 = disabled, default 6).

    Returns:
        Tuple of (metrics dict, trades list). Metrics contains normalized
        performance metrics; trades is a list of per-trade records.
    """
    pdf = _prepare_df(test_df, preds_df)

    median_price = float(pdf["Close"].median())
    spread_total = (spread_ticks + slippage_ticks) * tick_size / median_price

    def commission_fn(order_size: float, price: float) -> float:  # noqa: ARG001
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
        stats["_trades"],
        commission_per_lot=commission_per_lot,
        contract_size=contract_size,
    )

    return metrics, trades
