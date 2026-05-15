"""ML signal strategy. ATR stops, confidence gate, drawdown circuit breaker."""

from __future__ import annotations

import logging

from backtesting import Strategy
import pandas as pd

logger = logging.getLogger("thesis.backtest")

_MIN_ATR_FLOOR: float = 0.0001
_DEFAULT_CONTRACT_SIZE: float = 100.0
_MIN_BARS_FOR_SIGNAL: int = 2
_MIN_ORDER_SIZE: int = 1


def _calendar_day(value: object) -> object:
    """Return calendar date (NY market close = 5PM → next day boundary)."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    ts_ny = ts.tz_convert("America/New_York")
    ny_offset_hours = ts_ny.utcoffset().total_seconds() / 3600
    hours_to_midnight_ny = 24 - (17 + 7) + ny_offset_hours
    return (ts_ny + pd.Timedelta(hours=hours_to_midnight_ny)).date()


class MLSignalStrategy(Strategy):
    """Trade ML signals. ATR stops/TPs, confidence filter, daily loss gate.

    Signal shift: pred_label[i-1] → trade at open[i] (next bar fill).
    Runtime config via bt.run() kwargs.
    """

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

    def init(self) -> None:
        """Register indicators. Init risk state."""
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
        self._last_exit_bar: int = 0
        self._position_was_open: bool = False

        self._peak_equity: float = self.equity
        self._dd_cooldown_left: int = 0
        self._dd_cutoff_breached: bool = False
        self._daily_start_equity: float = self.equity
        self._current_date: object = None

    def _floor_atr(self, atr: float) -> float:
        """Floor ATR at min_atr to prevent zero stops."""
        return max(atr, self.min_atr)

    def _update_risk_state(self) -> None:
        """Update peak equity, drawdown cooldown, daily P&L."""
        eq = self.equity
        self._peak_equity = max(self._peak_equity, eq)

        if self._dd_cooldown_left > 0:
            self._dd_cooldown_left -= 1

        if self.max_drawdown_cutoff > 0 and self._peak_equity > 0:
            dd = (self._peak_equity - eq) / self._peak_equity
            if dd >= self.max_drawdown_cutoff and not self._dd_cutoff_breached:
                self._dd_cutoff_breached = True
                self._dd_cooldown_left = self.dd_cooldown_bars
                logger.warning(
                    "Drawdown circuit breaker triggered: %.1f%% > %.1f%% cutoff",
                    dd * 100,
                    self.max_drawdown_cutoff * 100,
                )

        bar_date = _calendar_day(self.data.index[-1])
        if self._current_date != bar_date:
            self._current_date = bar_date
            self._daily_start_equity = eq

    def _is_trading_allowed(self) -> bool:
        """Check all risk gates. Return True if new trades allowed."""
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

    def _compute_lots(self, confidence: float | None) -> float:
        """Return lot size clamped to min/max."""
        return max(self.min_lots, min(self.lots_per_trade, self.max_lots))

    def next(self) -> None:
        """Evaluate signal, place orders if risk gates pass."""
        if self._position_was_open and not self.position:
            self._last_exit_bar = len(self.data)
            self._position_was_open = False
            self._entry_bar.pop("long", None)
            self._entry_bar.pop("short", None)

        self._update_risk_state()

        if len(self.signals) < _MIN_BARS_FOR_SIGNAL:
            return
        raw_signal = float(self.signals[-2])
        if raw_signal not in (-1, 0, 1):
            signal = 1 if raw_signal > 0 else (-1 if raw_signal < 0 else 0)
        else:
            signal = int(raw_signal)
        atr = self._floor_atr(self.atr[-1])

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

        if not self._is_trading_allowed():
            return

        confidence: float | None = None
        if self.confidence_threshold > 0 and self._has_proba:
            if signal == 1:
                confidence = float(self.proba_long[-2])
            elif signal == -1:
                confidence = float(self.proba_short[-2])
            else:
                return

            if confidence < self.confidence_threshold:
                return

        proxy_entry_price = self.data.Close[-1]
        lots = self._compute_lots(confidence)
        size = lots * self.contract_size
        size = max(_MIN_ORDER_SIZE, round(size))

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
