"""Backtest strategy using backtesting.py — HybridGRU strategy class.

Contains the ``HybridGRUStrategy`` class which trades on ML-predicted signals
with ATR-based stop-loss. All runner/persistence/stats logic lives in sibling
modules.
"""

from backtesting import Strategy


class HybridGRUStrategy(Strategy):
    """Trade on ML signals with ATR stop-loss.

    No manual signal shift — backtesting.py natively delays execution
    by 1 bar (evaluates at close[i], executes at open[i+1]).

    Position sizing: fixed lot size per trade (avoids runaway sizing
    that occurs when buy()/sell() are called without explicit size
    and backtesting.py defaults to "max affordable" with leverage).
    When auto_lot_sizing=True, lot size is calculated dynamically
    based on current equity and risk_per_trade_pct.

    Confidence filtering: when confidence_threshold > 0, only trade
    when the predicted class probability exceeds the threshold.

    Stop-loss: set via backtesting.py's native ``sl=`` parameter on
    buy()/sell() calls. The stop price is computed as entry_price ±
    (ATR × atr_stop_mult), floored by min_atr to prevent unrealistic
    stops in low-ATR regimes. A manual stop-check fallback also closes
    positions when the open/low/high crosses the tracked stop level,
    providing conservative detection against brief pierces.

    Attributes:
        atr_stop_mult: ATR multiplier for stop-loss distance.
        lots_per_trade: Fixed lot size when auto_lot_sizing is False.
        confidence_threshold: Minimum class probability to trade (0 = disabled).
        min_atr: Floor to prevent microscopic stops in low-vol regimes.
        contract_size: Units per lot (e.g., 100 oz for XAUUSD).
        horizon_bars: Max bars to hold (0 = hold until opposite signal/stop).
        auto_lot_sizing: Enable dynamic position sizing.
        risk_per_trade_pct: Risk per trade as % of equity.
        min_lot_size: Minimum lot size for auto sizing.
        max_lot_size: Maximum lot size for auto sizing.
        enable_performance_adjustment: Scale size with equity performance.
        enable_volatility_adjustment: Reduce size in high-volatility periods.
        max_capital_risk_pct: Max % of initial capital risked per trade.
        performance_multiplier: Max position size increase factor.
        performance_reduction: Min position size reduction factor.
    """

    atr_stop_mult = 0.75
    lots_per_trade = 1.0
    confidence_threshold = 0.0  # 0 = disabled, trade all signals
    min_atr = 0.0001  # floor to prevent microscopic stops
    contract_size = 100  # units per lot (from DataConfig)
    horizon_bars = (
        0  # 0 = disabled (hold until opposite signal or stop); N = exit after N bars
    )
    # Auto lot sizing parameters
    auto_lot_sizing = False
    risk_per_trade_pct = 1.0
    min_lot_size = 0.1
    max_lot_size = 10.0
    # Enhanced auto lot sizing parameters
    enable_performance_adjustment = True
    enable_volatility_adjustment = True
    max_capital_risk_pct = 10.0
    performance_multiplier = 1.2
    performance_reduction = 0.8

    def init(self) -> None:
        """Register indicators and detect optional prediction probabilities.

        Registers ``signals`` from ``data.pred_label`` and ``ATR`` from
        ``data.atr_14`` with the backtesting indicator system, and sets an
        internal ``_has_proba`` flag indicating whether per-class probability
        columns exist.
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

    def _floor_atr(self, atr: float) -> float:
        """Floor ATR to prevent unrealistic stops in low-volatility regimes."""
        return max(atr, self.min_atr)

    def _calc_lot_size(self, atr: float, entry_price: float) -> float:
        """Calculate dynamic lot size based on risk management rules.

        Considers initial capital, current equity performance, ATR-based
        volatility adjustment, and maximum drawdown protection.

        Args:
            atr: Current ATR value for stop-loss calculation.
            entry_price: Expected entry price for volatility normalization.

        Returns:
            Lot size clamped between min_lot_size and max_lot_size.
            Falls back to lots_per_trade when auto_lot_sizing is False or
            computed size is less than 1 lot.
        """
        if not self.auto_lot_sizing:
            return self.lots_per_trade

        risk_amount = self.equity * (self.risk_per_trade_pct / 100.0)

        performance_adjustment = 1.0
        if self.enable_performance_adjustment:
            performance_ratio = self.equity / self._initial_capital
            performance_adjustment = min(
                self.performance_multiplier,
                max(self.performance_reduction, performance_ratio),
            )

        volatility_adjustment = 1.0
        if self.enable_volatility_adjustment:
            current_price = (
                self.data.Close[-1] if hasattr(self.data, "Close") else 2000.0
            )
            atr_ratio = atr / current_price
            volatility_adjustment = min(1.0, max(0.5, 1.0 - (atr_ratio - 0.01) * 10))

        adjusted_risk_amount = (
            risk_amount * performance_adjustment * volatility_adjustment
        )

        stop_distance_dollars = atr * self.atr_stop_mult
        risk_per_lot = stop_distance_dollars * self.contract_size

        if risk_per_lot <= 0 or adjusted_risk_amount <= 0:
            return self.lots_per_trade

        lot_size = adjusted_risk_amount / risk_per_lot

        max_risk_per_trade = self._initial_capital * (self.max_capital_risk_pct / 100.0)
        max_lot_by_capital = max_risk_per_trade / risk_per_lot
        lot_size = min(lot_size, max_lot_by_capital)

        lot_size = max(self.min_lot_size, min(self.max_lot_size, lot_size))

        if lot_size < 1:
            return self.lots_per_trade

        return round(lot_size)

    def _check_stop(self) -> None:
        """Check if price has crossed the tracked stop level.

        Uses open price (execution price) vs stored stop to detect crossings.
        High/Low may briefly pierce the stop without actually filling at that
        price, so we use open for conservative detection.
        """
        if not self._stops or not self.position:
            return
        open_price = self.data.Open[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]

        if self.position.is_long:
            sl = self._stops.get("long")
            if sl is not None and open_price <= sl:
                self.position.close()
                self._stops.pop("long", None)
            elif sl is not None and low <= sl < open_price:
                self.position.close()
                self._stops.pop("long", None)
        elif self.position.is_short:
            sl = self._stops.get("short")
            if sl is not None and open_price >= sl:
                self.position.close()
                self._stops.pop("short", None)
            elif sl is not None and high >= sl > open_price:
                self.position.close()
                self._stops.pop("short", None)

    def next(self) -> None:
        """Evaluate the latest model signal and place orders if appropriate.

        Processing order:
            1. Time-based exit — close positions exceeding horizon_bars.
            2. Confidence gate — skip low-confidence signals.
            3. Dynamic lot sizing — calculate size based on equity and risk.
            4. Execute trades with native ATR-based stop-loss.
        """
        signal = int(self.signals[-1])
        atr = self._floor_atr(self.atr[-1])

        if self.horizon_bars > 0 and self.position:
            entry_bar = self._entry_bar.get("long") or self._entry_bar.get("short")
            if entry_bar is not None:
                bars_held = len(self.data) - entry_bar
                if bars_held >= self.horizon_bars:
                    self.position.close()
                    direction = "long" if self.position.is_long else "short"
                    self._entry_bar.pop(direction, None)

        if self.confidence_threshold > 0 and self._has_proba:
            if signal == 1:
                confidence = float(self.proba_long[-1])
            elif signal == -1:
                confidence = float(self.proba_short[-1])
            else:
                return

            if confidence < self.confidence_threshold:
                return

        proxy_entry_price = self.data.Close[-1]
        lot_size = self._calc_lot_size(atr, proxy_entry_price)
        size = lot_size * self.contract_size

        if signal == 1 and not self.position:
            self._entry_bar["long"] = len(self.data)
            sl_price = proxy_entry_price - (atr * self.atr_stop_mult)
            self.buy(size=size, sl=sl_price)

        elif signal == -1 and not self.position:
            self._entry_bar["short"] = len(self.data)
            sl_price = proxy_entry_price + (atr * self.atr_stop_mult)
            self.sell(size=size, sl=sl_price)
