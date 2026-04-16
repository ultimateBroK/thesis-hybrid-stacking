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

    Stop-loss: managed manually via market orders (no sl= parameter)
    to avoid backtesting.py's built-in tie-break bias when both TP and
    SL are hit on the same bar (the framework always attributes the
    exit to the Long side). Manual tracking also allows using a
    max(atr, min_atr) floor to prevent unrealistic stops in low-ATR
    regimes.
    """

    atr_stop_mult = 0.75
    lots_per_trade = 1.0
    confidence_threshold = 0.0  # 0 = disabled, trade all signals
    min_atr = 0.0001  # floor to prevent microscopic stops
    contract_size = 100  # units per lot (from DataConfig)
    horizon_bars = (
        0  # 0 = disabled (hold until opposite signal or stop); N = exit after N bars
    )

    def init(self) -> None:
        """
        Register required indicator series for the strategy and detect optional prediction probabilities.

        Registers `signals` from `data.pred_label` and `ATR` from `data.atr_14` with the backtesting indicator system, and sets an internal `_has_proba` flag indicating whether per-class probability columns exist. If probability columns are present, registers `proba_short` (`pred_proba_class_minus1`), `proba_hold` (`pred_proba_class_0`), and `proba_long` (`pred_proba_class_1`) as indicators.
        """
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

        # Manual stop tracking (filled on entry, consumed in next bar)
        # dict: direction -> stop_price
        self._stops: dict[str, float] = {}
        # Time-based exit: bar index when position was entered (direction -> bar index)
        self._entry_bar: dict[str, int] = {}

    def _floor_atr(self, atr: float) -> float:
        """Floor ATR to min_atr to prevent unrealistic stops in low-volatility regimes."""
        return max(atr, self.min_atr)

    def _check_stop(self) -> None:
        """
        Check if price has crossed the tracked stop level from the previous bar.

        Uses open price (execution price) vs stored stop to detect crossings.
        High/Low may briefly pierce the stop without actually filling at that price,
        so we use open for conservative detection.
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
                # Gap below stop: exit at stop
                self.position.close()
                self._stops.pop("long", None)
        elif self.position.is_short:
            sl = self._stops.get("short")
            if sl is not None and open_price >= sl:
                self.position.close()
                self._stops.pop("short", None)
            elif sl is not None and high >= sl > open_price:
                # Gap above stop: exit at stop
                self.position.close()
                self._stops.pop("short", None)

    def next(self) -> None:
        """
        Evaluate the latest model signal and, if appropriate, place a market order with an ATR-based stop-loss.

        Checks the most recent signal, optional predicted-class confidence (when `confidence_threshold > 0` and probability columns are available), and current position to decide whether to enter a new long or short trade. Uses manual stop tracking (market orders) instead of backtesting.py's sl= parameter to avoid the built-in tie-break bias when TP and SL are both hit on the same bar. A signal of 0 (hold) or a confidence value below the threshold results in no action.

        Time-based exit: when `horizon_bars > 0`, positions are closed after holding for `horizon_bars` bars
        (aligned with the labeling horizon). This prevents indefinite holding when no opposing signal occurs.
        """
        signal = int(self.signals[-1])
        price = self.data.Close[-1]
        atr = self._floor_atr(self.atr[-1])

        # Check manual stops first (from previous bar's entry)
        self._check_stop()

        # Time-based exit: close positions that have exceeded horizon_bars
        if self.horizon_bars > 0 and self.position:
            entry_bar = self._entry_bar.get("long") or self._entry_bar.get("short")
            if entry_bar is not None:
                bars_held = len(self.data) - entry_bar
                if bars_held >= self.horizon_bars:
                    self.position.close()
                    direction = "long" if self.position.is_long else "short"
                    self._entry_bar.pop(direction, None)
                    self._stops.pop(direction, None)

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
        size = self.lots_per_trade * self.contract_size

        if signal == 1 and not self.position:
            # Flat → enter long with manual stop
            sl = price - atr * self.atr_stop_mult
            self._stops["long"] = sl
            self._entry_bar["long"] = len(self.data)
            self.buy(size=size)
        elif signal == -1 and not self.position:
            # Flat → enter short with manual stop
            sl = price + atr * self.atr_stop_mult
            self._stops["short"] = sl
            self._entry_bar["short"] = len(self.data)
            self.sell(size=size)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_stats(stats: pd.Series) -> dict:
    """
    Convert a Backtesting.py statistics Series into a dictionary with snake_case keys.

    The function omits keys that begin with an underscore and normalizes display-style keys by lowercasing, replacing spaces and punctuation with underscores or removing them, and mapping `%` to `pct` and `#` to `num`.

    Parameters:
        stats (pd.Series): Series-like statistics object produced by Backtesting.py

    Returns:
        dict: A dictionary of normalized metric names to their original values.
    """
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
    """
    Convert a backtesting.py trades DataFrame into a JSON-serializable list of trade records.

    Each record contains entry/exit timestamps, direction ("long" or "short"), entry/exit prices, raw size, PnL, return percentage (converted from fractional `ReturnPct` to percent), rounded commission (computed as lots * `commission_per_lot`), and duration. If `trades_df` is empty, returns an empty list.

    Parameters:
        trades_df (pd.DataFrame): Raw trades DataFrame from backtesting.py stats.
        commission_per_lot (float): Commission charged per lot used to compute per-trade commission.
        contract_size (float): Units per lot used to convert raw `Size` into lot counts.

    Returns:
        list[dict]: List of trade dictionaries with keys:
            - entry_time (str)
            - exit_time (str)
            - direction (str): "long" if `Size` > 0, "short" otherwise.
            - entry_price (float)
            - exit_price (float)
            - size (float)
            - pnl (float)
            - return_pct (float): `ReturnPct` multiplied by 100.
            - commission (float): Rounded to two decimals.
            - duration (str)
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
    """
    Prepare and return a pandas DataFrame merging market data and prediction columns for consumption by backtesting.py.

    Parameters:
        test_df (pl.DataFrame): Market data with a `timestamp` column and required feature `atr_14`. Expected bar columns include `open`, `high`, `low`, `close`, and optionally `volume`.
        preds_df (pl.DataFrame): Predictions with a `timestamp` column and required `pred_label`. May include probability columns `pred_proba_class_minus1`, `pred_proba_class_0`, and `pred_proba_class_1`.

    Returns:
        pd.DataFrame: Pandas DataFrame indexed by `timestamp` (DatetimeIndex) with price columns renamed to `Open`, `High`, `Low`, `Close`, and `Volume` (set to 0 if absent). Prediction columns from `preds_df` are preserved.

    Raises:
        ValueError: If `pred_label` is missing from `preds_df` or `atr_14` is missing from `test_df`.
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
    """
    Run a backtest on prepared market and prediction data using HybridGRUStrategy, with spread, commission, and margin derived from the provided configuration.

    Parameters:
        pdf (pd.DataFrame): Pandas DataFrame indexed by timestamp containing market columns required by backtesting.py (Open, High, Low, Close, Volume) and strategy inputs (e.g., `pred_label`, `atr_14`).
        config (Config): Configuration object with `backtest` and `data` sections that supply cost parameters, contract sizing, leverage, initial capital, and strategy parameters.

    Returns:
        tuple[pd.Series, Backtest]: A pair where the first element is the backtesting statistics as a pandas Series and the second is the Backtest instance used to run the strategy.
    """
    bc = config.backtest
    dc = config.data

    # Spread: absolute ticks → relative rate
    median_price = float(pdf["Close"].median())
    spread_total = (bc.spread_ticks + bc.slippage_ticks) * dc.tick_size / median_price

    # Commission: callable for per-lot model
    def commission_fn(order_size: float, price: float) -> float:
        """
        Calculate commission for an order based on the number of lots and configured per-lot commission.

        Parameters:
                order_size (float): Signed order size in base units; absolute value is divided by the configured contract size to get lots.
                price (float): Order price (ignored by this calculation; included for compatibility).

        Returns:
                commission (float): Commission amount in the same currency as `bc.commission_per_lot`.
        """
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
        contract_size=dc.contract_size,
        horizon_bars=config.labels.horizon_bars,
    )
    return stats, bt


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_backtest(config: Config) -> None:
    """
    Run a full CFD backtest from files specified in `config` and persist results and artifacts.

    Loads test market data and model predictions from configured Parquet paths, validates required inputs, prepares merged market/prediction data, executes the backtest using the configured strategy and risk/cost settings, and writes outputs to disk. Written outputs include a JSON file with normalized metrics and trade records, an optional trades detail CSV and equity-curve CSV when trades are present, and an optional Bokeh HTML chart under the configured session directory.

    Parameters:
        config: Application configuration object containing paths and backtest/data settings (must provide
            `paths.test_data`, `paths.predictions`, `paths.backtest_results`; optional `paths.session_dir`).
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
    """
    Run the full backtest pipeline using in-memory Polars DataFrames and return normalized metrics.

    Parameters:
        test_df (pl.DataFrame): Market/test data containing price columns and `atr_14`.
        preds_df (pl.DataFrame): Predictions data containing `timestamp` and `pred_label` (optional `pred_proba_*` columns allowed).
        config (Config): Configuration object with `backtest`, `data`, and `paths` sections used to parameterize the backtest.

    Returns:
        dict: Normalized metrics dictionary extracted from the backtest results.
    """
    pdf = _prepare_df(test_df, preds_df)
    stats, _ = _run_bt(pdf, config)
    return _normalize_stats(stats)
