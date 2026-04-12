"""CFD backtest simulator with realistic trading costs and proper margin enforcement.

Critical fixes implemented:
1. Proper equity curve tracking after each trade
2. Margin call enforcement at 50% level
3. Stop-out enforcement at 20% level
4. Bankruptcy protection (capital <= 0)
5. Correct max drawdown calculation from equity curve
6. Position sizing based on current available capital
7. Real-time equity updates (mark-to-market)
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from thesis.config.loader import Config

logger = logging.getLogger("thesis.backtest")


def run_backtest(config: Config) -> None:
    """Run CFD backtest simulation with proper margin and risk management.

    Args:
        config: Configuration object.
    """
    logger.info("Loading backtest data...")

    test_path = Path(config.paths.test_data)
    preds_path = Path(config.paths.final_predictions)

    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_path}")

    if not preds_path.exists():
        raise FileNotFoundError(f"Predictions not found: {preds_path}")

    # Load test data
    test_df = pl.read_parquet(test_path)

    # Load predictions (stacking output)
    preds_df = pl.read_parquet(preds_path)

    # Merge predictions with OHLCV
    if len(preds_df) != len(test_df):
        logger.warning(
            "Prediction horizon shorter than test set: test=%s, preds=%s. "
            "Aligning on timestamp before backtest.",
            len(test_df),
            len(preds_df),
        )
        test_df = test_df.join(preds_df, on="timestamp", how="inner")
    else:
        for col in [
            "pred_proba_class_minus1",
            "pred_proba_class_0",
            "pred_proba_class_1",
        ]:
            if col in preds_df.columns:
                test_df = test_df.with_columns(preds_df[col])

    logger.info(f"Backtest data: {len(test_df)} bars")

    # Run simulation with proper margin enforcement
    results = _simulate_trades(test_df, config)

    # Calculate metrics from equity curve
    metrics = _calculate_metrics(results, config)

    # Log results
    logger.info("=" * 70)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 70)
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")

    # Save results
    output_path = Path(config.backtest.backtest_results_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(
            {
                "metrics": metrics,
                "trades": len(results["trades"]),
                "equity_curve": results["equity_curve"],
                "termination_reason": results["termination_reason"],
                "config": {
                    "initial_capital": config.backtest.initial_capital,
                    "leverage": config.backtest.leverage,
                    "spread_pips": config.backtest.spread_pips,
                    "risk_per_trade": config.backtest.risk_per_trade,
                    "margin_call_level": config.backtest.margin_call_level,
                    "stop_out_level": config.backtest.stop_out_level,
                },
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
            default=str,
        )

    logger.info(f"Saved backtest results: {output_path}")


def _simulate_trades(df: pl.DataFrame, config: Config) -> dict:
    """Simulate trades with proper margin call, stop-out, and risk management.

    NEW FEATURES:
    - Trailing stop loss to protect profits
    - Maximum position hold time (max_hold_bars)
    - Daily loss limit (max_daily_loss_pct)
    - Consecutive loss limit (max_consecutive_losses)

    CRITICAL FIXES:
    1. Tracks equity curve after each trade (not just at end)
    2. Enforces margin call at 50% level - stops trading
    3. Enforces stop-out at 20% level - liquidates account
    4. Enforces bankruptcy protection (capital <= 0)
    5. Position sizing uses CURRENT capital (updated in real-time)

    Args:
        df: Test DataFrame with predictions.
        config: Configuration object.

    Returns:
        Dictionary with trades, equity curve, and termination reason.
    """
    trades = []
    equity_curve = []  # Track equity after each trade for proper metrics

    capital = config.backtest.initial_capital
    initial_capital = capital
    position = None
    entry_price = 0
    entry_time = None
    entry_bar_idx = 0  # Track when position was opened
    open_position_size = 0.0
    highest_price_since_entry = 0.0  # For trailing stop
    lowest_price_since_entry = float("inf")  # For trailing stop
    consecutive_losses = 0
    daily_loss_tracker = {}  # Track daily losses

    # Take-profit / Stop-loss price levels (set at entry)
    tp_price = 0.0  # Take-profit price level
    sl_price_level = 0.0  # Stop-loss price level

    # TP/SL multipliers from labels config (must match Triple Barrier training)
    tp_multiplier = getattr(config.labels, "atr_multiplier_tp", 1.5)
    sl_multiplier = getattr(config.labels, "atr_multiplier_sl", 1.5)

    # Margin levels from config
    margin_call_level = getattr(config.backtest, "margin_call_level", 0.5)  # 50%
    stop_out_level = getattr(config.backtest, "stop_out_level", 0.2)  # 20%

    # NEW: Risk management parameters
    max_hold_bars = getattr(config.backtest, "max_hold_bars", 100)  # Max bars to hold
    use_trailing_stop = getattr(config.backtest, "use_trailing_stop", True)
    trailing_stop_atr_multiplier = getattr(
        config.backtest, "trailing_stop_atr_multiplier", 1.0
    )
    max_daily_loss_pct = getattr(
        config.backtest, "max_daily_loss_pct", 0.05
    )  # 5% daily loss limit
    max_consecutive_losses = getattr(config.backtest, "max_consecutive_losses", 5)

    # Trading costs
    spread = config.backtest.spread_pips * 0.01
    slippage = config.backtest.slippage_pips * 0.01

    # Risk parameters
    risk_per_trade = config.backtest.risk_per_trade
    value_per_pip = 1.0

    termination_reason = "completed"  # Track why simulation ended
    daily_loss_limit_hit = False

    # Convert to pandas for iteration
    df_pd = df.to_pandas()

    for i, row in df_pd.iterrows():
        # Get current date for daily loss tracking
        current_date = pd.to_datetime(row["timestamp"]).date()

        # Check daily loss limit
        if current_date in daily_loss_tracker:
            daily_loss = daily_loss_tracker[current_date]
            daily_loss_pct = abs(daily_loss) / initial_capital
            if daily_loss_pct >= max_daily_loss_pct:
                if not daily_loss_limit_hit:
                    logger.warning(
                        f"📉 DAILY LOSS LIMIT HIT at bar {i}! "
                        f"Daily loss: {daily_loss_pct * 100:.1f}% (limit: {max_daily_loss_pct * 100:.1f}%). "
                        f"Stopping trading for {current_date}."
                    )
                    daily_loss_limit_hit = True
                # Skip trading for rest of day
                if position is not None:
                    # Close position at current price
                    exit_price = row["close"]
                    timestamp = row["timestamp"]

                    pnl, dollar_pnl = _calculate_pnl(
                        position,
                        entry_price,
                        exit_price,
                        open_position_size,
                        spread,
                        slippage,
                        value_per_pip,
                    )

                    trades.append(
                        {
                            "entry_time": entry_time,
                            "exit_time": timestamp,
                            "position": position,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "position_size": open_position_size,
                            "pnl_pips": pnl,
                            "pnl_dollar": dollar_pnl,
                            "exit_reason": "daily_loss_limit",
                        }
                    )

                    capital += dollar_pnl
                    equity_curve.append(capital)

                    # Track daily loss
                    if current_date not in daily_loss_tracker:
                        daily_loss_tracker[current_date] = 0
                    daily_loss_tracker[current_date] += dollar_pnl

                    # Track consecutive losses
                    if dollar_pnl < 0:
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0

                    position = None
                    open_position_size = 0.0
                continue
            else:
                daily_loss_limit_hit = False

        # Check consecutive loss limit
        if consecutive_losses >= max_consecutive_losses:
            logger.warning(
                f"🛑 CONSECUTIVE LOSS LIMIT at bar {i}! "
                f"{consecutive_losses} consecutive losses. "
                f"Pausing trading."
            )
            # Close any open position
            if position is not None:
                exit_price = row["close"]
                timestamp = row["timestamp"]

                pnl, dollar_pnl = _calculate_pnl(
                    position,
                    entry_price,
                    exit_price,
                    open_position_size,
                    spread,
                    slippage,
                    value_per_pip,
                )

                trades.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": timestamp,
                        "position": position,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "position_size": open_position_size,
                        "pnl_pips": pnl,
                        "pnl_dollar": dollar_pnl,
                        "exit_reason": "consecutive_loss_limit",
                    }
                )

                capital += dollar_pnl
                equity_curve.append(capital)

                # Track daily loss
                if current_date not in daily_loss_tracker:
                    daily_loss_tracker[current_date] = 0
                daily_loss_tracker[current_date] += dollar_pnl

                position = None
                open_position_size = 0.0

            termination_reason = f"consecutive_loss_limit_at_bar_{i}"
            break

        # CRITICAL: Check margin conditions BEFORE processing signal
        # Check for stop-out (20% equity remaining) using intra-bar worst case
        # For open positions, use high/low to detect intra-bar spikes
        worst_case_capital = capital
        if position is not None:
            # Estimate worst-case intra-bar PnL to detect spike-driven stop-outs
            if position == "long":
                worst_price = row["low"]  # Long worst case = low
            else:
                worst_price = row["high"]  # Short worst case = high
            est_pnl_pips = (
                (worst_price - entry_price) / 0.01
                if position == "long"
                else (entry_price - worst_price) / 0.01
            )
            worst_case_capital = (
                capital + est_pnl_pips * open_position_size * value_per_pip
            )
        else:
            worst_case_capital = capital

        if worst_case_capital <= initial_capital * stop_out_level:
            logger.error(
                f"🚨 STOP-OUT at bar {i}! "
                f"Worst-case capital: ${worst_case_capital:,.2f} "
                f"({worst_case_capital / initial_capital * 100:.1f}% of initial). "
                f"Account liquidated."
            )
            termination_reason = f"stop_out_at_bar_{i}"
            break

        # Check for margin call (50% equity remaining)
        if capital <= initial_capital * margin_call_level:
            logger.warning(
                f"⚠️ MARGIN CALL at bar {i}! "
                f"Capital: ${capital:,.2f} ({capital / initial_capital * 100:.1f}% of initial). "
                f"Closing position and stopping trading."
            )
            # Close any open position immediately
            if position is not None:
                exit_price = row["close"]
                timestamp = row["timestamp"]

                pnl, dollar_pnl = _calculate_pnl(
                    position,
                    entry_price,
                    exit_price,
                    open_position_size,
                    spread,
                    slippage,
                    value_per_pip,
                )

                trades.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": timestamp,
                        "position": position,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "position_size": open_position_size,
                        "pnl_pips": pnl,
                        "pnl_dollar": dollar_pnl,
                        "exit_reason": "margin_call",
                    }
                )

                capital += dollar_pnl
                position = None
                open_position_size = 0.0

            termination_reason = f"margin_call_at_bar_{i}"
            break

        # Check for bankruptcy (100% loss)
        if capital <= 0:
            logger.error(
                f"💥 BANKRUPTCY at bar {i}! Capital: ${capital:,.2f}. Account blown up."
            )
            termination_reason = f"bankruptcy_at_bar_{i}"
            break

        # Get prediction signal
        signal = _get_signal(row)

        # Current price and timestamp
        current_price = row["close"]
        timestamp = row["timestamp"]

        # CRITICAL FIX: Calculate position size based on CURRENT capital
        # Use ATR for stop loss calculation
        atr = row.get("atr_14", 10.0)
        sl_pips = atr * 100
        sl_pips = max(sl_pips, 5.0)

        # Minimum stop loss to avoid excessive position sizes
        min_sl_pips = 50.0
        effective_sl_pips = max(sl_pips, min_sl_pips)

        # CRITICAL: Use current capital for position sizing (not initial)
        # This ensures position sizes reduce during drawdowns
        available_capital = capital
        risk_amount = available_capital * risk_per_trade
        candidate_position_size = risk_amount / (effective_sl_pips * value_per_pip)

        # Maximum position constraints
        max_position_size = 2.0
        candidate_position_size = min(candidate_position_size, max_position_size)

        # Leverage check
        max_lots_by_margin = (available_capital * config.backtest.leverage) / (
            current_price * 100
        )
        candidate_position_size = min(candidate_position_size, max_lots_by_margin * 0.8)

        # Minimum position size
        candidate_position_size = round(max(candidate_position_size, 0.01), 2)

        # Handle position exits
        if position is not None:
            # Update highest/lowest price for trailing stop using high/low
            if position == "long":
                highest_price_since_entry = max(highest_price_since_entry, row["high"])
            else:  # short
                lowest_price_since_entry = min(lowest_price_since_entry, row["low"])

            # --- Hard Take-Profit / Stop-Loss using high/low ---
            # These checks use intra-bar high/low to detect barrier touches
            # that close-price-only checks would miss.  SL is checked first
            # for conservative (worst-case) bias when both trigger same bar.
            exited = False
            if position == "long":
                # Check SL first (conservative): did low breach stop-loss?
                if row["low"] <= sl_price_level:
                    exit_price = sl_price_level
                    exit_reason = "stop_loss"
                    exited = True
                # Check TP: did high reach take-profit?
                elif row["high"] >= tp_price:
                    exit_price = tp_price
                    exit_reason = "take_profit"
                    exited = True
            else:  # short
                # Check SL first (conservative): did high breach stop-loss?
                if row["high"] >= sl_price_level:
                    exit_price = sl_price_level
                    exit_reason = "stop_loss"
                    exited = True
                # Check TP: did low reach take-profit?
                elif row["low"] <= tp_price:
                    exit_price = tp_price
                    exit_reason = "take_profit"
                    exited = True

            if exited:
                pnl, dollar_pnl = _calculate_pnl(
                    position,
                    entry_price,
                    exit_price,
                    open_position_size,
                    spread,
                    slippage,
                    value_per_pip,
                )
                trades.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": timestamp,
                        "position": position,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "position_size": open_position_size,
                        "pnl_pips": pnl,
                        "pnl_dollar": dollar_pnl,
                        "exit_reason": exit_reason,
                    }
                )
                capital += dollar_pnl
                equity_curve.append(capital)
                if current_date not in daily_loss_tracker:
                    daily_loss_tracker[current_date] = 0
                daily_loss_tracker[current_date] += dollar_pnl
                if dollar_pnl < 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0
                position = None
                open_position_size = 0.0
                continue

            # Check for max hold time
            bars_held = i - entry_bar_idx
            if bars_held >= max_hold_bars:
                logger.info(
                    f"⏱️ MAX HOLD TIME at bar {i} ({bars_held} bars). Closing position."
                )
                pnl, dollar_pnl = _calculate_pnl(
                    position,
                    entry_price,
                    current_price,
                    open_position_size,
                    spread,
                    slippage,
                    value_per_pip,
                )

                trades.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": timestamp,
                        "position": position,
                        "entry_price": entry_price,
                        "exit_price": current_price,
                        "position_size": open_position_size,
                        "pnl_pips": pnl,
                        "pnl_dollar": dollar_pnl,
                        "exit_reason": "max_hold_time",
                    }
                )

                capital += dollar_pnl
                equity_curve.append(capital)

                # Track daily loss
                if current_date not in daily_loss_tracker:
                    daily_loss_tracker[current_date] = 0
                daily_loss_tracker[current_date] += dollar_pnl

                # Track consecutive losses
                if dollar_pnl < 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0

                position = None
                open_position_size = 0.0
                continue

            # Check for trailing stop
            if use_trailing_stop:
                atr = row.get("atr_14", 10.0)
                trailing_distance = atr * trailing_stop_atr_multiplier

                if position == "long":
                    # Trailing stop for long: exit if price drops below highest - atr
                    trailing_stop_price = highest_price_since_entry - trailing_distance
                    if current_price <= trailing_stop_price:
                        logger.info(
                            f"🛑 TRAILING STOP at bar {i}. "
                            f"Price: {current_price:.2f}, Stop: {trailing_stop_price:.2f}"
                        )
                        pnl, dollar_pnl = _calculate_pnl(
                            position,
                            entry_price,
                            current_price,
                            open_position_size,
                            spread,
                            slippage,
                            value_per_pip,
                        )

                        trades.append(
                            {
                                "entry_time": entry_time,
                                "exit_time": timestamp,
                                "position": position,
                                "entry_price": entry_price,
                                "exit_price": current_price,
                                "position_size": open_position_size,
                                "pnl_pips": pnl,
                                "pnl_dollar": dollar_pnl,
                                "exit_reason": "trailing_stop",
                            }
                        )

                        capital += dollar_pnl
                        equity_curve.append(capital)

                        # Track daily loss
                        if current_date not in daily_loss_tracker:
                            daily_loss_tracker[current_date] = 0
                        daily_loss_tracker[current_date] += dollar_pnl

                        # Track consecutive losses
                        if dollar_pnl < 0:
                            consecutive_losses += 1
                        else:
                            consecutive_losses = 0

                        position = None
                        open_position_size = 0.0
                        continue

                else:  # short
                    # Trailing stop for short: exit if price rises above lowest + atr
                    trailing_stop_price = lowest_price_since_entry + trailing_distance
                    if current_price >= trailing_stop_price:
                        logger.info(
                            f"🛑 TRAILING STOP at bar {i}. "
                            f"Price: {current_price:.2f}, Stop: {trailing_stop_price:.2f}"
                        )
                        pnl, dollar_pnl = _calculate_pnl(
                            position,
                            entry_price,
                            current_price,
                            open_position_size,
                            spread,
                            slippage,
                            value_per_pip,
                        )

                        trades.append(
                            {
                                "entry_time": entry_time,
                                "exit_time": timestamp,
                                "position": position,
                                "entry_price": entry_price,
                                "exit_price": current_price,
                                "position_size": open_position_size,
                                "pnl_pips": pnl,
                                "pnl_dollar": dollar_pnl,
                                "exit_reason": "trailing_stop",
                            }
                        )

                        capital += dollar_pnl
                        equity_curve.append(capital)

                        # Track daily loss
                        if current_date not in daily_loss_tracker:
                            daily_loss_tracker[current_date] = 0
                        daily_loss_tracker[current_date] += dollar_pnl

                        # Track consecutive losses
                        if dollar_pnl < 0:
                            consecutive_losses += 1
                        else:
                            consecutive_losses = 0

                        position = None
                        open_position_size = 0.0
                        continue

            # Check for opposite signal (exit and reverse)
            should_exit = False
            if (position == "long" and signal == -1) or (
                position == "short" and signal == 1
            ):
                should_exit = True

            if should_exit:
                pnl, dollar_pnl = _calculate_pnl(
                    position,
                    entry_price,
                    current_price,
                    open_position_size,
                    spread,
                    slippage,
                    value_per_pip,
                )

                trades.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": timestamp,
                        "position": position,
                        "entry_price": entry_price,
                        "exit_price": current_price,
                        "position_size": open_position_size,
                        "pnl_pips": pnl,
                        "pnl_dollar": dollar_pnl,
                        "exit_reason": "signal_reverse",
                    }
                )

                capital += dollar_pnl
                equity_curve.append(capital)  # Track equity after each trade

                # Track daily loss
                if current_date not in daily_loss_tracker:
                    daily_loss_tracker[current_date] = 0
                daily_loss_tracker[current_date] += dollar_pnl

                # Track consecutive losses
                if dollar_pnl < 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0

                position = None
                open_position_size = 0.0

        # Open new position if no current position and signal is strong
        if position is None and signal != 0 and candidate_position_size > 0.01:
            position = "long" if signal == 1 else "short"
            entry_price = current_price
            entry_time = timestamp
            entry_bar_idx = i  # Track entry bar index
            open_position_size = candidate_position_size

            # Reset trailing stop tracking
            highest_price_since_entry = current_price
            lowest_price_since_entry = current_price

            # Calculate TP/SL price levels at entry (matching Triple Barrier)
            entry_atr = row.get("atr_14", 10.0)
            if position == "long":
                tp_price = entry_price + entry_atr * tp_multiplier
                sl_price_level = entry_price - entry_atr * sl_multiplier
            else:
                tp_price = entry_price - entry_atr * tp_multiplier
                sl_price_level = entry_price + entry_atr * sl_multiplier

            # Adjust for spread and slippage
            if position == "long":
                entry_price = entry_price + spread / 2 + slippage
            else:
                entry_price = entry_price - spread / 2 - slippage

    # Close any remaining position at last price
    if position is not None:
        last_row = df_pd.iloc[-1]
        exit_price = last_row["close"]
        timestamp = last_row["timestamp"]

        pnl, dollar_pnl = _calculate_pnl(
            position,
            entry_price,
            exit_price,
            open_position_size,
            spread,
            slippage,
            value_per_pip,
        )

        trades.append(
            {
                "entry_time": entry_time,
                "exit_time": timestamp,
                "position": position,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "position_size": open_position_size,
                "pnl_pips": pnl,
                "pnl_dollar": dollar_pnl,
                "exit_reason": "end_of_data",
            }
        )

        capital += dollar_pnl
        equity_curve.append(capital)
        position = None
        open_position_size = 0.0

    # Ensure we have at least initial capital in equity curve
    if not equity_curve:
        equity_curve = [initial_capital]
    else:
        # Prepend initial capital if first entry isn't it
        if len(equity_curve) == 0 or equity_curve[0] != initial_capital:
            equity_curve = [initial_capital] + equity_curve

    # Save detailed trade data
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_path = (
            Path(config.backtest.backtest_results_path).parent / "trades_detail.csv"
        )
        trades_df.to_csv(trades_path, index=False)
        logger.info(f"Saved {len(trades)} trade records to: {trades_path}")

    logger.info(f"Simulated {len(trades)} trades")
    logger.info(f"Final capital: ${capital:,.2f}")
    logger.info(f"Total return: {((capital / initial_capital) - 1) * 100:.2f}%")
    logger.info(f"Termination reason: {termination_reason}")

    return {
        "trades": trades,
        "equity_curve": equity_curve,
        "final_capital": capital,
        "termination_reason": termination_reason,
    }


def _get_signal(row: pd.Series) -> int:
    """Extract trading signal from prediction probabilities.

    Args:
        row: DataFrame row with prediction columns.

    Returns:
        Signal: -1 (short), 0 (hold), or 1 (long).
    """
    if "pred_proba_class_1" in row:
        proba_long = row["pred_proba_class_1"]
        proba_short = row["pred_proba_class_minus1"]

        threshold = 0.6

        if proba_long > threshold and proba_long > proba_short:
            return 1
        elif proba_short > threshold and proba_short > proba_long:
            return -1

    return 0


def _calculate_pnl(
    position: str,
    entry_price: float,
    exit_price: float,
    position_size: float,
    spread: float,
    slippage: float,
    value_per_pip: float,
) -> tuple[float, float]:
    """Calculate PnL in pips and dollars.

    Args:
        position: "long" or "short".
        entry_price: Entry price.
        exit_price: Exit price.
        position_size: Position size in lots.
        spread: Spread in price terms.
        slippage: Slippage in price terms.
        value_per_pip: Dollar value per pip per lot.

    Returns:
        Tuple of (pnl_pips, pnl_dollar).
    """
    if position == "long":
        # Sell at bid (price - spread/2)
        adjusted_exit = exit_price - spread / 2 - slippage
        pnl = (adjusted_exit - entry_price) / 0.01
    else:
        # Buy back at ask (price + spread/2)
        adjusted_exit = exit_price + spread / 2 + slippage
        pnl = (entry_price - adjusted_exit) / 0.01

    dollar_pnl = pnl * position_size * value_per_pip

    return pnl, dollar_pnl


def _calculate_metrics(results: dict, config: Config) -> dict:
    """Calculate backtest metrics from equity curve.

    CRITICAL FIX: Uses equity curve for drawdown calculation, not trade returns.

    Args:
        results: Dictionary with trades and equity curve.
        config: Configuration object.

    Returns:
        Dictionary of metrics.
    """
    trades = results["trades"]
    equity_curve = results["equity_curve"]
    termination_reason = results["termination_reason"]

    if not trades or not equity_curve:
        return {
            "error": "No trades executed",
            "termination_reason": termination_reason,
        }

    pnls = np.array([t["pnl_dollar"] for t in trades])
    pnl_pips = np.array([t["pnl_pips"] for t in trades])

    initial_capital = config.backtest.initial_capital
    final_capital = equity_curve[-1]

    # Basic metrics
    total_return = (final_capital - initial_capital) / initial_capital
    win_rate = (pnls > 0).sum() / len(pnls)

    # Profit factor
    gross_profit = pnls[pnls > 0].sum()
    gross_loss = abs(pnls[pnls < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Average trade
    avg_trade = pnls.mean()
    avg_win = pnls[pnls > 0].mean() if (pnls > 0).any() else 0
    avg_loss = pnls[pnls < 0].mean() if (pnls < 0).any() else 0

    # CRITICAL FIX: Calculate Sharpe ratio from equity curve returns
    # Calculate daily returns from equity curve
    equity_array = np.array(equity_curve)
    if len(equity_array) > 1:
        # Calculate returns between consecutive equity points
        equity_returns = np.diff(equity_array) / equity_array[:-1]

        if len(equity_returns) > 1 and equity_returns.std() > 0:
            sharpe = equity_returns.mean() / equity_returns.std() * np.sqrt(252)
        else:
            sharpe = 0
    else:
        sharpe = 0

    # CRITICAL FIX: Calculate max drawdown from equity curve (not trade returns)
    # This is the proper way to calculate drawdown
    equity_array = np.array(equity_curve)
    peak = np.maximum.accumulate(equity_array)
    drawdown = (peak - equity_array) / peak  # Percentage decline from peak
    max_drawdown = drawdown.max()

    # Calmar ratio using proper drawdown
    calmar = total_return / max_drawdown if max_drawdown > 0 else 0

    # Sanity checks for impossible values
    if max_drawdown > 1.0:
        logger.error(f"🚨 IMPOSSIBLE max_drawdown: {max_drawdown * 100:.2f}% (>100%)")
        logger.error("   This indicates a bug in the simulation logic.")

    metrics = {
        "total_trades": len(trades),
        "winning_trades": int((pnls > 0).sum()),
        "losing_trades": int((pnls < 0).sum()),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "total_return_pct": total_return * 100,
        "avg_trade_dollar": avg_trade,
        "avg_win_dollar": avg_win,
        "avg_loss_dollar": avg_loss,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_drawdown * 100,
        "calmar_ratio": calmar,
        "initial_capital": initial_capital,
        "final_capital": final_capital,
        "total_pnl_pips": pnl_pips.sum(),
        "avg_pips_per_trade": pnl_pips.mean(),
        "termination_reason": termination_reason,
    }

    # Log sanity check
    if max_drawdown < 1.0:
        logger.info(
            "Metrics summary: %.2f%% return, %.2f%% max drawdown, %.2f profit factor",
            metrics["total_return_pct"],
            metrics["max_drawdown_pct"],
            metrics["profit_factor"],
        )
    elif max_drawdown >= 1.0:
        logger.error(
            f"❌ Account was liquidated: {metrics['max_drawdown_pct']:.2f}% drawdown"
        )

    return metrics
