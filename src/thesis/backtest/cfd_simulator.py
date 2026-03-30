"""CFD backtest simulator with realistic trading costs."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import polars as pl

from thesis.config.loader import Config

logger = logging.getLogger("thesis.backtest")


def run_backtest(config: Config) -> None:
    """Run CFD backtest simulation.
    
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
    # Need to align with test data properly
    # For now, use predictions directly if lengths match
    if len(preds_df) != len(test_df):
        logger.warning(f"Length mismatch: test={len(test_df)}, preds={len(preds_df)}")
        # Try to align by timestamp
        test_df = test_df.join(preds_df, on="timestamp", how="inner")
    else:
        # Add prediction columns
        for col in ["pred_proba_class_minus1", "pred_proba_class_0", "pred_proba_class_1"]:
            if col in preds_df.columns:
                test_df = test_df.with_columns(preds_df[col])
    
    logger.info(f"Backtest data: {len(test_df)} bars")
    
    # Run simulation
    results = _simulate_trades(test_df, config)
    
    # Calculate metrics
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
        json.dump({
            "metrics": metrics,
            "trades": len(results),
            "config": {
                "initial_capital": config.backtest.initial_capital,
                "leverage": config.backtest.leverage,
                "spread_pips": config.backtest.spread_pips,
                "risk_per_trade": config.backtest.risk_per_trade,
            },
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2, default=str)
    
    logger.info(f"Saved backtest results: {output_path}")


def _simulate_trades(df: pl.DataFrame, config: Config) -> List[Dict]:
    """Simulate trades based on predictions with proper risk-based position sizing.
    
    Args:
        df: Test DataFrame with predictions.
        config: Configuration object.
        
    Returns:
        List of trade dictionaries.
    """
    trades = []
    capital = config.backtest.initial_capital
    position = None  # Current position: None, 'long', or 'short'
    entry_price = 0
    entry_time = None
    position_size = 0  # Position size in lots
    
    # Trading costs
    spread = config.backtest.spread_pips * 0.01  # Convert pips to price
    slippage = config.backtest.slippage_pips * 0.01
    
    # Risk parameters
    risk_per_trade = config.backtest.risk_per_trade  # e.g., 0.01 = 1%
    value_per_pip = 1.0  # XAU/USD: 1 lot = $1 per pip
    
    # Convert to pandas for easier iteration
    df_pd = df.to_pandas()
    
    for i, row in df_pd.iterrows():
        # Get prediction
        if "pred_proba_class_1" in row:
            # Use probabilities
            proba_long = row["pred_proba_class_1"]
            proba_short = row["pred_proba_class_minus1"]
            # proba_hold extracted but not used in current threshold logic
            
            # Threshold for trading (need high confidence)
            threshold = 0.6
            
            signal = 0  # Hold
            if proba_long > threshold and proba_long > proba_short:
                signal = 1  # Long
            elif proba_short > threshold and proba_short > proba_long:
                signal = -1  # Short
        else:
            signal = 0
        
        # Current price and ATR for stop loss calculation
        current_price = row["close"]
        timestamp = row["timestamp"]
        
        # Get ATR for position sizing (need atr_14 column from features)
        # Default to 10 pips if not available
        atr = row.get("atr_14", 10.0)  # ATR in price terms
        sl_pips = atr * 100  # Convert price to pips (multiply by 100 for XAU/USD)
        sl_pips = max(sl_pips, 5.0)  # Minimum 5 pip stop loss
        
        # Calculate position size based on risk
        # Use minimum 50 pip stop to avoid excessive position sizes on tight stops
        min_sl_pips = 50.0  # Minimum stop loss in pips
        effective_sl_pips = max(sl_pips, min_sl_pips)
        
        # Position Size (lots) = (Capital × Risk%) / (SL in pips × $ per pip per lot)
        risk_amount = capital * risk_per_trade
        position_size = risk_amount / (effective_sl_pips * value_per_pip)
        
        # Maximum position size constraints
        # Conservative: max 2.0 lots per trade regardless of account size
        max_position_size = 2.0
        position_size = min(position_size, max_position_size)
        
        # Leverage check: ensure we don't exceed 1:100 leverage
        # Margin required = position_size × 100 oz × current_price / leverage
        # Available margin = capital
        max_lots_by_margin = (capital * config.backtest.leverage) / (current_price * 100)
        position_size = min(position_size, max_lots_by_margin * 0.8)  # Use 80% of max as safety
        
        # Round to 2 decimal places, minimum 0.01 lots
        position_size = round(max(position_size, 0.01), 2)
        
        # Check if position should be closed
        if position is not None:
            # Check for opposite signal (exit and reverse)
            if (position == "long" and signal == -1) or (position == "short" and signal == 1):
                # Close position
                exit_price = current_price
                
                if position == "long":
                    # Sell at bid (price - spread/2)
                    exit_price = exit_price - spread / 2 - slippage
                    pnl = (exit_price - entry_price) / 0.01  # Pips
                else:
                    # Buy back at ask (price + spread/2)
                    exit_price = exit_price + spread / 2 + slippage
                    pnl = (entry_price - exit_price) / 0.01  # Pips
                
                # Calculate dollar PnL with proper position sizing
                dollar_pnl = pnl * position_size * value_per_pip
                
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": timestamp,
                    "position": position,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "position_size": position_size,
                    "pnl_pips": pnl,
                    "pnl_dollar": dollar_pnl,
                })
                
                capital += dollar_pnl
                position = None
        
        # Open new position if no current position and signal is strong
        if position is None and signal != 0 and position_size > 0.01:  # Minimum 0.01 lots
            position = "long" if signal == 1 else "short"
            entry_price = current_price
            entry_time = timestamp
            
            # Adjust for spread
            if position == "long":
                entry_price = entry_price + spread / 2 + slippage  # Buy at ask
            else:
                entry_price = entry_price - spread / 2 - slippage  # Sell at bid
    
    # Close any remaining position at last price
    if position is not None:
        last_row = df_pd.iloc[-1]
        exit_price = last_row["close"]
        timestamp = last_row["timestamp"]
        
        if position == "long":
            exit_price = exit_price - spread / 2 - slippage
            pnl = (exit_price - entry_price) / 0.01
        else:
            exit_price = exit_price + spread / 2 + slippage
            pnl = (entry_price - exit_price) / 0.01
        
        dollar_pnl = pnl * position_size * value_per_pip
        
        trades.append({
            "entry_time": entry_time,
            "exit_time": timestamp,
            "position": position,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "position_size": position_size,
            "pnl_pips": pnl,
            "pnl_dollar": dollar_pnl,
        })
        
        capital += dollar_pnl
    
    # Save detailed trade data for analysis
    trades_df = pd.DataFrame(trades)
    trades_path = Path(config.backtest.backtest_results_path).parent / 'trades_detail.csv'
    trades_df.to_csv(trades_path, index=False)
    logger.info(f"Saved {len(trades)} trade records to: {trades_path}")
    
    logger.info(f"Simulated {len(trades)} trades")
    logger.info(f"Final capital: ${capital:,.2f}")
    logger.info(f"Total return: {((capital / config.backtest.initial_capital) - 1) * 100:.2f}%")
    
    return trades


def _calculate_metrics(trades: List[Dict], config: Config) -> Dict:
    """Calculate backtest metrics.
    
    Args:
        trades: List of trade dictionaries.
        config: Configuration object.
        
    Returns:
        Dictionary of metrics.
    """
    if not trades:
        return {"error": "No trades executed"}
    
    pnls = np.array([t["pnl_dollar"] for t in trades])
    pnl_pips = np.array([t["pnl_pips"] for t in trades])
    
    initial_capital = config.backtest.initial_capital
    final_capital = initial_capital + pnls.sum()
    
    # Returns
    returns = pnls / initial_capital
    
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
    
    # Sharpe ratio (simplified, assuming risk-free rate = 0)
    if len(returns) > 1 and returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(252)  # Annualized
    else:
        sharpe = 0
    
    # Max drawdown (simplified)
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    max_drawdown = drawdown.max()
    
    # Calmar ratio
    calmar = total_return / max_drawdown if max_drawdown > 0 else 0
    
    return {
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
    }
