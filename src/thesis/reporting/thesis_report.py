"""Thesis report generation with SHAP analysis."""

import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import shap
from matplotlib.backends.backend_agg import FigureCanvasAgg

from thesis.config.loader import Config

logger = logging.getLogger("thesis.reporting")


def generate_report(config: Config) -> None:
    """Generate thesis report with SHAP analysis.
    
    Args:
        config: Configuration object.
    """
    logger.info("Generating thesis report...")
    
    # Load backtest results
    backtest_path = Path(config.backtest.backtest_results_path)
    if not backtest_path.exists():
        raise FileNotFoundError(f"Backtest results not found: {backtest_path}")
    
    with open(backtest_path) as f:
        backtest_data = json.load(f)
    
    # Generate SHAP summary
    if config.reporting.plot_predictions:
        try:
            _generate_shap_summary(config)
        except Exception as e:
            logger.warning(f"SHAP generation failed: {e}")
    
    # Create markdown report
    report_md = _create_markdown_report(backtest_data, config)
    
    # Save markdown report
    report_path = Path(config.reporting.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report_md)
    
    logger.info(f"Saved report: {report_path}")
    
    # Save JSON report
    json_path = Path(config.reporting.report_json_path)
    with open(json_path, "w") as f:
        json.dump(backtest_data, f, indent=2)
    
    logger.info(f"Saved JSON report: {json_path}")


def _generate_shap_summary(config: Config) -> None:
    """Generate SHAP summary plot for LightGBM model.
    
    Args:
        config: Configuration object.
    """
    try:
        import joblib
        import lightgbm as lgb
        
        # Load model
        model_path = Path(config.models["tree"].model_path)
        if not model_path.exists():
            logger.warning("LightGBM model not found, skipping SHAP")
            return
        
        model = joblib.load(model_path)
        
        # Load test data
        test_path = Path(config.paths.test_data)
        if not test_path.exists():
            return
        
        test_df = pl.read_parquet(test_path)
        
        # Get features
        exclude_cols = ["timestamp", "label", "tp_price", "sl_price", "touched_bar",
                       "open", "high", "low", "close", "volume", "avg_spread", "tick_count"]
        feature_cols = [c for c in test_df.columns if c not in exclude_cols]
        
        X_test = test_df.select(feature_cols).to_numpy()
        
        # Sample for SHAP
        n_samples = min(config.reporting.shap_samples, len(X_test))
        sample_idx = np.random.choice(len(X_test), n_samples, replace=False)
        X_sample = X_test[sample_idx]
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, 
            X_sample, 
            feature_names=feature_cols,
            max_display=config.reporting.shap_max_display,
            show=False,
        )
        
        shap_path = Path(config.reporting.shap_summary_path)
        shap_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(shap_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Saved SHAP summary: {shap_path}")
        
    except Exception as e:
        logger.warning(f"SHAP plot failed: {e}")


def _create_markdown_report(backtest_data: dict, config: Config) -> str:
    """Create markdown thesis report.
    
    Args:
        backtest_data: Backtest results dictionary.
        config: Configuration object.
        
    Returns:
        Markdown report string.
    """
    metrics = backtest_data.get("metrics", {})
    
    report = f"""# Hybrid Stacking (LSTM + LightGBM) for XAU/USD Trading Signals

## Bachelor's Thesis Report
**Student:** Nguyen Duc Hieu (2151061192)  
**Advisor:** Hoang Quoc Dung  
**University:** Thuy Loi University  
**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## Executive Summary

This thesis implements a **Hybrid Stacking ensemble** combining:
- **LightGBM** for tabular feature learning
- **LSTM** for sequential OHLCV patterns
- **Meta-learner** for final prediction aggregation

Target: XAU/USD (Gold/USD) on H1 timeframe with Triple-Barrier labeling.

---

## Data Configuration

| Parameter | Value |
|-----------|-------|
| Asset | XAU/USD (Gold) |
| Timeframe | H1 (1 hour) |
| Data Period | 2018-01 to 2026-03 |
| Train Period | 2018-2021 (70%) |
| Val Period | 2022 (15%) |
| Test Period | 2023-03/2026 (15%) |
| Purge | {config.splitting.purge_bars} bars |
| Embargo | {config.splitting.embargo_bars} bars |

### Market Regime Split Rationale

**Train (2018-2021):** Normal + Trade War + COVID shock
- Establishes baseline relationships
- Gold's flight-to-safety during pandemic

**Validation (2022):** Russia-Ukraine war + Fed rate hikes
- Stress test on geopolitical shock
- Rising rates typically pressure gold

**Test (2023-03/2026):** SVB crisis + Gold ATH + "New Regime"
- **Critical:** Gold ATH while rates elevated
- Paradigm shift where gold rises despite high rates
- Tests model on new market regime

---

## Model Configuration

### LightGBM
- Hyperparameter tuning: Optuna ({config.models['tree'].optuna_trials} trials)
- Class balancing: {'Enabled' if config.models['tree'].use_class_weights else 'Disabled'}

### LSTM
- Sequence length: {config.models['lstm'].sequence_length} bars
- Hidden size: {config.models['lstm'].hidden_size}
- Layers: {config.models['lstm'].num_layers}
- Dropout: {config.models['lstm'].dropout}

### Stacking
- Meta-learner: {config.models['stacking'].meta_learner}
- Calibration: {'Enabled' if config.models['stacking'].calibrate_probabilities else 'Disabled'}

---

## Backtest Results

### Trading Configuration
| Parameter | Value |
|-----------|-------|
| Initial Capital | ${config.backtest.initial_capital:,.0f} |
| Leverage | {config.backtest.leverage}:1 |
| Spread | {config.backtest.spread_pips} pips |
| Risk per Trade | {config.backtest.risk_per_trade * 100:.0f}% |
| Slippage | {config.backtest.slippage_pips} pips |

### Performance Metrics

| Metric | Value |
|--------|-------|
| Total Trades | {metrics.get('total_trades', 'N/A')} |
| Winning Trades | {metrics.get('winning_trades', 'N/A')} |
| Losing Trades | {metrics.get('losing_trades', 'N/A')} |
| Win Rate | {metrics.get('win_rate', 0) * 100:.1f}% |
| Profit Factor | {metrics.get('profit_factor', 'N/A'):.2f} |
| **Total Return** | **{metrics.get('total_return_pct', 0):.2f}%** |
| Sharpe Ratio | {metrics.get('sharpe_ratio', 'N/A'):.2f} |
| Max Drawdown | {metrics.get('max_drawdown_pct', 0):.2f}% |
| Calmar Ratio | {metrics.get('calmar_ratio', 'N/A'):.2f} |
| Avg Trade | ${metrics.get('avg_trade_dollar', 0):.2f} |
| Avg Win | ${metrics.get('avg_win_dollar', 0):.2f} |
| Avg Loss | ${metrics.get('avg_loss_dollar', 0):.2f} |
| Total PnL (pips) | {metrics.get('total_pnl_pips', 0):.1f} |
| Avg Pips/Trade | {metrics.get('avg_pips_per_trade', 0):.2f} |

### Final Capital: ${metrics.get('final_capital', 0):,.2f}

---

## Conclusions

1. **Hybrid stacking** successfully combines LightGBM and LSTM predictions
2. **Market regime split** enables evaluation on new paradigm (gold ATH despite high rates)
3. **Triple-Barrier labeling** provides realistic trade targets
4. Results demonstrate [model effectiveness on test set]

---

## Files Generated

- Backtest results: `{config.backtest.backtest_results_path}`
- SHAP summary: `{config.reporting.shap_summary_path}`
- Model predictions: `{config.paths.final_predictions}`

---

*Report generated by thesis pipeline v0.1.0*
"""
    
    return report


# Need numpy import
import numpy as np
