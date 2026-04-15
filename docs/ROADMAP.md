# Roadmap (Todo)

> What is done and what is still pending.

---

## Completed

### Core Pipeline

- [x] **Data preparation** — Raw tick data to OHLCV bars aggregation
- [x] **Feature engineering** — 11 technical indicators (RSI, ATR, MACD, ATR ratio, price distance, pivot position, ATR percentile, 4 session dummies)
- [x] **Label generation** — Triple Barrier method with ATR-based take-profit and stop-loss
- [x] **Data splitting** — Train/validation/test with purge and embargo (anti-leakage)
- [x] **Correlation filtering** — Automatic removal of highly correlated features (>0.90) computed on train set only

### Models

- [x] **GRU feature extractor** — 2-layer GRU, 64 hidden units, 24-bar sequences
- [x] **LightGBM classifier** — Tuned hyperparameters with class weight balancing
- [x] **Hybrid training pipeline** — GRU hidden states + static features → LightGBM
- [x] **SHAP feature importance** — Interpretability analysis of the hybrid model

### Evaluation

- [x] **CFD backtest** — via `backtesting.py` with native margin, spread, commission, ATR stop-loss
- [x] **Fixed lot position sizing** — Prevents runaway sizing with leverage
- [x] **Comprehensive metrics** — 20+ trading metrics (Sharpe, Sortino, Calmar, SQN, drawdown, etc.)
- [x] **Interactive Bokeh chart** — HTML chart with equity, drawdown, and trade markers

### Visualization

- [x] **Data charts** — Candlestick, label distribution, feature correlation, feature distributions
- [x] **Model charts** — Confusion matrix, confidence distribution, feature importance
- [x] **Backtest charts** — Equity curve, drawdown, trade analysis
- [x] **Thesis report** — Auto-generated markdown report with all metrics and charts
- [x] **Interactive Streamlit dashboard** — ECharts-based visualization on :8501 (`src/thesis/dashboard/app.py`)

### Infrastructure

- [x] **Config management** — Single TOML config file with dataclasses
- [x] **Session-based output** — Timestamped results folder for each run
- [x] **CLI entry point** — `main.py` with `--force` and `--ablation` flags
- [x] **Pixi package management** — Reproducible environment with pixi.toml
- [x] **Test suite** — Unit and integration tests with 70% coverage minimum
- [x] **Code quality** — Ruff linting and formatting
- [x] **CI/CD workflows** — GitHub Actions for testing and releases
- [x] **Git conventions** — Conventional commits, branch strategy, PR templates
- [x] **Optuna hyperparameter search** — Integrated auto-tuning with `optuna_trials` and `optuna_timeout`

---

## Pending

### Model Improvements

- [ ] **Walk-forward validation** — Implement rolling window cross-validation for more robust evaluation
- [ ] **Ensemble methods** — Combine multiple trained models for more stable predictions
- [ ] **Transformer encoder** — Experiment with self-attention as an alternative to GRU
- [ ] **Multi-timeframe features** — Add features from 4H and daily timeframes
- [ ] **Sentiment features** — Incorporate news sentiment or macro indicators
- [ ] **Volume profile analysis** — Add volume-based features (VWAP, volume clusters)

### Backtest Enhancements

- [ ] **Swap/rollover costs** — Model overnight holding costs in the CFD simulator
- [ ] **Slippage model** — Add realistic slippage based on volatility and liquidity
- [ ] **Multi-asset support** — Extend to other currency pairs (EUR/USD, GBP/USD)
- [ ] **Monte Carlo simulation** — Randomize trade order to test robustness of metrics
- [ ] **Parameter sensitivity analysis** — Test how small changes in spread, leverage, etc. affect results

### Operational

- [ ] **Real-time inference** — Deploy model for live signal generation
- [ ] **Model versioning** — Track and compare model versions across experiments
- [ ] **Data pipeline monitoring** — Alert when data quality drops or drifts

### Documentation & Research

- [ ] **API documentation** — Auto-generated docs from docstrings
- [ ] **Experiment log** — Track all experiments with parameters and results
- [ ] **Literature comparison** — Compare results with published research benchmarks
- [ ] **Statistical significance tests** — Add Diebold-Mariano or similar tests for model comparison

---

## Progress Summary

| Category | Completed | Pending | Total |
|----------|-----------|---------|-------|
| Core Pipeline | 5 | 0 | 5 |
| Models | 4 | 6 | 10 |
| Evaluation | 4 | 5 | 9 |
| Visualization | 5 | 0 | 5 |
| Infrastructure | 9 | 3 | 12 |
| Documentation | 0 | 4 | 4 |
| **Total** | **27** | **18** | **45** |

> **Overall: 60% complete** — The core research pipeline is fully functional. Pending items are enhancements and production features.
