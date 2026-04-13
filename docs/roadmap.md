# Roadmap (Todo)

> What has been completed and what is still planned. Updated as the project evolves.

---

## Legend

| Icon | Status |
|------|--------|
| ✅ | Completed |
| 🔧 | In Progress / Needs Refinement |
| ⬜ | Planned / Not Started |

---

## Completed

### Data Layer
- ✅ Raw tick data download from Dukascopy (`data_download.py`)
- ✅ Tick-to-OHLCV aggregation (1-hour candles)
- ✅ OHLCV data quality validation (gaps, spikes, duplicates)
- ✅ Timezone handling (America/New_York market time)
- ✅ Weekend data filtering
- ✅ Processed data caching as Parquet files

### Feature Engineering
- ✅ Technical indicators: EMA (34, 89), RSI (14), MACD (12,26,9), ATR (14)
- ✅ Pivot points (daily P, R1, S1)
- ✅ Session encoding (Asia 00-08, London 08-17, NY PM 17-21)
- ✅ Lag features (periods: 1, 2, 3, 5, 10)
- ✅ Spread features
- ✅ Microstructure features (candlestick patterns, volume delta, body-to-wick ratio)
- ✅ High-correlation feature removal (threshold: 0.90)
- ✅ Feature list persistence

### Label Generation
- ✅ Triple-Barrier labeling (TP = 1.5x ATR, SL = 1.5x ATR, horizon = 20 bars)
- ✅ 3-class labeling: Long (+1), Hold (0), Short (-1)
- ✅ Lookahead bias prevention (sequential bar-by-bar checking)
- ✅ Simultaneous TP/SL handling (conservative → Hold)
- ✅ Label distribution logging

### Data Splitting
- ✅ Time-based train/val/test split (2018-2022 / 2023 / 2024-2026)
- ✅ Purge bars (25) to prevent overlap leakage
- ✅ Embargo bars (10) for additional safety
- ✅ Class distribution logging per split
- ✅ Walk-forward cross-validation support

### Models
- ✅ LightGBM classifier with class weights
- ✅ Optuna hyperparameter optimization (up to 100 trials)
- ✅ Walk-forward cross-validation for LightGBM
- ✅ Out-of-fold (OOF) prediction generation
- ✅ PyTorch LSTM classifier (2-layer, 128 hidden units)
- ✅ LSTM sequence generation (120-bar windows)
- ✅ LSTM early stopping with patience
- ✅ LSTM normalization with saved statistics

### Stacking (Meta-Learning)
- ✅ Logistic regression meta-learner
- ✅ Time-aware cross-validation for stacking
- ✅ Confidence threshold filtering (0.6)
- ✅ Probability calibration (isotonic regression)
- ✅ Multiple meta-learner support (logistic, ridge, lasso, elastic_net, lightgbm)
- ✅ Test set prediction generation

### Backtesting
- ✅ CFD trading simulator with realistic costs
- ✅ Spread (2 pips) and slippage (1 pip) modeling
- ✅ Leverage (50:1) and margin calculation
- ✅ Fixed-fractional position sizing (1% risk per trade)
- ✅ Trailing stop with ATR multiplier
- ✅ Risk management: margin call, stop-out, max daily loss
- ✅ Consecutive loss limit
- ✅ Trade log export (CSV)
- ✅ Multiple metrics calculation (Sharpe, Sortino, Calmar, etc.)

### Reporting
- ✅ Markdown thesis report generation
- ✅ SHAP feature importance analysis and plot
- ✅ Model disagreement analysis
- ✅ Confidence histogram
- ✅ JSON metrics export
- ✅ Equity curve plotting
- ✅ Drawdown visualization
- ✅ Trade distribution charts

### Infrastructure
- ✅ TOML-based configuration (`config.toml`)
- ✅ Session-based output management
- ✅ Environment variable overrides (`THESIS_*` prefix)
- ✅ Pipeline orchestration (9 stages with enable/disable)
- ✅ CLI entry point (`main.py`)
- ✅ Jupyter notebook (`main.ipynb`)
- ✅ TUI dashboard (`scripts/dashboard.py`)
- ✅ `results/latest` symlink

### Validation & Testing
- ✅ Label validation (lookahead bias detection)
- ✅ Math consistency validation (Kelly criterion, expected return)
- ✅ Pipeline integrity checker
- ✅ Unit test suite (20+ test files)
- ✅ Integration test suite
- ✅ Test coverage requirement (70%+)
- ✅ Test fixtures and conftest setup
- ✅ Regression tests for edge cases

### Code Quality
- ✅ Ruff linting and formatting
- ✅ Type annotations on public functions
- ✅ pathlib usage (no os.path)
- ✅ Structured logging (no print statements)
- ✅ Conventional commit messages

---

## Planned / In Progress

### Model Improvements
- 🔧 Additional meta-learner tuning
- 🔧 Ensemble weight optimization
- ⬜ Transformer-based model as third base learner
- ⬜ Attention mechanism for LSTM
- ⬜ Hyperparameter sensitivity analysis

### Feature Enhancements
- ⬜ Order flow features (if tick-level data permits)
- ⬜ Macro-economic indicators integration (interest rates, CPI)
- ⬜ Sentiment analysis features (news, social media)
- ⬜ Cross-asset features (DXY, S&P 500 correlation)
- ⬜ Volatility regime detection features

### Backtesting & Evaluation
- ⬜ Monte Carlo simulation for robustness testing
- ⬜ Multi-timeframe backtesting (M15, H4, D1)
- ⬜ Walk-forward analysis (rolling out-of-sample)
- ⬜ Parameter stability analysis across market regimes
- ⬜ Benchmark comparison (buy-and-hold, moving average crossover)

### Infrastructure & Deployment
- ⬜ Real-time inference pipeline
- ⬜ REST API for predictions
- ⬜ Cloud training support (AWS/GCP)
- ⬜ Model versioning with MLflow
- ⬜ Automated CI/CD pipeline
- ⬜ Docker containerization

### Documentation
- ⬜ API reference documentation (Sphinx)
- ⬜ Video walkthrough / tutorial
- ⬜ FAQ section
- ⬜ Contribution guidelines
- ⬜ Changelog

### Research
- ⬜ Ablation study (LSTM only vs LightGBM only vs Stacking)
- ⬜ Feature importance across market regimes
- ⬜ Optimal confidence threshold analysis
- ⬜ Label generation method comparison (triple barrier vs fixed pips vs trend scanning)
- ⬜ Cost sensitivity analysis (different spread/slippage levels)

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| 0.1.0 | 2026-04 | Initial thesis submission — full pipeline with LSTM + LightGBM stacking |
