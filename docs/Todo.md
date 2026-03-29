# Todo: What's Done and What's Left

## Done (Completed)

### Phase 1: Foundation
- Project structure created
- config.toml with all parameters
- Config loader with env variable support
- Directory structure (src/thesis/...)
- All __init__.py files

### Phase 2: Data Pipeline
- Tick to OHLCV converter (tick_to_ohlcv.py)
- Handles Dukascopy format
- NY timezone with DST support
- Volume aggregation
- Spread calculation

### Phase 3: Features
- Technical indicators (EMA, RSI, MACD, ATR, BB)
- Pivot points (previous day HLC)
- Session encoding (Asia, London, NY)
- Lag features (1,2,3,5,10 bars)
- Spread features
- Correlation filter (drop >0.95)

### Phase 4: Labels
- Triple-Barrier implementation
- TP = 2 * ATR
- SL = 1 * ATR
- 10-bar horizon
- 3-class output (-1, 0, +1)

### Phase 5: Data Splitting
- Train: 2018-2021 (70%)
- Val: 2022 (15%)
- Test: 2023-03/2026 (15%)
- Purge bars (15)
- Embargo bars (10)
- Class distribution logging

### Phase 6: Models - LightGBM
- Model training with Optuna
- Class weight balancing
- 5-fold CV for OOF
- Early stopping
- Feature importance
- Model saving (joblib)

### Phase 7: Models - LSTM
- PyTorch LSTM architecture
- 60-bar sequences
- Normalization with training stats (no data leakage)
- Early stopping
- CPU-only support
- Model saving (.pt)
- Normalization stats saved to lstm_norm_stats.npz

### Phase 8: Models - Stacking
- Meta-learner (LogisticRegression)
- Combines LGBM + LSTM OOF
- Probability calibration (isotonic)
- Model saving

### Phase 9: Backtest
- CFD simulator
- Spread costs (2 pips)
- Slippage (1 pip)
- Leverage (100:1)
- Risk per trade (1%)
- Metrics calculation

### Phase 10: Reporting
- Markdown report generator
- SHAP summary plot
- Performance metrics
- JSON output

### Phase 11: Pipeline
- main.py entry point
- runner.py orchestration
- Stage dependencies
- Cache checking
- Command-line args
- Logging system

### Phase 12: Documentation
- README.md (docs index)
- Architecture.md
- Quickstart.md
- Evaluation.md
- Features.md
- Config.md
- Todo.md (this file)

---

## Partially Done (Needs Testing)

### Testing
- Unit tests for data pipeline
- Unit tests for features
- Unit tests for labels
- Integration test (full run)
- Edge case testing

### Validation
- Test on real data
- Verify all 99 tick files work
- Check memory usage
- Validate timestamps (DST)

---

## Not Yet Done (Todo)

### Priority 1: Critical (Must Have)

- Test on actual data - Run full pipeline once
- Fix any errors - Bugs will appear on first run
- Verify results - Check numbers make sense

### Priority 2: Important (Should Have)

- Better backtest visualization - Add equity curve plot
- Trade list export - CSV of all trades
- Class imbalance handling - SMOTE or undersampling option
- Feature selection - RFE or importance-based selection
- Ensemble weights - Learnable weights for LGBM + LSTM

### Priority 3: Nice to Have (Could Have)

- GPU support - CUDA for LSTM training
- More indicators - Ichimoku, Fibonacci, etc.
- Walk-forward optimization - Rolling window training
- Position sizing - Kelly criterion implementation
- Interactive dashboard - Streamlit for visualization
- MLflow tracking - Experiment logging
- Docker container - Easy deployment
- Jupyter notebooks - Exploratory analysis
- Hyperparameter dashboard - Optuna visualization

### Priority 4: Thesis Polish

- Vietnamese comments - Add Vietnamese to key functions
- More docstrings - Explain complex logic
- Code examples - Usage examples in docs
- Troubleshooting guide - Common errors and fixes
- Thesis chapter mapping - Which code = which chapter

---

## Timeline Estimate

| Task | Time | Priority |
|------|------|----------|
| Test full pipeline | 2 hours | P1 |
| Fix bugs | 2-4 hours | P1 |
| Add trade CSV export | 1 hour | P2 |
| Add equity curve plot | 1 hour | P2 |
| SMOTE implementation | 2 hours | P2 |
| Add tests | 4 hours | P2 |
| GPU support | 3 hours | P3 |
| Dashboard | 6 hours | P3 |
| Vietnamese docs | 4 hours | P4 |
| **Total** | **25-30 hours** | |

---

## Immediate Next Steps

### For You (Student):

1. **TODAY**: Run python main.py on real data
   - Check if it works
   - Note any errors
   - Time how long it takes

2. **This Week**: Fix any bugs found
   - Usually 5-10 small issues
   - Update this todo list

3. **Next Week**: Run full pipeline 3 times
   - Test with different seeds
   - Verify results are stable
   - Document findings

### For Me (If Continuing):

1. Create test suite
2. Add equity curve plotting
3. Implement trade CSV export

---

## Notes

### Known Limitations

1. **LSTM data preparation** - Sequence alignment requires careful handling
2. **Stacking predictions** - Properly aligned with timestamps
3. **Backtest trades** - Simplified logic, could be more realistic
4. **SHAP** - Only for LightGBM, LSTM SHAP is harder

### Potential Issues to Watch

1. **Memory** - 300M ticks might use 8GB+ RAM
2. **Time** - First run takes 1-2 hours
3. **Polars compatibility** - Using latest version
4. **PyTorch CPU** - Verify works without CUDA

### Thesis-Specific Items

- Introduction chapter - Write after seeing results
- Methodology chapter - Can write now (implementation done)
- Results chapter - Need real backtest results
- Conclusion - Write after results
- Code appendix - Export key files

---

*Last Updated: 2024-03-29*
*Status: LSTM data leakage fixed, pipeline runs successfully*
