# Thesis Project Plan

## Objective
Build an end-to-end ML pipeline for Bachelor's thesis at Thuy Loi University:  
**Hybrid Stacking (LSTM + LightGBM)** for **XAU/USD H1 trading signal forecasting**.

Pipeline: raw ticks → OHLCV → features → Triple-Barrier labels → train/val/test → LightGBM → LSTM → stacking → backtest → SHAP report.

---

## Constraints & Requirements
- **Hardware**: CPU-only (PyTorch CPU)
- **Package Manager**: Pixi (pixi.toml)
- **Config**: Central `config.toml` with `THESIS_<SECTION>__<KEY>` env overrides
- **Backtest**: Single OOS run only (no overfitting)
- **Code**: `src/thesis/` package, Python ≥3.13
- **Timeline**: 23 Mar – 28 Jun (10 weeks)

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1)
1. Create `config.toml` with all parameters
2. Scaffold `src/thesis/` package structure
3. Implement config loader (TOML + env overrides)

### Phase 2: Data Pipeline (Week 2)
4. **Data ingestion**: Load raw parquet → OHLCV H1
   - Compute mid price, resample, ATR spike filter
   - DST-aware day-roll (America/New_York)
   - Output: `data/processed/ohlcv.parquet`

### Phase 3: Feature Engineering (Week 3)
5. **Features**: EMA(20,50,200), RSI(14), MACD(12,26,9), ATR(14), Pivot Points, Spread, Session hour

### Phase 4: Labels & Splitting (Week 4)
6. **Labels**: Triple-Barrier (TP=Close+2ATR×0.5, SL=Close−1ATR×0.5, horizon=10, classes={+1,0,−1})
7. **Split**: Train 2018-2021 (~70%) / Val 2022 (~15%) / Test 2023-03/2026 (~15%) with 10-20 bar purge/embargo
   - **Train**: Normal + Trade War + COVID shock
   - **Val**: Russia-Ukraine war + Fed rate hikes (stress test)
   - **Test**: SVB crisis + Gold ATH + "New Regime" (gold rises despite high rates)

### Phase 5: Models (Weeks 5-7)
8. **LightGBM baseline**: Optuna tuning, class weighting
9. **LSTM**: 60-bar sequences, PyTorch CPU, early stopping
10. **Hybrid Stacking**: Meta-learner on 6 OOF probabilities

### Phase 6: Evaluation & Reporting (Weeks 8-9)
11. **Backtest**: CFD costs (spread, slippage, commission, swap)
12. **SHAP**: Global + local explainability
13. **Reporting**: Markdown/JSON + figures

### Phase 7: Integration (Week 10)
14. **Pipeline runner**: Caching orchestration
15. **Entry point**: `main.py` → `thesis.pipeline.runner.run_thesis_workflow()`

---

## Success Criteria
- End-to-end pipeline runs without errors
- One complete OOS backtest with realistic CFD costs
- SHAP analysis provides actionable insights
- Reproducible results with fixed random seeds
