# Architecture: How This Project Works

## 🏗️ Big Picture

```
┌─────────────────────────────────────────────────────────────────┐
│                        THESIS PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Stage 1: DATA          Stage 2: FEATURES      Stage 3: LABELS  │
│  ├─ Read ticks        ├─ EMA (20,50,200)    ├─ Triple-Barrier   │
│  ├─ Make H1 candles   ├─ RSI, MACD, ATR    │   TP = 2×ATR     │
│  └─ Save to parquet   ├─ Pivot points       │   SL = 1×ATR      │
│                       ├─ Session hours      │   10 bars         │
│                       └─ Lag features       └─ 3 classes       │
│                                                                  │
│  Stage 4: SPLIT                                                  │
│  ├─ Train: 2018-2022 (60%)      ← Training period             │
│  ├─ Val:   2023 (15%)          ← Validation for stacking        │
│  └─ Test:  2024-2026 (25%)     ← Final backtest period        │
│                                                                  │
│  Stage 5: LIGHTGBM    Stage 6: LSTM         Stage 7: STACKING  │
│  ├─ Tabular learning  ├─ Sequential data    ├─ Combine both     │
│  ├─ Optuna tuning     ├─ 60-bar sequences   ├─ Meta-learner     │
│  └─ Feature importance └─ PyTorch CPU        └─ Calibrated      │
│                                                                  │
│  Stage 8: BACKTEST     Stage 9: REPORTING                        │
│  ├─ CFD simulation     ├─ SHAP explainability                    │
│  ├─ Real costs         ├─ Performance metrics                     │
│  └─ Equity curve       └─ Thesis markdown                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Code Structure

```
src/thesis/
├── __init__.py              # Package info
│
├── config/
│   ├── __init__.py
│   └── loader.py           # Read config.toml
│
├── data/
│   ├── tick_to_ohlcv.py    # Stage 1: Ticks → Candles
│   └── splitting.py        # Stage 4: Train/Val/Test split
│
├── features/
│   └── engineering.py      # Stage 2: Technical indicators
│
├── labels/
│   └── triple_barrier.py   # Stage 3: Label generation
│
├── models/
│   ├── lightgbm_model.py   # Stage 5: Tree model
│   ├── lstm_model.py       # Stage 6: Neural network
│   └── stacking.py         # Stage 7: Meta-learner
│
├── backtest/
│   └── cfd_simulator.py    # Stage 8: Trading simulation
│
├── reporting/
│   └── thesis_report.py    # Stage 9: Results & SHAP
│
└── pipeline/
    └── runner.py           # Connect all stages
```

## 🔄 Data Flow

### Step-by-Step:

**1. Input**: Dukascopy tick files
```
data/raw/XAUUSD/
├── 2018-01.parquet   (5M ticks)
├── 2018-02.parquet   (5M ticks)
├── ...
└── 2026-03.parquet
```

**2. Output Chain**:
```
ticks → ohlcv.parquet → features.parquet → labels.parquet
                                          ↓
                    ┌─────────────────────┼─────────────────────┐
                    ↓                     ↓                     ↓
              train.parquet          val.parquet          test.parquet
                    ↓                     ↓                     ↓
            lightgbm_oof.parquet   lstm_oof.parquet   final_predictions.parquet
                    ↓                     ↓                     ↓
                    └──────────┬──────────┘                     ↓
                               ↓                                ↓
                         stacking_meta.pkl                backtest_results.json
                               ↓                                ↓
                         final_predictions.parquet      thesis_report.md
```

## 🔧 Key Components

### 1. Config System (config.toml)
- One file controls everything
- Environment variables can override
- Example: THESIS_DATA__TIMEFRAME=30m

### 2. Data Processing
```python
# Tick → OHLCV
mid = (ask + bid) / 2
candle = {
    open: first(mid),
    high: max(mid),
    low: min(mid),
    close: last(mid),
    volume: sum(ask_vol + bid_vol)
}
```

### 3. Triple-Barrier Labels
```python
# For each candle:
tp = close + 2 * atr      # Take profit
sl = close - 1 * atr      # Stop loss

# Look ahead 10 bars:
if high hits tp first:  label = +1 (Long)
if low hits sl first:   label = -1 (Short)
if neither:             label = 0  (Hold)
```

### 4. Data Splitting (Market Regime Based)

Actual dates used:
- Train: 2018-01-01 to 2022-12-31 (60%)
- Validation: 2023-01-01 to 2023-12-31 (15%)
- Test: 2024-01-01 to 2026-03-31 (25%)

Important: Test period includes gold's 2024-2026 bull run (2065 to 4494 USD).

### 5. Model Stacking
```
LightGBM outputs:  [P(short), P(hold), P(long)]
LSTM outputs:        [P(short), P(hold), P(long)]
                         ↓
                  Meta-features: 6 numbers
                         ↓
              Logistic Regression (meta-learner)
                         ↓
              Final: [P(short), P(hold), P(long)]
```

### 6. LSTM Normalization (Anti-Leakage)

Fixed March 28, 2026:
```python
# Training: Calculate and save stats
norm_stats = {
    'train_means': means,
    'train_stds': stds
}
np.save('models/lstm_norm_stats.npz', **norm_stats)

# Testing: Load training stats (NOT test stats)
stats = np.load('models/lstm_norm_stats.npz')
X_test_norm = (X_test - stats['train_means']) / stats['train_stds']
```

This prevents data leakage from test set into predictions.

### 7. Backtest Realism
```python
# Trading costs included:
entry_price = price + spread/2 + slippage  # Buy
exit_price  = price - spread/2 - slippage  # Sell

# Account for:
- Spread (2 pips)
- Slippage (1 pip)
- Leverage (100:1)
- Max position: 2.0 lots
```

## 🧠 Why This Architecture?

| Choice | Reason |
|--------|--------|
| Polars | Fast for large tick data |
| LightGBM | Good for tabular features |
| LSTM | Captures price patterns |
| Stacking | Combines both strengths |
| Triple-Barrier | Realistic profit targets |
| Market Regime Split | Tests on gold bull run |

## 🎯 Cache System

Each stage checks: "Already done?"

```python
if output.exists() and not force_rerun:
    use_cached_file()   # Skip stage
else:
    run_stage()         # Generate file
```

This saves hours when re-running.

## 🔄 Pipeline Dependencies

```
Stage 1 (data) ─────┐
                    ↓
Stage 2 (features) ─┤
                    ↓
Stage 3 (labels) ─────┤
                    ↓
Stage 4 (split) ──────┤
                    ├──→ Stage 5 (LightGBM) ───┐
                    └──→ Stage 6 (LSTM) ──────┼──→ Stage 7 (Stacking)
                                              ↓
                                    Stage 8 (Backtest) ───→ Stage 9 (Report)
```

If you run Stage 5, it auto-runs 1-4 first.

## 📊 Data Sizes

| File | Size | Rows |
|------|------|------|
| Raw ticks | ~500MB | 300M+ |
| OHLCV H1 | ~50MB | 52,000 |
| Features | ~80MB | 52,000 |
| Train set | ~35MB | 36,000 |
| Val set | ~8MB | 8,000 |
| Test set | ~8MB | 8,000 |

## 🔒 Security Notes

### Data Leakage Prevention
1. LSTM normalization uses training stats only (saved to models/lstm_norm_stats.npz)
2. Triple-barrier uses forward-looking bars (no future info in features)
3. Lag features use .shift() (no lookahead)
4. Pivot points use previous day's data with .shift(1)

### Purge and Embargo
```toml
[splitting]
purge_bars = 15      # Remove 15 bars around split
embargo_bars = 10    # Additional safety margin
```

---

Next: See Quickstart.md to run it!
