# Tuning Guide

Use this order. Do not tune everything at once.

---

## 1. Labels First

Label quality is the foundation. Bad labels → bad model, regardless of how good the features or model architecture are.

### Check Label Distribution

```bash
pixi run python main.py --stage 3 --force
```

After Stage 3, check the log output for label distribution:

```text
Class -1: N (XX.X%)   ← Short
Class 0: N (XX.X%)    ← Hold
Class 1: N (XX.X%)    ← Long
```

### Target Distribution

- Short and Long should be roughly balanced (30-50% each)
- Hold should be at least 5-10% (too rare = model ignores it)
- If Hold < 3%, the model learns to ignore the Hold class entirely

### Key Parameters

```toml
[labels]
atr_tp_multiplier = 2.0     # Take-profit distance (ATR multiples)
atr_sl_multiplier = 2.0     # Stop-loss distance (ATR multiples)
horizon_bars = 24            # Forward-looking window (hours on H1)
```

### Effects of Each Parameter

| Parameter | Increase → | Decrease → |
|---|---|---|
| `atr_tp_multiplier` | More Hold (wider TP = harder to hit) | More Long/Short |
| `atr_sl_multiplier` | More Long/Short (wider SL = harder to stop) | More Hold |
| `horizon_bars` | More Long/Short (more time to hit a barrier) | More Hold (timeout) |

### Tested Alternatives

| Config | Short | Hold | Long | Verdict |
|---|---|---|---|---|
| `horizon=24, TP=2.0, SL=2.0` | 43.6% | 9.0% | 47.4% | Current (accepted) |
| `horizon=48, TP=2.0, SL=2.0` | ~48% | ~1.5% | ~50% | Rejected (Hold too rare) |

### Profitability Check

Stage 3 also logs label profitability after trading costs. Watch for this warning:

```text
LABEL PROFITABILITY LOW: Long XX.X%, Short XX.X% -- labels may not be economically useful
```

If both Long and Short profitability fall below 60%, the labels may not produce useful signals even with a perfect model.

---

## 2. Features Second

Feature whitelist lives in:

```text
src/thesis/shared/constants.py → CORE_STATIC_FEATURES
```

### After Any Feature Change

```bash
pixi run python main.py --stage 2 --force
```

This rebuilds features and all downstream stages (3-6).

### Recently Pruned Features

| Feature | Reason Removed |
|---|---|
| `regime_strength` | Low importance, composite of existing ADX/slope |
| `upper_wick_ratio` | Low importance |
| `lower_wick_ratio` | Low importance |
| `volume_zscore_20` | Noisy, hard to defend |

### Feature Groups for Tuning

If you need to experiment with features:

| Group | Features | Safety |
|---|---|---|
| **Trend** (keep) | `ema34_vs_ema89`, `close_vs_ema_34`, `adx_14`, `ema_slope_20` | Core, well-established |
| **Momentum** (keep) | `return_1h`, `return_4h`, `macd_hist_atr`, `rsi_14` | Core, low redundancy |
| **Volatility** (keep) | `atr_pct_close`, `atr_ratio`, `atr_percentile`, `high_low_range_20` | Core, ATR-derived |
| **Position** (keep) | `price_dist_ratio`, `price_position_20`, `pivot_position`, `vwap` | Core, interpretable |
| **Candle** (minimal) | `candle_body_ratio` | Single feature, low risk |
| **Session** (keep) | `sess_asia`, `sess_london`, `sess_ny_am`, `sess_ny_pm` | 24/5 market model |

### Feature Engineering Parameters

```toml
[features]
rsi_period = 14              # RSI lookback (standard: 14)
atr_period = 14              # ATR lookback (standard: 14)
adx_period = 14              # ADX lookback (standard: 14)
ema_slope_period = 20        # EMA slope lookback
macd_fast = 12               # MACD fast EMA (standard: 12)
macd_slow = 26               # MACD slow EMA (standard: 26)
macd_signal = 9              # MACD signal line (standard: 9)
correlation_threshold = 0.75 # Drop features above this pairwise correlation
```

### Leakage Guard

Feature code must not use:
- `shift(-n)` (future-looking shift)
- `center=True` in rolling/ewm operations
- Raw OHLCV columns as features
- Label-derived columns as features

These are enforced by tests in `tests/unit/test_leakage.py`.

---

## 3. Model Third

### Architecture Selection

```toml
[model]
architecture = "stacking"    # Current: Classic Hybrid Stacking
# architecture = "lgbm"      # Ablation: LightGBM only
```

### LightGBM Hyperparameters

Current conservative configuration:

```toml
[model]
num_leaves = 15              # Low → less overfitting
max_depth = 4                # Shallow → less overfitting
learning_rate = 0.03         # Slow → more robust
n_estimators = 300           # Moderate, with early stopping
min_child_samples = 80       # High → less overfitting
subsample = 0.80             # Row subsampling
feature_fraction = 0.70      # Column subsampling per tree
reg_alpha = 0.05             # L1 regularization
reg_lambda = 10.0            # L2 regularization (strong)
early_stopping_rounds = 30   # Stop if no improvement
```

### Tuning Direction

| Direction | When | What to Change |
|---|---|---|
| **Less capacity** | Overfitting (train >> val accuracy) | Decrease `num_leaves`, `max_depth`, `n_estimators`; increase `min_child_samples`, `reg_lambda` |
| **More capacity** | Underfitting (both train and val accuracy low) | Increase `num_leaves`, `max_depth`; decrease `min_child_samples` |
| **Faster learning** | Need quicker convergence | Increase `learning_rate`, decrease `n_estimators` |
| **Slower learning** | Need more robust model | Decrease `learning_rate`, increase `n_estimators` |

### Stacking Parameters

```toml
stacking_base_models = ["logistic_regression", "random_forest", "lightgbm"]
stacking_meta_model = "logistic_regression"
stacking_meta_fraction = 0.20     # Fraction of train window for meta learner
stacking_passthrough = false      # Pass base features to meta learner
```

- `stacking_meta_fraction`: 0.20 means the last 20% of each train window is used to train the meta-learner. Increasing this gives the meta-learner more data but reduces base-learner training data.
- `stacking_passthrough`: enabling this passes the original features to the meta-learner alongside base probabilities (increases dimensionality, may overfit).

### Random Forest Parameters

```toml
random_forest_n_estimators = 300
random_forest_max_depth = 6
random_forest_min_samples_leaf = 80
```

### Walk-Forward Parameters

```toml
[validation]
train_window_bars = 6240     # ~1 year
test_window_bars = 1040      # ~2 months
step_bars = 1040             # Non-overlapping
purge_bars = 48              # Anti-leakage gap
embargo_bars = 50            # Additional gap
```

Do not reduce purge or embargo without understanding the leakage implications.

---

## 4. Backtest Fourth (Demo Only)

Backtest parameters are for presentation, not primary evidence:

```toml
[backtest]
initial_capital = 10000.0
leverage = 10
spread_ticks = 35
slippage_ticks = 5
lots_per_trade = 0.02
confidence_threshold = 0.50
min_bars_between_trades = 18
max_drawdown_cutoff = 0.30
```

Critical alignment rule: `atr_stop_multiplier` and `atr_tp_multiplier` in `[backtest]` must match `atr_sl_multiplier` and `atr_tp_multiplier` in `[labels]`. The pipeline enforces this with a barrier alignment guard.

---

## 5. Report Honestly

If LightGBM beats Hybrid Stacking, write that result honestly. The thesis contribution is the controlled evaluation pipeline, not guaranteed market outperformance.

Include in the report:
- All model comparison results, even if stacking underperforms
- Label distribution and profitability diagnostics
- Feature importance ranking
- Limitations and honest assessment

---

## Validation Commands

After any change, validate:

```bash
# Lint
pixi run ruff check src

# Format check
pixi run ruff format --check src

# Compile check
pixi run python -m compileall -q src tests

# Run model training
pixi run python main.py --stage 4 --force

# Run tests
pixi run test-fast

# Full test suite
pixi run test
```

---

## Quick Reference: Tuning Impact

| What You Change | Minimum Stage to Rerun | Command |
|---|---|---|
| `[labels]` params | Stage 3 | `--stage 3 --force` |
| Feature whitelist | Stage 2 | `--stage 2 --force` |
| `[features]` indicator params | Stage 2 | `--stage 2 --force` |
| `[model]` hyperparams | Stage 4 | `--stage 4 --force` |
| `[validation]` window params | Stage 4 | `--stage 4 --force` |
| `[backtest]` trading params | Stage 5 | `--stage 5 --force` |
| Report wording only | Stage 6 | `--stage 6 --force` |
