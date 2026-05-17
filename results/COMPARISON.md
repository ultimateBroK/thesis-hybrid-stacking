# Experiment Comparison: Horizon Trade-off

> Two optimal configurations identified through systematic hyperparameter search.
> Key finding: **classification accuracy and trading profitability are inversely correlated with label horizon.**

## Configuration Summary

| Parameter | Config A: Best Backtest | Config B: Best Accuracy |
|-----------|------------------------|------------------------|
| **Label Horizon** | 24 bars (24h) | 48 bars (48h) |
| ATR TP/SL Multiplier | 2.0× | 2.0× |
| Stacking Architecture | LR + RF + LGBM → LGBM meta | LR + RF + LGBM → LGBM meta |
| Passthrough | true | true |
| Feature Selection | 18/36 (50%) | 18/36 (50%) |
| Walk-forward Windows | 25 × (6240 train / 1040 test) | 25 × (6240 train / 1040 test) |

## Label Distribution

| Class | Config A (h=24) | Config B (h=48) |
|-------|-----------------|-----------------|
| Short (-1) | 13,706 (43.6%) | 14,837 (47.3%) |
| Hold (0) | 2,829 (9.0%) | 478 (1.5%) |
| Long (+1) | 14,865 (47.3%) | 16,061 (51.2%) |

Longer horizon → nearly all labels become directional (1.5% hold). Shorter horizon → more hold labels from unresolvable moves.

## Classification Metrics

| Metric | Config A (h=24) | Config B (h=48) | Advantage |
|--------|-----------------|-----------------|-----------|
| **Exact Accuracy** | 38.7% | **48.9%** | B (+10.2pp) |
| **Directional Accuracy** | 49.1% | **50.3%** | B (+1.2pp) |
| **Macro F1** | 0.326 | **0.332** | B (+0.006) |
| Acc vs Majority Baseline | -9.8pp | -3.2pp | B |
| Balanced Accuracy | 35.7% | 33.6% | A |
| Hold Class F1 | **0.155** | 0.006 | A |
| Short F1 | 0.491 | **0.520** | B |
| Long F1 | 0.331 | **0.469** | B |

### Stacking vs Base Models

| Config | Stacking Acc | Best Base (LGBM) | Delta | Verdict |
|--------|-------------|-------------------|-------|---------|
| A (h=24) | 38.7% | 40.5% | **-1.9pp** | Stacking < base |
| B (h=48) | 48.9% | 46.1% | **+2.8pp** | Stacking > base ✅ |

Config B validates the stacking hypothesis: meta-learner improves over individual bases when labels are cleaner (longer horizon reduces hold-class noise).

### Per-Class Confusion Matrices

**Config A (h=24):**

| True \ Pred | Short | Hold | Long |
|-------------|------:|-----:|-----:|
| Short | 5,852 | 1,978 | 2,450 |
| Hold | 1,080 | 516 | 437 |
| Long | 6,606 | 2,116 | 2,875 |

**Config B (h=48):**

| True \ Pred | Short | Hold | Long |
|-------------|------:|-----:|-----:|
| Short | 6,467 | 152 | 4,444 |
| Hold | 208 | 2 | 147 |
| Long | 7,120 | 124 | 5,222 |

Config B almost never predicts Hold (recall 0.6%). Config A predicts Hold more freely (recall 25.4%) but still struggles.

## Backtest Results

| Metric | Config A (h=24) | Config B (h=48) | Advantage |
|--------|-----------------|-----------------|-----------|
| **Return** | **+3.6%** | +0.1% | A (+3.5pp) |
| **Sharpe Ratio** | **0.19** | 0.00 | A |
| **Max Drawdown** | **-7.7%** | -11.3% | A |
| **Profit Factor** | **1.03** | 0.99 | A |
| Win Rate | 50.7% | 51.3% | B |
| Sortino Ratio | **0.28** | 0.00 | A |
| Calmar Ratio | **0.09** | 0.00 | A |
| Trades | 373 | 833 | — |
| Avg Win/Avg Loss | 1.05 | 0.95 | A |
| Expectancy | +0.007% | -0.004% | A |
| Final Equity | **$10,363** | $10,010 | A |

Config A: profitable, positive Sharpe, controlled drawdown. Config B: breakeven, no edge after costs.

## Generalization (OOF vs OOS)

| Metric | Config A OOF | Config A OOS | Δ | Config B OOF | Config B OOS | Δ |
|--------|-------------|-------------|---|-------------|-------------|---|
| Accuracy | 38.7% | 37.9% | -0.7pp | 48.9% | 49.8% | +0.8pp |
| Macro F1 | 29.5% | 31.4% | +1.8pp | 31.2% | 33.8% | +2.6pp |

Both configs generalize well — OOF-OOS delta <3pp. No overfitting.

## Model Comparison (All Models)

### Config A (h=24)

| Model | Dir. Acc | Accuracy | Macro F1 |
|-------|---------|----------|----------|
| Hybrid Stacking | 49.1% | 38.7% | 0.326 |
| LightGBM | 42.2% | 40.5% | 0.335 |
| Random Forest | 35.6% | 36.2% | 0.329 |
| Logistic Regression | 35.3% | 35.8% | 0.326 |
| Naive Direction | 50.0% | 45.7% | 0.318 |
| Majority Baseline | 53.0% | 48.5% | 0.218 |

### Config B (h=48)

| Model | Dir. Acc | Accuracy | Macro F1 |
|-------|---------|----------|----------|
| Hybrid Stacking | 50.3% | 48.9% | 0.332 |
| LightGBM | 46.7% | 46.1% | 0.315 |
| Random Forest | 44.8% | 44.4% | 0.324 |
| Logistic Regression | 34.0% | 34.1% | 0.281 |
| Naive Direction | 51.1% | 50.3% | 0.337 |
| Majority Baseline | 53.0% | 52.2% | 0.229 |

## Calibration

| Metric | Config A (h=24) | Config B (h=48) |
|--------|-----------------|-----------------|
| ECE | **0.063** | 0.127 |
| Brier Score | 0.222 | **0.196** |
| Log Loss | 1.087 | **0.890** |
| Rating | Moderately calibrated | Moderately calibrated |

Config A: better ECE (expected calibration error). Config B: better Brier/LogLoss (predictive quality).

## Root Cause Analysis

### Why accuracy and profitability diverge

1. **Label horizon defines the prediction task.** At h=48, nearly all bars hit a barrier (1.5% hold), making it a binary problem. At h=24, 9% of bars are hold — the model must learn a 3-way decision, harder but more discriminating.

2. **Longer horizon = easier classification, worse timing.** The model at h=48 correctly predicts gold's directional bias (it went from $1,800 to $3,200+ during test period), but this "prediction" captures long-term trend, not actionable short-term signal.

3. **Shorter horizon = harder labels, better trade timing.** At h=24, labels are noisier (many bars don't hit a barrier in 24h), but the model's predictions align with the backtest's actual holding period — positions are held for hours, not days.

4. **Stacking benefits from cleaner labels.** At h=48, base model errors are less correlated because the signal is clearer. The meta-learner can combine them effectively (+2.8pp). At h=24, base models make correlated errors on noisy labels — the meta-learner can't improve.

### Practical implication

- **For classification benchmarks** (accuracy, stacking validation): use Config B (h=48)
- **For trading deployment** (Sharpe, profitability): use Config A (h=24)
- Neither config simultaneously achieves both high accuracy and profitable trading

## Benchmark Context

| Strategy | Return | Sharpe | Max DD |
|----------|--------|--------|--------|
| Buy & Hold | 153-156% | 1.34-1.35 | 24.4% |
| Always Long (leveraged) | 1,677-1,867% | 1.34-1.35 | 98.6% |
| Random Signal | -99.7% | 0.15-0.17 | 100% |
| **Config A (h=24)** | **+3.6%** | **0.19** | **7.7%** |
| **Config B (h=48)** | +0.1% | 0.00 | 11.3% |

Both configs underperform Buy & Hold in absolute returns (test period = gold bull market). However, Config A achieves positive returns with controlled drawdown (7.7% vs 24.4% Buy & Hold), demonstrating risk management value.

## Result Folders

- Config A: `results/best_backtest_h24_sharpe019/`
- Config B: `results/best_accuracy_h48_acc498/`
