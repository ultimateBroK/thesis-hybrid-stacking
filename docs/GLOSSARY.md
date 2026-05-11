# Glossary

## Model Architecture

### Classic Hybrid Stacking

A two-level ensemble. Base models first produce probabilities for Short/Hold/Long. A meta-model (Logistic Regression) then learns how to combine those probabilities into a final prediction. This project uses three base learners: Logistic Regression, Random Forest, and LightGBM.

### Base Learner

A model in the first layer of stacking. This project uses:
- **Logistic Regression**: linear baseline, fast to train, provides calibrated probability estimates
- **Random Forest**: bagging tree ensemble (300 trees, max_depth=6), robust to outliers
- **LightGBM**: gradient boosting tree learner, typically the strongest single model

### Meta Learner

The second-layer model in stacking. This project uses Logistic Regression as the meta learner, which takes the 9-dimensional probability vector (3 classes × 3 base learners) and learns optimal combination weights.

### LightGBM-Only Ablation

A single-model baseline using only LightGBM without stacking. Serves as a comparison point to evaluate whether the added complexity of stacking provides value.

---

## Labeling

### Triple-Barrier Labeling

A financial labeling method (from Marcos Lopez de Prado) with three exits:
- **Take-profit (TP)**: upper barrier hit → Long label (+1)
- **Stop-loss (SL)**: lower barrier hit → Short label (-1)
- **Timeout/Horizon**: no barrier hit → Hold label (0)

It is safer than simply asking whether price is higher after N bars because it accounts for intra-period volatility through ATR-based barriers.

### Direction-Barrier Labeling

The specific implementation in this project. Barriers are computed using ATR:
```text
upper = close + atr_tp_multiplier × ATR
lower = close - atr_sl_multiplier × ATR
```

The algorithm scans forward up to `horizon_bars` to find the first barrier hit.

### Asymmetric Barriers

TP and SL can use different ATR multipliers. Current config uses symmetric 2.0/2.0, but asymmetric settings (e.g., 3.0/1.5) can produce different label distributions.

### Censored Labels

Bars whose forward horizon extends beyond available data are labeled -2 and dropped before training. This prevents training on incomplete information.

### Sample Weights (Average Uniqueness)

Lopez de Prado's method to reduce overlap bias. When multiple labels cover the same time period (overlapping forward windows), each gets a lower weight proportional to how many other labels overlap with it. Weights are normalized to mean 1.0.

---

## Validation

### Walk-Forward Validation

A chronological evaluation method. The model trains on past data and predicts future windows, then the window slides forward. This simulates how a model would perform in real-time deployment.

```text
Window 1: |<- train ->| gap |<- test ->|
Window 2:               |<- train ->| gap |<- test ->|
Window 3:                              |<- train ->| gap |<- test ->|
```

### Sliding Window

Each window uses a fixed-size train and test block that moves forward by `step_bars`. Non-overlapping test windows ensure each data point is predicted exactly once.

### Purge

A gap between the end of the training block and the start of the test block. This removes label lookahead leakage because triple-barrier labels use forward-looking data. Default: 48 bars (2 days on H1).

### Embargo

An additional gap after the test block before the next training block can use that data. Prevents information from the test period from leaking into the next window's training. Default: 50 bars.

### Out-of-Fold (OOF) Predictions

Predictions generated on test windows that were not used to train the model for that window. All OOF predictions are concatenated to form the final prediction file, giving one prediction per data point.

---

## Metrics

### Directional Accuracy

Accuracy computed only on bars where both the true and predicted labels are non-zero (Short or Long). Hold predictions on directional bars count as wrong. Useful for evaluating actual trading signal quality.

### MDA (Market Directional Accuracy)

Two variants:
- **MDA (no hold)**: accuracy on bars where true label is Short or Long only
- **MDA (binary)**: accuracy for Long vs Short only, Hold predictions count as wrong

### Macro F1

Average F1 score across all three classes (Short, Hold, Long). It penalizes models that ignore minority classes (especially Hold). More informative than accuracy for imbalanced datasets.

### Weighted F1

Support-weighted average F1. Gives more weight to classes with more samples. Can be misleading if the majority class dominates.

### Balanced Accuracy

Average recall across classes. Equivalent to (recall_Short + recall_Hold + recall_Long) / 3. Robust to class imbalance.

### Confusion Matrix

A 3×3 matrix showing true labels (rows) vs predicted labels (columns):
- Diagonal: correct predictions
- Off-diagonal: misclassifications
- Direction confusion matrix: 2×2 (Short vs Long only, Hold excluded)

### Calibration

How well do predicted probabilities match actual frequencies?
- **ECE (Expected Calibration Error)**: weighted average of |accuracy - confidence| across probability bins
- **Brier Score**: mean squared error of probability predictions
- **Log Loss**: cross-entropy of predicted probabilities

---

## Trading

### Backtest

A simulator that translates model signals into hypothetical trades using the `backtesting.py` library with fractional lot support. In this thesis it is an application demo, not the primary proof.

### CFD (Contract for Difference)

A derivative that allows trading on price movements without owning the underlying asset. XAU/USD CFDs are typically traded with leverage.

### Signal-to-Noise Ratio

In financial time series, the predictable component (signal) is typically much smaller than the random component (noise). This makes prediction inherently difficult and limits achievable accuracy.

### ATR (Average True Range)

Wilder's volatility measure. Used for:
- Barrier computation in labels
- Feature normalization (ATR-normalized distances)
- Stop-loss/take-profit in backtest

---

## Infrastructure

### Session

A timestamped directory containing all outputs from one pipeline run. Enables reproducibility and comparison across runs.

### Config Snapshot

A copy of `config.toml` saved at the start of each session. Ensures the exact configuration can be retrieved for any past run.

### Pipeline Cache

Stage outputs are cached to avoid re-computation. Cache invalidation can be by file path, config hash, or disabled entirely.

### GRU (Gated Recurrent Unit)

A recurrent neural network architecture. GRU was part of older experiment directions but is not the current production runtime for this thesis path.
