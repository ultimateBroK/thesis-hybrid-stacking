## Target: small but still “Hybrid Stacking”

Keep only this spine:

```text
1. Load OHLCV
2. Build features
3. Build triple-barrier labels
4. Walk-forward split
5. Train base models
6. Train meta model
7. Evaluate
8. Backtest demo
9. Report
```

Everything else is optional noise.

---

## What to remove / freeze first

### 1. Remove architecture switching

Current idea:

```text
architecture = "stacking" or "lgbm"
dispatcher.py
lgbm path
stacking path
```

For your thesis, topic is fixed: **Hybrid Stacking**.

So make Stage 4 only one path:

```python
train_hybrid_stacking(config)
```

No dispatcher. No architecture switch. No `lgbm_expanded_features`.

Keep individual LightGBM/RF/LogReg only as **baseline comparison inside the same training file**, not as separate architecture.

---

### 2. Remove advanced stacking modes

Keep only **one** stacking protocol:

```text
Outer walk-forward
Inside each train window:
  chronological split:
    80% base-train
    20% meta-train
```

Delete or ignore:

```text
expanding-origin internal OOF folds
calibration
passthrough raw features
soft-vote extras
FrozenEstimator
internal purge knobs
```

Use this simple logic:

```python
base_df = train_df[:80%]
meta_df = train_df[80%:]

base models train on base_df
base models predict meta_df
meta model trains on base predictions
base models refit on full train_df
base models predict test_df
meta model predicts final test labels
```

This is easy to draw, easy to explain, easy to defend.

---

### 3. Remove feature-selection pipeline complexity

Current:

```text
DropDuplicateFeatures
DropCorrelatedFeatures
RobustScaler
SelectKBest
fallback pipeline
selected feature names
```

Small version:

```text
select fixed feature list
fill/drop NaN
RobustScaler only for Logistic Regression
no feature selection
```

LightGBM and Random Forest do not need scaling.

Use fixed features:

```text
return_1h / log_returns
return_4h
return_24h
rsi_14
atr_14
adx_14
macd
macd_signal
ema_slope
ema_cross
bb_pctb
hour
day_of_week
```

No dynamic feature registry unless needed.

---

### 4. Make labels brutally clear

Keep triple-barrier, but remove unnecessary extras.

Keep:

```text
label
upper_barrier
lower_barrier
event_end
```

Optional but useful:

```text
touched_bar
```

Remove/freeze:

```text
sample_weight
average_uniqueness
label profitability warning
many diagnostics
```

For thesis, average uniqueness is nice but not necessary. It makes code sound quant-pro, but it increases defense burden.

---

### 5. Reduce reporting

Keep 4 outputs only:

```text
predictions.parquet / predictions.csv
metrics.json
model_comparison.json
report.md
```

Stop generating too many charts/files unless already stable.

Report only:

```text
label distribution
walk-forward windows
classification metrics
confusion matrix
model comparison
backtest demo
```

---

## Suggested new small folder structure

```text
src/thesis/
  config.py
  data.py
  features.py
  labels.py
  validation.py
  hybrid_stacking.py
  metrics.py
  backtest.py
  report.py
  pipeline.py
```

That is enough.

Current 6-stage folders look professional, but for ADHD + deadline, they create too many doors.

---

## Minimum viable Hybrid Stacking file

One file should contain the core model:

```text
src/thesis/hybrid_stacking.py
```

Functions:

```python
def train_hybrid_stacking(df, feature_cols, config):
    windows = make_walk_forward_windows(...)
    all_preds = []

    for window in windows:
        result = train_one_window(df, window, feature_cols, config)
        all_preds.append(result)

    return concat_predictions(all_preds)
```

```python
def train_one_window(df, window, feature_cols, config):
    train_df = ...
    test_df = ...

    base_df, meta_df = chronological_split(train_df, meta_fraction=0.2)

    base_models = {
        "logreg": LogisticRegression(...),
        "rf": RandomForestClassifier(...),
        "lgbm": LGBMClassifier(...),
    }

    fit base models on base_df

    meta_X = predict_proba(base_models, meta_df)
    meta_y = meta_df["label"]

    meta_model = LogisticRegression(...)
    fit meta_model on meta_X, meta_y

    refit base models on full train_df

    test_meta_X = predict_proba(base_models, test_df)
    final_pred = meta_model.predict(test_meta_X)

    also save base model predictions for comparison
```

That is your thesis. Everything else is support.

---

## Do this in order

### Step 1 — Freeze current branch

```bash
git checkout hybrid-refactor
git checkout -b hybrid-small
```

Do not destroy old work.

---

### Step 2 — Make one simple config

```toml
[model]
stacking_meta_fraction = 0.20
stacking_meta_model = "logistic_regression"
stacking_passthrough = false
stacking_internal_folds = 0
stacking_calibrate_base = false
prediction_confidence_threshold = 0.0
```

This already forces simpler behavior in your current code.

---

### Step 3 — Delete by disabling first, not physically deleting

Before deleting files, make pipeline call only:

```text
generate_data
generate_features
generate_labels
train_stacking_walk_forward
run_backtest
generate_report
```

No architecture dispatcher.

---

### Step 4 — Fix label ambiguity

This is the one code change I would do before simplification.

Ambiguous TP+SL in same candle should not become Hold.

Make it censored/drop.

---

### Step 5 — Only then physically clean folders

After results run, remove unused files.

Do **not** start by deleting. That creates chaos.

---

## What to keep

Keep these because they defend thesis:

```text
triple-barrier labeling
walk-forward validation
purge / embargo
base models
meta model
baseline comparison
classification report
backtest demo
```

## What to cut

Cut these because they bloat code:

```text
architecture dispatcher
LightGBM separate architecture path
expanded feature mode
internal OOF stacking
calibration
passthrough raw features
feature-engine DropCorrelated/SelectKBest pipeline
average uniqueness weights
too many report artifacts
too many cache modes
over-detailed session machinery
```

---

## Caveman final

Do not make project “smarter.”

Make it **straighter**.

```text
Hybrid stacking only.
One validation method.
One label method.
One feature list.
One report.
```

Small code. Clear defense. Better chance to finish.
