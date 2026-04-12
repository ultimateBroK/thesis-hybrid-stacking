# Review Fixes Implementation Plan

## TL;DR
> **Summary**: Fix 7 critical bugs in the thesis pipeline: stacking data leakage, backtest TP/high-low checks, triple-barrier tie-breaking, purge/embargo settings, walk-forward CV, LSTM normalization lookahead, and torch seeding.
> **Estimated Effort**: Large

## Context

### Original Request
Fix critical bugs in the Hybrid Stacking (LSTM + LightGBM) thesis project for XAU/USD H1 trading signal prediction. The bugs range from data leakage in stacking to missing backtest features.

### Key Findings from Code Review

#### Fix 1: Stacking Data Leakage (CRITICAL)
**Location**: `src/thesis/models/lightgbm_model.py` lines 356-360
**Issue**: After Optuna tuning, the model is refit on `X_train + X_val` combined. Then `predict_proba(X_val)` is called - these are in-sample predictions (100% data leakage).
**Current Code**:
```python
# Lines 356-360 in lightgbm_model.py
logger.info("Training final model on combined train+val data...")
X_combined = np.vstack([X_train, X_val])
y_combined = np.concatenate([y_train, y_val])
model.fit(X_combined, y_combined)  # ← Trained on val!

# Then in train_lightgbm() line 109:
val_probs = model.predict_proba(X_val_df)  # ← In-sample!
```

#### Fix 2: Backtest Missing TP + High/Low Checks (CRITICAL)
**Location**: `src/thesis/backtest/cfd_simulator.py`
**Issues**:
1. No Take Profit orders placed (only trailing stop exists)
2. Only checks `close` price for margin calls, stop-outs, trailing stops
3. Missing intra-bar spike detection using `high`/`low`

**Current Code** (lines 296-306, 456, 506):
```python
# Only checks close price
if capital <= initial_capital * stop_out_level:
    ...
if current_price <= trailing_stop_price:  # Uses close only
    ...
```

#### Fix 3: Triple Barrier Tie-Breaking (CRITICAL)
**Location**: `src/thesis/labels/triple_barrier.py` lines 180-189
**Issue**: When both TP and SL hit in same bar, uses close price to guess which hit first - this is unknowable and inflates performance.
**Current Code**:
```python
if tp_touched and sl_touched:
    # Both touched on same bar - use close price to break tie
    future_close = close_arr[future_idx]
    if future_close >= current_close:
        label = 1  # Long
    else:
        label = -1  # Short
```

#### Fix 4: Purge/Embargo < Horizon Bars (HIGH)
**Location**: `config.toml` lines 78-79, 192-193, 265-266
**Issue**: `horizon_bars = 20` but purge/embargo values are 10-15. Target labels extend beyond purge window causing data leakage.

#### Fix 5: Missing Walk-Forward CV Flag (HIGH)
**Location**: `config.toml` line 82-86
**Issue**: Walk-forward CV configuration exists but `use_walk_forward_cv = true` flag is missing under `[splitting]`.

#### Fix 6: LSTM Normalization Lookahead (HIGH)
**Location**: `src/thesis/models/lstm_model.py` lines 101-104, 265-269
**Issue**: `_create_sequences()` computes mean/std on each dataset separately. Val sequences normalized using val's own statistics (future information leak).
**Current Code**:
```python
X_train, y_train, train_means, train_stds = _create_sequences(train_df, ...)
X_val, y_val, _, _ = _create_sequences(val_df, ...)  # ← Uses val stats!
```

#### Fix 7: Missing Torch Seed (MEDIUM)
**Location**: `main.py` - no torch seeding
**Issue**: No `torch.manual_seed()` call - LSTM training non-reproducible.

## Objectives

### Core Objective
Eliminate all critical data leakage and methodology flaws to ensure valid thesis results.

### Deliverables
- [ ] Fix 1: Stacking data leakage eliminated
- [ ] Fix 2: Backtest includes TP orders and high/low checks
- [ ] Fix 3: Ambiguous triple-barrier bars assigned label 0
- [ ] Fix 4: All purge/embargo values ≥ 25 bars
- [ ] Fix 5: Walk-forward CV enabled in config
- [ ] Fix 6: LSTM uses train-only normalization stats
- [ ] Fix 7: Reproducible torch seeding added

### Definition of Done
- [ ] `pixi run test` passes with 70%+ coverage
- [ ] `pixi run lint` shows no errors
- [ ] `pixi run workflow --force` completes end-to-end
- [ ] Backtest results show realistic performance (not inflated)

### Guardrails (Must NOT)
- Do NOT change model architectures or hyperparameters
- Do NOT alter train/val/test date ranges
- Do NOT modify feature engineering logic
- Preserve backward compatibility of config structure

## TODOs

- [x] **Step 1: Fix Triple Barrier Tie-Breaking (Foundation Fix)**
  When both TP and SL barriers are touched in the same bar, assign label 0 (Hold) instead of using close price to guess.
  **Files**: `src/thesis/labels/triple_barrier.py`
  Lines 180-189: Replace tie-breaking logic — `label = 0` for ambiguous bars.
  **Acceptance**: Labels stage runs; class distribution shows increased Hold labels.

- [x] **Step 2: Fix Config Purge/Embargo Values**
  Increase all purge/embargo values to ≥ 25 (above horizon_bars=20).
  **Files**: `config.toml`
  `purge_bars = 25`, `cv_purge = 25`, `cv_embargo = 10`, `stacking_purge = 25`, `stacking_embargo = 10`.
  **Acceptance**: Config loads; values visible in pipeline logs.

- [x] **Step 3: Add Walk-Forward CV Flag**
  Add `use_walk_forward_cv = true` to `[splitting]` section.
  **Files**: `config.toml`
  **Acceptance**: LightGBM training uses walk-forward CV (visible in logs).

- [x] **Step 4: Fix LSTM Normalization Lookahead**
  Compute normalization stats on train set only, pass to val/test sequence creation.
  **Files**: `src/thesis/models/lstm_model.py`
  Add `norm_stats` param to `_create_sequences()`, pass train stats to val call.
  **Acceptance**: LSTM training runs; normalization stats saved correctly.

- [x] **Step 5: Fix Stacking Data Leakage in LightGBM**
  Do NOT refit on train+val before generating validation predictions. Keep model trained on train only.
  **Files**: `src/thesis/models/lightgbm_model.py`
  Remove combined refit in `_train_with_optuna()`. Fit on train only for OOS stacking inputs.
  **Acceptance**: LightGBM val predictions are truly OOS; stacking uses unbiased base predictions.

- [x] **Step 6: Verify LSTM No Similar Leakage**
  Verify LSTM doesn't have similar stacking leakage (no refit on combined data).
  **Files**: `src/thesis/models/lstm_model.py`
  **Acceptance**: Confirmed LSTM uses early stopping only, no combined refit.

- [x] **Step 7: Add Take-Profit to Backtest + High/Low Checks**
  Add hard TP orders (±1.5×ATR), check high/low for intra-bar margin calls and stops.
  **Files**: `src/thesis/backtest/cfd_simulator.py`
  Add TP price tracking, high/low exit checks, SL-first priority for ambiguous bars.
  **Acceptance**: Backtest shows TP exits; trade records include "take_profit" exit reason.

- [x] **Step 8: Add Global Torch Seed**
  Add `torch.manual_seed()` and `np.random.seed()` in main.py after config load.
  **Files**: `main.py`
  **Acceptance**: LSTM training produces identical results across runs with same seed.

- [x] **Step 9: Update Tests**
  Add/update tests for tie-breaking, LSTM normalization, backtest TP/high-low.
  **Files**: tests under `tests/unit/`
  **Acceptance**: New tests pass; coverage ≥ 70%.

- [x] **Step 10: Run Full Verification**
  Run `pixi run lint && pixi run format && pixi run test`.
  **Acceptance**: All commands succeed; no errors.

## Implementation Order

1. **Step 1** (Triple Barrier) - Foundation fix, affects labels
2. **Step 2** (Config purge/embargo) - Config-only change
3. **Step 3** (Walk-forward CV flag) - Config-only change
4. **Step 4** (LSTM normalization) - Independent fix
5. **Step 5** (LightGBM stacking leakage) - Critical model fix
6. **Step 6** (Verify LSTM) - Quick check, no changes expected
7. **Step 7** (Backtest TP/high-low) - Complex backtest changes
8. **Step 8** (Torch seed) - Simple addition
9. **Step 9** (Tests) - Add test coverage
10. **Step 10** (Verification) - Final validation

## Dependencies
- Step 1 must complete before running full pipeline (Step 10)
- Steps 2-3 are independent and can be done in parallel
- Steps 4-8 are independent of each other
- Step 9 depends on Steps 1-8
- Step 10 depends on all previous steps

## Verification

### Unit Tests
`pytest tests/unit/test_labels.py -v` passes
`pytest tests/unit/test_lstm_model.py -v` passes
`pytest tests/unit/test_backtest.py -v` passes

### Integration Tests
`pytest tests/integration/ -v` passes
Coverage ≥ 70%

### End-to-End
`pixi run workflow --force` completes without errors
Backtest results show realistic metrics (not >100% returns with <10% drawdown)
Label distribution shows expected Hold percentage increase

### Code Quality
`pixi run lint` shows no errors
`pixi run format` makes no changes
No dead code detected

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Backtest TP logic introduces bugs | Medium | High | Add comprehensive unit tests; verify with known scenarios |
| LSTM normalization change affects convergence | Low | Medium | Monitor validation loss curves; compare before/after |
| Config changes break existing sessions | Low | Medium | Test with fresh data; document breaking changes |
| Triple barrier change reduces signal quality | Medium | Medium | Accept as correct methodology; document in thesis |
| Stacking fix reduces reported performance | High | Low | This is the goal - current performance is inflated |

## Notes for Implementation

### Backtest TP Implementation Details
The backtest should check barriers in this priority order:
1. Take Profit (high >= TP for long, low <= TP for short)
2. Stop Loss (low <= SL for long, high >= SL for short)
3. Trailing Stop (existing logic)
4. Signal reverse (existing logic)
5. Max hold time (existing logic)

When both TP and SL could hit in same bar (high >= TP AND low <= SL for long), the order of checks determines outcome. For conservative simulation, check SL first (assume worst case).

### LSTM Normalization Stats Persistence
The normalization stats are already saved to `lstm_norm_stats.npz` (line 215). For test predictions, `generate_test_predictions()` already loads these stats (lines 277-288 in stacking.py). The fix only needs to ensure val predictions also use train stats.

### Stacking Pipeline Flow After Fix
1. LightGBM: Train on train only → predict on val (OOS) → save model
2. LSTM: Train on train only → predict on val (OOS) → save model
3. Stacking: Train meta-learner on OOS predictions from both models
4. Test predictions: Load trained models, generate predictions on test set
5. Optional: Could add refit stage after meta-learner training to retrain base models on train+val for final test predictions (trade-off: more data vs. potential overfitting)

The current implementation is correct for step 4 - `generate_test_predictions()` uses the saved models without refitting, which is appropriate for true OOS evaluation.
