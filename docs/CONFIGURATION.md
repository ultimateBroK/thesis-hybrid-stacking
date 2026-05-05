# Configuration Guide

This project exposes only the parameters that matter for a student ML thesis.
Most financial-engineering details stay as code defaults so the report focuses
on reproducible modelling, not overfitting a trading system.

## Scope

The core experiment is:

1. Build OHLCV bars from raw data.
2. Create deterministic technical features.
3. Generate a 3-class target: `Short`, `Hold`, `Long`.
4. Validate with walk-forward time-series splits and purge/embargo gaps.
5. Train a compact hybrid model: GRU temporal embedding + LightGBM classifier.
6. Report ML metrics first: accuracy, F1, baseline comparison, confusion matrix.
7. Use backtest metrics only as an application demo.

## Model Inputs

GRU sequence input (19 features, multiclass objective):

```toml
feature_cols = ["log_returns", "return_1h", "return_4h", "atr_14",
    "close_vs_ema_34", "ema34_vs_ema89", "price_position_20",
    "candle_body_ratio", "macd_hist", "rsi_14", "atr_percentile",
    "adx_14", "ema_slope_20", "regime_strength", "volume_zscore_20",
    "open_norm", "high_norm", "low_norm", "close_norm"]
input_size = 19
hidden_size = 64
```

LightGBM static input (22 features):

```text
Trend: ema34_vs_ema89, close_vs_ema_34, adx_14, ema_slope_20, regime_strength
Momentum: return_1h, return_4h, macd_hist, rsi_14
Volatility: atr_14, atr_percentile, high_low_range_20
Position: price_dist_ratio, price_position_20, pivot_position
Candle: candle_body_ratio, upper_wick_ratio, lower_wick_ratio
Session: sess_london, sess_overlap
Volume: volume_zscore_20
```

With PCA-reduced GRU hidden states (16 dims), the hybrid feature space is
`16 GRU PCA features + 22 tabular features = 38 features`.

---

## Section Reference

All defaults below match `config.toml` and `src/thesis/config.py` dataclasses.

### `[data]`

| Parameter | Default | Description |
| --- | --- | --- |
| `symbol` | `"XAUUSD"` | Display symbol used in session names and reports. |
| `timeframe` | `"1H"` | Bar timeframe. |
| `market_tz` | `"America/New_York"` | Timezone for session-aware feature engineering. |
| `start_date` | `"2018-01-01"` | Inclusive data start date. |
| `end_date` | `"2026-04-30"` | Inclusive data end date. |
| `tick_size` | `0.01` | Minimum price movement. |
| `contract_size` | `100` | Units per trading lot (for backtest demo). |

### `[validation]`

| Parameter | Default | Description |
| --- | --- | --- |
| `method` | `"sliding"` | Validation method: `"sliding"` (walk-forward) or `"static"` (fixed split). |
| `train_window_bars` | `26280` | Training window size (~3 years of H1 bars). |
| `test_window_bars` | `4380` | Test window size (~6 months of H1 bars). |
| `step_bars` | `4380` | Step between consecutive windows. Equals `test_window_bars` for non-overlapping folds. |
| `purge_bars` | `25` | Bars removed at the train/test boundary to prevent label leakage. |
| `embargo_bars` | `50` | Additional gap after purge for extra safety. |
| `min_train_bars` | `10000` | Minimum training bars required to produce a window. Windows below this are skipped. |
| `oof_ensemble` | `true` | Aggregate out-of-fold predictions across all walk-forward windows. |

Walk-forward window sizes are **bar-based**, not fixed calendar durations. For
example, `4380` H1 bars is only approximately six months; weekends, holidays,
and missing broker data can change the actual calendar span.

### `[features]`

| Parameter | Default | Description |
| --- | --- | --- |
| `rsi_period` | `14` | RSI lookback period. |
| `atr_period` | `14` | ATR lookback period. |
| `adx_period` | `14` | ADX lookback period for trend strength. |
| `ema_slope_period` | `20` | EMA span whose 5-bar rate-of-change feeds regime detection. |
| `macd_fast` | `12` | MACD fast EMA span. |
| `macd_slow` | `26` | MACD slow EMA span. |
| `macd_signal` | `9` | MACD signal EMA span. |
| `correlation_threshold` | `0.75` | Threshold for correlation-based feature filtering. |
| `static_feature_cols` | *(22 features, see below)* | Whitelist of tabular features consumed by LightGBM. |

Default `static_feature_cols` (22 features in 7 groups):

```toml
static_feature_cols = [
  # --- Trend ---
  "ema34_vs_ema89",
  "close_vs_ema_34",
  "adx_14",
  "ema_slope_20",
  "regime_strength",
  # --- Momentum ---
  "return_1h",
  "return_4h",
  "macd_hist",
  "rsi_14",
  # --- Volatility / Regime ---
  "atr_14",
  "atr_percentile",
  "high_low_range_20",
  # --- Position / Location ---
  "price_dist_ratio",
  "price_position_20",
  "pivot_position",
  # --- Candle Structure ---
  "candle_body_ratio",
  "upper_wick_ratio",
  "lower_wick_ratio",
  # --- Session ---
  "sess_london",
  "sess_overlap",
  # --- Volume / Activity ---
  "volume_zscore_20",
]
```

New regime features added relative to v1 baseline:
- `adx_14` — Average Directional Index (trend strength, non-directional)
- `ema_slope_20` — 5-bar slope of EMA(20), captures acceleration/deceleration
- `regime_strength` — composite ADX × EMA_slope interaction term
- Candle structure features: `candle_body_ratio`, `upper_wick_ratio`, `lower_wick_ratio`
- Position features: `price_dist_ratio`, `pivot_position`, `atr_percentile`

### `[labels]`

| Parameter | Default | Description |
| --- | --- | --- |
| `atr_multiplier` | `2.5` | ATR multiplier for triple-barrier width. Defines target classes. |
| `horizon_bars` | `24` | Maximum bars before forced exit if no barrier hit. |

### `[model]`

| Parameter | Default | Description |
| --- | --- | --- |
| `architecture` | `"hybrid"` | Model architecture: `"static"` (LightGBM only) or `"hybrid"` (GRU embedding + LightGBM). |
| `objective` | `"multiclass"` | LightGBM objective: `"multiclass"` (3-class Short/Hold/Long) or `"regression"` (continuous returns). This is **independent** of `[gru].objective`. |
| `static_expanded` | `false` | If true, use ALL features (not just `static_feature_cols`) for the static baseline. |
| `num_leaves` | `31` | LightGBM leaf count. Controls tree complexity. |
| `max_depth` | `6` | LightGBM max tree depth. |
| `learning_rate` | `0.02` | LightGBM learning rate. |
| `n_estimators` | `500` | Maximum boosting iterations. |
| `min_child_samples` | `50` | Minimum samples per leaf. Higher = more conservative. |
| `subsample` | `0.80` | Row subsample ratio per iteration. |
| `subsample_freq` | `5` | Frequency of subsampling (every N iterations). |
| `feature_fraction` | `0.70` | Feature subsample ratio per iteration. |
| `reg_alpha` | `0.05` | L1 regularization. |
| `reg_lambda` | `5.0` | L2 regularization. |
| `early_stopping_rounds` | `40` | Stop training if validation metric does not improve for this many rounds. |

### `[gru]`

| Parameter | Default | Description |
| --- | --- | --- |
| `objective` | `"multiclass"` | GRU training objective: `"multiclass"` (3-class focal loss) or experimental `"regression"` (MSE on forward returns). **Independent** of `[model].objective`. |
| `feature_cols` | *(19 features, see below)* | Input features for the GRU sequence. |
| `input_size` | `19` | Number of input features (must match `feature_cols` length). |
| `hidden_size` | `64` | GRU hidden state dimension. Stable default after 128 degraded OOS class signal. |
| `num_layers` | `2` | Number of stacked GRU layers. |
| `sequence_length` | `48` | Number of bars in each input sequence. |
| `dropout` | `0.3` | Dropout between GRU layers (variational dropout applied across timesteps). |
| `bidirectional` | `false` | If true, GRU reads both forward and backward. Disabled by default to prevent look-ahead bias. |
| `gradient_accumulation_steps` | `1` | Number of forward passes before gradient update (simulates larger batch sizes). |
| `learning_rate` | `0.0005` | Adam optimizer learning rate. |
| `batch_size` | `256` | Training batch size. |
| `epochs` | `100` | Maximum training epochs. Up from v1's 50 — cosine annealing extends useful training. |
| `patience` | `20` | Early-stopping patience (epochs without validation improvement). |
| `min_epochs` | `10` | Minimum epochs before early-stopping can trigger. |
| `warmup_epochs` | `3` | Linear LR warmup before cosine annealing takes over. |
| `focal_loss_gamma` | `2.0` | Gamma parameter for focal loss (only used when `objective = "multiclass"`). |
| `contrastive_pretrain_epochs` | `10` | Epoches of contrastive pre-training (triplet loss). Set to 0 to disable. |
| `temperature_scaling` | `false` | Apply temperature scaling calibration to GRU probabilities. |
| `pca_components` | `16` | PCA dimension for GRU hidden states before LightGBM. Set to 0 to disable. |

Default `feature_cols` (19 features):

```toml
feature_cols = [
  "log_returns", "return_1h", "return_4h", "atr_14",
  "close_vs_ema_34", "ema34_vs_ema89", "price_position_20",
  "candle_body_ratio", "macd_hist", "rsi_14", "atr_percentile",
  "adx_14", "ema_slope_20", "regime_strength", "volume_zscore_20",
  "open_norm", "high_norm", "low_norm", "close_norm",
]
```

The last 4 columns (`open_norm` through `close_norm`) are **raw OHLCV rolling z-scores**
— normalized price bars over a 20-bar lookback. These give the GRU direct access to
price levels while avoiding absolute price leakage.

**LR Schedule:** Cosine annealing with warm restarts (`T_0=10`, `T_mult=2`) replaces
v1's fixed LR. A 3-epoch linear warmup precedes the schedule. Gradient clipping
(max norm 1.0) and plateau detection (diagnostic only, patience 5) improve training
stability on non-stationary financial data.

### `[backtest]`

| Parameter | Default | Description |
| --- | --- | --- |
| `initial_capital` | `10000.0` | Starting equity in account currency. |
| `leverage` | `10` | Margin leverage (margin = 1/leverage). |
| `spread_ticks` | `35` | Spread in ticks applied on entry/exit. |
| `slippage_ticks` | `5` | Slippage in ticks applied on execution. |
| `commission_per_lot` | `10.0` | Commission per lot per trade. |
| `atr_stop_multiplier` | `2.0` | ATR multiplier for stop-loss distance. **Must match** `[labels] atr_sl_multiplier`. |
| `atr_tp_multiplier` | `2.0` | ATR multiplier for take-profit distance (`0` = disabled). **Must match** `[labels] atr_tp_multiplier`. |
| `lots_per_trade` | `0.01` | Fixed lot size after confidence filtering. |
| `min_lots` | `0.01` | Minimum lot safety bound. |
| `max_lots` | `0.5` | Maximum lot safety bound; not used for confidence amplification by default. |
| `confidence_threshold` | `0.50` | Minimum predicted probability to open a trade (`0` = disabled). |
| `min_bars_between_trades` | `6` | **Trade cooldown**: minimum bars between position exit and next entry. Reduces overtrading and correlation between consecutive trades. |
| `max_drawdown_cutoff` | `0.30` | Circuit breaker: stop if equity drops below `peak * (1 - cutoff)`. |
| `dd_cooldown_bars` | `12` | Bars to pause trading after a drawdown cutoff breach. |
| `max_open_positions` | `1` | Maximum simultaneous open positions. |
| `daily_loss_limit` | `0.03` | Stop trading for the day after a `-N` equity drawdown (e.g. 3%). |

**Position sizing rule:** Confidence filters entries only. Lot size stays fixed at
`lots_per_trade` and is clamped to `[min_lots, max_lots]`. Confidence-based lot
amplification is disabled because it magnified high-confidence wrong predictions
in the latest OOS run.

### `[workflow]`

| Parameter | Default | Description |
| --- | --- | --- |
| `force_rerun` | `false` | Ignore cache and rerun all pipeline stages. |
| `random_seed` | `2024` | Global random seed for reproducibility. |
| `n_jobs` | `-1` | Parallel worker count (`-1` = all CPUs). |

### `[paths]`

| Parameter | Default | Description |
| --- | --- | --- |
| `data_raw` | `"data/raw/XAUUSD"` | Raw tick/data directory. |
| `data_processed` | `"data/processed"` | Processed parquet output directory. |
| `ohlcv` | `"data/processed/ohlcv.parquet"` | OHLCV bars parquet. |
| `features` | `"data/processed/features.parquet"` | Engineered features parquet. |
| `labels` | `"data/processed/labels.parquet"` | Label parquet. |
| `train_data` | `"data/processed/train.parquet"` | Training split. |
| `val_data` | `"data/processed/val.parquet"` | Validation split. |
| `test_data` | `"data/processed/test.parquet"` | Test split. |
| `model` | `"models/lightgbm_model.pkl"` | LightGBM model artifact. |
| `gru_model` | `"models/gru_model.pt"` | GRU model artifact. |
| `predictions` | `"data/predictions/final_predictions.parquet"` | Final predictions. |
| `backtest_results` | `"results/backtest_results.json"` | Backtest output. |
| `report` | `"results/thesis_report.md"` | Generated report. |

---

## Parameters Worth Changing

Use these for experiments:

| Section | Parameter | Why it matters |
| --- | --- | --- |
| `validation` | `train_window_bars`, `test_window_bars` | Controls time-series evaluation stability. |
| `labels` | `atr_tp_multiplier`, `atr_sl_multiplier`, `horizon_bars` | Defines the target classes. This changes the ML problem. |
| `model` | `architecture` | Set to `"hybrid"` for the default architecture. |
| `model` | `num_leaves`, `max_depth`, `n_estimators` | Controls LightGBM capacity and overfitting. |
| `gru` | `objective` | Keep `"multiclass"` as stable default; use `"regression"` only as an experiment. |
| `gru` | `hidden_size`, `sequence_length`, `epochs` | Controls temporal model capacity and runtime. |
| `gru` | `pca_components` | Controls dimensionality of GRU features passed to LightGBM. |
| `backtest` | `confidence_threshold`, `lots_per_trade`, `min_bars_between_trades` | Demo-only risk/filter controls. Do not use them to claim model quality. |

## Default Experiment Profile

The default `config.toml` is intentionally conservative:

```toml
[model]
architecture = "hybrid"
objective = "multiclass"
num_leaves = 31
max_depth = 6
n_estimators = 500
learning_rate = 0.02

[gru]
objective = "multiclass"       # Stable default after regression degraded OOS signal
hidden_size = 64
epochs = 100
patience = 20
batch_size = 256
pca_components = 16            # PCA reduction before LightGBM

[validation]
method = "sliding"

[backtest]
min_bars_between_trades = 6    # Trade cooldown reduces overtrading
lots_per_trade = 0.01          # Fixed conservative risk
max_lots = 0.5                 # Safety cap; no confidence amplification by default
```

This gives faster runs and more repeatable comparisons.

## Evaluation Rules

Treat a result as useful only if it beats simple baselines:

| Metric | Minimum expectation |
| --- | --- |
| Exact accuracy | Higher than majority-class baseline. |
| Macro F1 | Better than predicting only `Hold`. |
| Directional accuracy | Higher than 50% on non-Hold predictions. |
| High-confidence accuracy | Higher than full-sample accuracy. |

Backtest return is secondary. A profitable backtest with weak ML metrics is not
a reliable thesis result; it is likely noise or overfitting.
