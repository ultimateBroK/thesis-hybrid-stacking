# Configuration Guide

> How to adjust settings to optimize your results. Read the **Features** section first so you understand what each parameter does before changing it.

---

## Part 1: Features — What You Need to Know

Before you change any settings, it helps to understand what each part of the pipeline does and how the parameters relate to each other. This section explains the building blocks.

---

### 1.1 Data Features

The model looks at OHLCV candles (Open, High, Low, Close, Volume) on the **1-hour timeframe**. Each candle summarizes one hour of trading.

**What you can change:**
| Parameter | Default | What It Does |
|-----------|---------|-------------|
| `timeframe` | `"1H"` | Candle size. `"1H"` = 1 hour. Other options: `"15T"` (15 min), `"4H"` (4 hours) |
| `start_date` / `end_date` | 2018-01-01 to 2026-03-31 | Date range for training and testing |

**Trade-offs:**
- **Smaller timeframe** (e.g., 15 min) → more data points, more noise, faster signals
- **Larger timeframe** (e.g., 4 hours) → fewer data points, less noise, slower signals

---

### 1.2 Technical Indicators

These are mathematical formulas applied to price data to identify trends and patterns.

| Indicator | Default Setting | What It Measures |
|-----------|----------------|-----------------|
| **EMA** (Exponential Moving Average) | Periods: 34, 89 | Trend direction. Short EMA above long EMA = uptrend. |
| **RSI** (Relative Strength Index) | Period: 14 | Overbought/oversold. Above 70 = overbought. Below 30 = oversold. |
| **MACD** (Moving Average Convergence Divergence) | Fast: 12, Slow: 26, Signal: 9 | Momentum. MACD above signal = bullish. Below = bearish. |
| **ATR** (Average True Range) | Period: 14 | Volatility. Higher ATR = bigger price moves. Used for TP/SL sizing. |

**You can toggle these on/off:**
```toml
[features]
use_technical = true    # EMA, RSI, MACD, ATR
use_pivots = true       # Daily pivot points
use_session = true      # Trading session encoding
use_spread = true       # Spread features
```

---

### 1.3 Microstructure Features

These analyze the "shape" of each candle and trading activity:

| Feature | What It Detects |
|---------|----------------|
| Bullish/Bearish Engulfing | Reversal patterns |
| Doji | Market indecision |
| Hammer | Potential bottom reversal |
| Shooting Star | Potential top reversal |
| Marubozu | Strong directional move |
| Volume Delta | Buying vs selling pressure |
| Body-to-Wick Ratio | How decisive the move was |

---

### 1.4 Session Features

The gold market trades 24 hours, but activity varies by time zone:

| Session | Hours (UTC) | Character |
|---------|-------------|-----------|
| Asia | 00:00 - 08:00 | Lower volatility, range-bound |
| London | 08:00 - 17:00 | Higher volatility, trending |
| NY PM | 17:00 - 21:00 | High volatility, news-driven |

The model uses session information to adjust its expectations.

---

### 1.5 Triple-Barrier Labels

Labels are generated using the **Triple-Barrier Method**. For each candle, the system sets three "barriers":

```
         Take Profit (TP) = Close + 1.5 x ATR
         ───────────────────────────────────────
         
              Price moves here...
         
         ───────────────────────────────────────
         Stop Loss (SL) = Close - 1.5 x ATR
         
         Time Limit: 20 bars (hours)
```

- If price hits **TP first** → Label = **Long** (+1)
- If price hits **SL first** → Label = **Short** (-1)
- If **20 hours pass** without hitting either → Label = **Hold** (0)

**Why ATR-based?** Because ATR measures actual volatility. In quiet markets, TP/SL are tighter. In volatile markets, they are wider. This adapts to market conditions automatically.

---

### 1.6 The Stacking Architecture

Two independent models make predictions, then a meta-learner combines them:

```
Raw Data
   │
   ├──► LightGBM (sees: technical features, lag features, session info)
   │         │
   │         └──► 3 probabilities: [P(Short), P(Hold), P(Long)]
   │
   ├──► LSTM (sees: 120 hours of OHLCV data as a sequence)
   │         │
   │         └──► 3 probabilities: [P(Short), P(Hold), P(Long)]
   │
   └──► Meta-Learner (Logistic Regression)
             │
             └──► Final prediction + confidence score
                   │
                   └──► If confidence < 0.6 → Hold
```

**Why two models?**
- **LightGBM** is good at finding patterns across many features at one point in time
- **LSTM** is good at finding patterns in sequences over time
- Together, they capture different aspects of market behavior

---

## Part 2: Configuration Parameters

All settings are in `config.toml` at the project root. Below is a guide to every section.

---

### 2.1 Data Settings (`[data]`)

```toml
[data]
timeframe = "1H"                    # Candle timeframe
start_date = "2018-01-01"           # Data start
end_date = "2026-03-31"             # Data end
market_tz = "America/New_York"      # Market timezone
symbol = "XAUUSD"                   # Trading instrument
tick_size = 0.01                    # Minimum price change
contract_size = 100                 # Ounces per contract
```

**Tips:**
- Do not change `market_tz`, `tick_size`, or `contract_size` unless you are using a different data source
- Change `start_date` / `end_date` to limit the data range (faster testing)

---

### 2.2 Splitting Settings (`[splitting]`)

```toml
[splitting]
train_start = "2018-01-01"
train_end = "2022-12-31 23:59:59"    # ~60% of data
val_start = "2023-01-01"
val_end = "2023-12-31 23:59:59"      # ~15% of data
test_start = "2024-01-01"
test_end = "2026-03-31 23:59:59"     # ~25% of data (out-of-sample)

purge_bars = 25       # Gap between train and val/test
embargo_bars = 10     # Extra buffer after training data
```

**What purge and embargo do:**
- **Purge** removes data near the boundary to prevent the model from "seeing" future information
- **Embargo** adds extra safety buffer after training data

**Tips:**
- Increase `purge_bars` to 50 for extra safety (at the cost of less training data)
- The train/val/test dates should match distinct market regimes

---

### 2.3 Feature Settings (`[features]`)

```toml
[features]
use_technical = true     # EMA, RSI, MACD, ATR
use_pivots = true        # Daily pivot points
use_session = true       # Asia/London/NY encoding
use_spread = true        # Spread metrics

ema_periods = [34, 89]   # EMA lookback periods
rsi_period = 14          # RSI lookback
macd_fast = 12           # MACD fast period
macd_slow = 26           # MACD slow period
macd_signal = 9          # MACD signal period
atr_period = 14          # ATR lookback

lag_periods = [1, 2, 3, 5, 10]  # Lookback for lag features
drop_high_corr = true           # Remove correlated features
correlation_threshold = 0.90    # Correlation cutoff
```

**Tips:**
- **More features** is not always better. Remove noisy features with `use_* = false`
- **Lag periods**: Adding more lags (e.g., `[1, 2, 3, 5, 10, 20]`) gives the tree model more history but can add noise
- **Correlation threshold**: Lower to 0.85 to remove more correlated features. Raise to 0.95 to keep more

---

### 2.4 Label Settings (`[labels]`)

```toml
[labels]
atr_multiplier_tp = 1.5   # TP = Close + 1.5 x ATR
atr_multiplier_sl = 1.5   # SL = Close - 1.5 x ATR
horizon_bars = 20          # Max wait (in candles / hours)
num_classes = 3            # Long, Hold, Short

# Alternative: use fixed pip values instead of ATR
use_fixed_pips = false
tp_pips = 20
sl_pips = 10
```

**Tips:**
- **Symmetric TP/SL (1.5/1.5)** is the default. You can make TP larger for trend-following (e.g., 2.0/1.0)
- **Larger ATR multipliers** → fewer but higher-quality signals
- **Smaller ATR multipliers** → more signals but lower quality
- **Longer horizon** (e.g., 30) → more Long/Short labels, fewer Holds
- **Shorter horizon** (e.g., 10) → more Holds, fewer directional signals
- **Fixed pips mode**: Set `use_fixed_pips = true` to use absolute pip distances instead of ATR-based

---

### 2.5 LightGBM Settings (`[models.tree]`)

```toml
[models.tree]
use_optuna = true         # Auto-tune hyperparameters
optuna_trials = 100       # Number of tuning experiments
optuna_timeout = 3600     # Max tuning time in seconds

# Manual hyperparameters (used when use_optuna = false)
num_leaves = 31           # Maximum tree complexity
max_depth = -1            # -1 = no limit
learning_rate = 0.05      # Step size per tree
n_estimators = 500        # Maximum number of trees
min_child_samples = 20    # Minimum samples per leaf
subsample = 0.8           # Fraction of data per tree
colsample_bytree = 0.8    # Fraction of features per tree
reg_alpha = 0.0           # L1 regularization
reg_lambda = 0.0          # L2 regularization
early_stopping_rounds = 50  # Stop if no improvement for 50 rounds

use_class_weights = true  # Handle imbalanced classes
```

**Tips:**
- **With Optuna** (`use_optuna = true`): Most parameters are auto-tuned. Just adjust `optuna_trials` (more trials = better but slower)
- **Without Optuna** (`use_optuna = false`): You control all hyperparameters manually
- **Reduce `optuna_trials`** to 20-30 for quick testing
- **Increase `num_leaves`** (e.g., 63) for more complex patterns (risk of overfitting)
- **Increase `reg_alpha` and `reg_lambda`** (e.g., 0.1) to reduce overfitting

---

### 2.6 LSTM Settings (`[models.lstm]`)

```toml
[models.lstm]
sequence_length = 120     # How many candles to look at (120 hours = 5 days)
hidden_size = 128         # Internal memory size
num_layers = 2            # Number of stacked LSTM layers
dropout = 0.3             # Dropout rate (prevents overfitting)
bidirectional = false     # Read sequences forward only

batch_size = 128          # Training batch size
epochs = 50               # Maximum training rounds
learning_rate = 0.001     # Step size for optimizer
weight_decay = 1e-5       # L2 regularization
patience = 10             # Early stopping patience
min_delta = 0.001         # Minimum improvement to count

device = "cpu"            # "cpu" or "cuda" (GPU)
num_workers = 4           # Data loading workers
```

**Tips:**
- **GPU users:** Set `device = "cuda"` for 5-10x faster training
- **Quick test:** Reduce `epochs` to 20 and `sequence_length` to 60
- **More memory:** Increase `hidden_size` to 256 (needs more GPU RAM)
- **Overfitting:** Increase `dropout` to 0.4-0.5 or increase `weight_decay`
- **Sequence length:** 120 = 5 trading days. Longer sequences capture more history but are harder to train

---

### 2.7 Stacking Settings (`[models.stacking]`)

```toml
[models.stacking]
meta_learner = "logistic_regression"  # How to combine base models
n_folds = 5                           # CV folds for stacking
stacking_purge = 25                   # Purge gap in stacking CV
stacking_embargo = 10                 # Embargo gap in stacking CV
calibrate_probabilities = true        # Calibrate output probabilities
calibration_method = "isotonic"       # "isotonic" or "platt"
```

**Meta-learner options:**
| Option | Description |
|--------|-------------|
| `logistic_regression` | Simple, stable, hard to overfit (recommended) |
| `ridge` | L2-regularized regression |
| `lasso` | L1-regularized (can zero out one model's contribution) |
| `elastic_net` | Mix of L1 and L2 |
| `lightgbm` | More flexible but can overfit with small data |

**Tips:**
- Stick with `logistic_regression` unless you have a specific reason to change
- `calibrate_probabilities = true` ensures the predicted probabilities are well-calibrated
- Increasing `n_folds` gives more training data for the meta-learner but takes longer

---

### 2.8 Backtest Settings (`[backtest.cfd]`)

```toml
[backtest.cfd]
initial_capital = 100000.0   # Starting money ($100,000)
leverage = 50                 # 50:1 leverage
risk_per_trade = 0.01        # Risk 1% of capital per trade
spread_pips = 2.0            # Broker spread (trading cost)
slippage_pips = 1.0          # Execution slippage
max_positions = 1            # Only 1 trade at a time
max_hold_bars = 100          # Close after 100 hours if no exit

# Risk management
use_trailing_stop = true
trailing_stop_atr_multiplier = 1.0
max_daily_loss_pct = 0.05    # Stop trading after losing 5% in one day
max_consecutive_losses = 5   # Pause after 5 consecutive losses
margin_call_level = 0.5      # Margin call at 50% level
stop_out_level = 0.2         # Force close at 20% level
```

**Tips:**
- **Lower `risk_per_trade`** (e.g., 0.005 = 0.5%) for more conservative trading
- **Higher `spread_pips`** makes results more realistic (most brokers charge 1.5-3 pips on XAU/USD)
- **`use_trailing_stop = true`** protects profits by following the price
- **`max_daily_loss_pct`** prevents catastrophic losses in a single day

---

### 2.9 Workflow Settings (`[workflow]`)

```toml
[workflow]
# Toggle each stage on/off
run_data_pipeline = true
run_feature_engineering = true
run_label_generation = true
run_data_splitting = true
run_lightgbm = true
run_lstm = true
run_stacking = true
run_backtest = true
run_reporting = true

force_rerun = false    # If true, ignore cached results
n_jobs = -1            # -1 = use all CPU cores
random_seed = 42       # For reproducibility
```

**Tips:**
- Set `run_* = false` for stages you want to skip
- Set `force_rerun = true` to regenerate everything from scratch
- Change `random_seed` for different results (but always note the seed for reproducibility)

---

### 2.10 Environment Variable Overrides

You can override any config setting using environment variables with the `THESIS_` prefix:

```bash
# Override LSTM device
export THESIS_MODELS_LSTM_DEVICE="cuda"

# Override random seed
export THESIS_WORKFLOW_RANDOM_SEED="123"

# Override number of Optuna trials
export THESIS_MODELS_TREE_OPTUNA_TRIALS="20"

# Then run the pipeline
pixi run workflow
```

**Format:** `THESIS_` + section name + `_` + key name, all uppercase, dots replaced by underscores.

---

## Part 3: Common Tuning Scenarios

### Scenario A: Quick Test (fast iteration)

```toml
[models.lstm]
epochs = 15
sequence_length = 60
batch_size = 256

[models.tree]
use_optuna = false
n_estimators = 100

[workflow]
run_data_pipeline = false    # If OHLCV already exists
run_feature_engineering = false  # If features already exist
```

### Scenario B: Maximum Accuracy (slow but thorough)

```toml
[models.lstm]
epochs = 100
hidden_size = 256
sequence_length = 120
dropout = 0.2

[models.tree]
use_optuna = true
optuna_trials = 200
```

### Scenario C: Conservative Trading

```toml
[labels]
atr_multiplier_tp = 2.0    # Larger TP target
atr_multiplier_sl = 1.0    # Tighter stop loss
horizon_bars = 30          # More patient

[backtest.cfd]
risk_per_trade = 0.005     # 0.5% risk
max_daily_loss_pct = 0.03  # 3% daily limit
```

### Scenario D: Aggressive Trading

```toml
[labels]
atr_multiplier_tp = 1.0    # Quick profits
atr_multiplier_sl = 1.5    # Wider stops
horizon_bars = 10          # Short holding period

[backtest.cfd]
risk_per_trade = 0.02      # 2% risk
max_positions = 2          # Allow 2 simultaneous trades
```
