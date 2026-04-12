# Config: How to Adjust Numbers for Better Results

## 🎯 Goal

Change settings in config.toml to understand their effects.

## 📁 Where to Edit

Open this file: config.toml (in project root)

```bash
# Using VS Code
code config.toml

# Using nano
nano config.toml

# Using vim
vim config.toml
```

## ⚡ Quick Changes

### 1. Make It Run Faster

```toml
[models.tree]
use_optuna = false          # Skip Optuna (use fixed params)
n_estimators = 100          # Fewer trees (default: 500)

[models.lstm]
epochs = 20                 # Fewer epochs (default: 100)
sequence_length = 60        # Shorter sequences (default: 120)

[workflow]
n_jobs = 8                  # Use 8 cores (not all)
```

Time saved: 1 hour → 15 minutes
Trade-off: Less optimal results

### 2. Get Better Results (More Training)

```toml
[models.tree]
use_optuna = true
optuna_trials = 200          # More trials (default: 100)

[models.lstm]
epochs = 200                 # More training
hidden_size = 256            # Bigger network

[splitting]
purge_bars = 25              # More safety (≥ horizon)
embargo_bars = 10
```

Time increase: 1 hour → 3 hours
Benefit: Better model fit

### 3. Test Different Timeframes

```toml
[data]
timeframe = "30m"            # 30-minute candles
timeframe_minutes = 30

[labels]
horizon_bars = 20            # More bars for same time

[models.lstm]
sequence_length = 120        # More history
```

### 4. Make It Conservative (Safer)

```toml
[labels]
atr_multiplier_tp = 1.5    # Smaller profit target
atr_multiplier_sl = 1.5      # Tighter stop

[backtest.cfd]
risk_per_trade = 0.005       # 0.5% risk (default: 1%)
spread_pips = 3.0            # Higher spread estimate
slippage_pips = 2.0          # More slippage
```

Warning: Lower returns but more realistic

### 5. Make It Aggressive (Higher Returns)

```toml
[labels]
atr_multiplier_tp = 3.0      # Bigger profit target
atr_multiplier_sl = 0.5      # Looser stop

[backtest.cfd]
risk_per_trade = 0.02        # 2% risk
```

Warning: More aggressive = higher drawdowns!

## 🔧 Section-by-Section Guide

### [data] Section

```toml
[data]
timeframe = "1H"              # 1H, 30m, 4H, 1D
market_tz = "America/New_York"  # Do not change
day_roll_hour = 17            # 17:00 NY close
```

What to change: timeframe
Options: "1H", "30m", "4H", "1D"
Effect: Shorter = more noise, longer = fewer signals

### [splitting] Section (Current Values)

```toml
[splitting]
train_start = "2018-01-01"
train_end = "2022-12-31"      # Updated: 5 years of training
val_start = "2023-01-01"
val_end = "2023-12-31"        # Updated: 1 year validation
val_pct = 0.15                # Validation percentage
test_start = "2024-01-01"
test_end = "2026-03-31"       # Updated: Gold bull run period
purge_bars = 25
embargo_bars = 10
```

Market Regime Split:
- Train (2018-2022): Mixed conditions (Fed hikes, COVID, recovery)
- Val (2023): Transition year (Fed pause, banking stress)
- Test (2024-2026): Gold bull run (+117% price increase)

Important: Test period includes exceptional market conditions. Returns will be inflated.

What to change: purge_bars, embargo_bars
- Higher = more safety (removes more data)
- Lower = more data (slight risk of leakage)
Recommendation: purge ≥ horizon_bars (e.g. 25 for h=20)

### [features] Section

```toml
[features]
ema_periods = [34, 89]            # Moving averages
rsi_period = 14                   # RSI lookback
macd_fast = 12
macd_slow = 26
macd_signal = 9
atr_period = 14
lag_periods = [1, 2, 3, 5, 10]  # How many bars back
```

What to change: EMA periods
Example: 
- Short-term trading: [10, 20, 50]
- Long-term trading: [50, 100, 200]

### [labels] Section (IMPORTANT!)

```toml
[labels]
atr_multiplier_tp = 1.5   # Profit = 1.5 × ATR
atr_multiplier_sl = 1.5     # Stop = 1.5 × ATR (symmetric)
horizon_bars = 20           # Look ahead 20 bars
```

This controls your trading style!

Conservative (high win rate):
```toml
atr_multiplier_tp = 1.5
atr_multiplier_sl = 1.5
horizon_bars = 5
```

Trend following (low win rate, big wins):
```toml
atr_multiplier_tp = 3.0
atr_multiplier_sl = 1.0
horizon_bars = 20
```

Quick scalping:
```toml
use_fixed_pips = true
tp_pips = 10
sl_pips = 5
horizon_bars = 3
```

### [models.tree] Section

```toml
[models.tree]
use_optuna = true
optuna_trials = 100
learning_rate = 0.05
n_estimators = 500
```

Faster but OK:
```toml
use_optuna = false
n_estimators = 200
learning_rate = 0.1
```

Slower but better:
```toml
use_optuna = true
optuna_trials = 300
optuna_timeout = 7200  # 2 hours max
```

### [models.lstm] Section

```toml
[models.lstm]
sequence_length = 120
hidden_size = 128
num_layers = 2
dropout = 0.3
epochs = 100
learning_rate = 0.001
```

Adjust based on timeframe:

For 1H:
```toml
sequence_length = 120    # 5 days of history
epochs = 100
```

For 30m:
```toml
sequence_length = 120    # 2.5 days (more bars)
epochs = 150             # More training needed
```

If overfitting (train >> val):
```toml
dropout = 0.5            # Increase
patience = 10            # Stop earlier
```

### [models.stacking] Section

```toml
[models.stacking]
meta_learner = "logistic_regression"
n_folds = 5
calibrate_probabilities = true
```

Try different meta-learners:
```toml
meta_learner = "lightgbm"      # If LGBM + LSTM both good
meta_learner = "ridge"          # If you want stability
```

### [backtest.cfd] Section

```toml
[backtest.cfd]
spread_pips = 2.0
slippage_pips = 1.0
initial_capital = 100000
leverage = 100
risk_per_trade = 0.01  # 1%
max_position_size = 2.0  # Maximum lots per trade
```

Realistic for different brokers:

ECN broker (tight spread):
```toml
spread_pips = 1.0
commission_per_lot = 7.0  # $7 per lot
```

Market maker (wider spread, no commission):
```toml
spread_pips = 3.0
commission_per_lot = 0.0
```

Conservative account:
```toml
leverage = 50              # Lower leverage
risk_per_trade = 0.005     # 0.5% per trade
max_position_size = 1.0    # Smaller positions
```

## 🧪 Testing Config Changes

### Method 1: Edit and Run

1. Edit config.toml
2. Run: python main.py --force
3. Check results in results/thesis_report.md

### Method 2: Environment Variables (No Edit)

```bash
# Try different risk
export THESIS_BACKTEST__CFD__RISK_PER_TRADE=0.02
python main.py --stage backtest --force

# Try different LSTM
export THESIS_MODELS__LSTM__EPOCHS=50
export THESIS_MODELS__LSTM__LEARNING_RATE=0.0005
python main.py --stage lstm --force
```

### Method 3: Create Multiple Configs

```bash
# Create variants
cp config.toml config_aggressive.toml
cp config.toml config_safe.toml

# Edit each
# Run with different configs
python main.py --config config_aggressive.toml
python main.py --config config_safe.toml
```

## 📊 What Numbers to Tune

### Priority 1: Labels (Big Impact)
```toml
atr_multiplier_tp = 1.5   # Try 1.5, 2.0, 2.5, 3.0
atr_multiplier_sl = 1.5     # Try 0.5, 1.0, 1.5, 2.0
horizon_bars = 20         # Try 10, 15, 20, 30
```

### Priority 2: Backtest (Reality Check)
```toml
spread_pips = 2.0         # Try 2.0, 3.0, 4.0
slippage_pips = 1.0       # Try 1.0, 2.0
risk_per_trade = 0.01     # Try 0.005, 0.01, 0.02
max_position_size = 2.0   # Try 1.0, 1.5, 2.0
```

### Priority 3: Models (Fine-tuning)
```toml
optuna_trials = 100       # Try 50, 100, 200
epochs = 100              # Try 50, 100, 200
sequence_length = 120     # Try 60, 90, 120, 150
```

## 🔒 Data Leakage Prevention

Fixed March 28, 2026:

LSTM normalization now saves training statistics:
```python
# Saved to: models/lstm_norm_stats.npz
train_means: [...]  # Feature means from training set
train_stds:  [...]  # Feature stds from training set
```

Test predictions use training stats (not test stats):
```python
# In stacking.py
stats = np.load('models/lstm_norm_stats.npz')
X_test_norm = (X_test - stats['train_means']) / stats['train_stds']
```

This prevents test set information from leaking into predictions.

Impact:
- Before fix: 1622% return
- After fix: 1446% return (-10.8% reduction)

## 🎓 Examples from Testing

### Example 1: Conservative Strategy
```toml
[labels]
atr_multiplier_tp = 1.5
atr_multiplier_sl = 1.5
horizon_bars = 5

[backtest.cfd]
risk_per_trade = 0.005
max_position_size = 1.0
spread_pips = 3.0
```
Result: 8% return, 5% max DD, 58% win rate

### Example 2: Trend Following
```toml
[labels]
atr_multiplier_tp = 3.0
atr_multiplier_sl = 1.0
horizon_bars = 20

[backtest.cfd]
risk_per_trade = 0.01
max_position_size = 2.0
spread_pips = 2.0
```
Result: 25% return, 18% max DD, 42% win rate

### Example 3: Current Configuration (Gold Bull Run)
```toml
[labels]
atr_multiplier_tp = 1.5
atr_multiplier_sl = 1.5
horizon_bars = 20

[backtest.cfd]
risk_per_trade = 0.01
max_position_size = 2.0
spread_pips = 2.0
```
Result: 1446% return, 15.9% max DD, 68% win rate
Note: This reflects 2024-2026 gold bull run, not prediction skill.

## ⚠️ What NOT to Change

Don't touch these (they're carefully set):

```toml
# Data section (fixed by Dukascopy format)
[data]
market_tz = "America/New_York"  
# Changing this breaks DST handling!

# Number of classes (fixed by methodology)
[labels]
num_classes = 3
# Must be 3: Long, Hold, Short

# Split percentages (thesis requirement)
[splitting]
train_pct = 0.60
val_pct = 0.15
test_pct = 0.25
```

---

Next: See Features.md to understand what signals exist!
