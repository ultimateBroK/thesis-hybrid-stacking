# Features: What Signals Are Used

## 🎯 Goal

Understand what information the model sees before trading.

## 📊 Feature Categories

The model uses 6 types of features:
1. Technical indicators
2. Pivot points
3. Session times
4. Spread info
5. Lag features
6. Price distances

## 1️⃣ Technical Indicators (Most Important)

### EMA (Exponential Moving Average)

What: Average price over time, giving more weight to recent prices

Features created:
- ema_20 - 20-bar EMA
- ema_50 - 50-bar EMA  
- ema_200 - 200-bar EMA

How to read:
```
Price > EMA_20 = Bullish (uptrend)
Price < EMA_20 = Bearish (downtrend)
EMA_20 > EMA_50 > EMA_200 = Strong uptrend
```

Example:
```
Gold: 2,350 USD
EMA_20: 2,320 USD  ← Price above EMA = bullish
EMA_50: 2,300 USD
EMA_200: 2,200 USD
```

### RSI (Relative Strength Index)

What: Measures speed of price changes (0-100)

Feature: rsi_14

How to read:
```
RSI > 70 = Overbought (might drop)
RSI < 30 = Oversold (might rise)
RSI = 50 = Neutral
```

Example:
```
RSI = 75  ← Overbought, expect pullback
RSI = 25  ← Oversold, expect bounce
RSI = 45  ← Neutral, no signal
```

### MACD (Moving Average Convergence Divergence)

What: Shows relationship between two EMAs

Features created:
- macd - MACD line
- macd_signal - Signal line
- macd_hist - Histogram (MACD - Signal)

How to read:
```
MACD > Signal  = Buy signal
MACD < Signal  = Sell signal
Histogram ↑    = Momentum increasing
Histogram ↓    = Momentum decreasing
```

Example:
```
MACD: 1.5
Signal: 0.8
Hist: 0.7  ← Positive and growing = bullish
```

### ATR (Average True Range)

What: Measures volatility (how much price moves)

Feature: atr_14

How to read:
```
High ATR = Volatile (big moves, risky)
Low ATR = Quiet (small moves, calm)
```

Example:
```
ATR = 15 USD  ← High volatility
ATR = 5 USD   ← Low volatility
```

Used for: Triple-Barrier labels!
- TP = 2 × ATR = 30 USD profit target
- SL = 1 × ATR = 15 USD stop loss

### Bollinger Bands

What: Price envelope showing volatility channels

Features created:
- bb_upper - Upper band
- bb_middle - Middle band (20 EMA)
- bb_lower - Lower band
- bb_width - Band width (volatility)
- bb_position - Where price is in bands (0-1)

How to read:
```
Price near upper = Overbought
Price near lower = Oversold
bb_position > 0.8 = Near top
bb_position < 0.2 = Near bottom
```

Example:
```
Price: 2,350 USD
BB Upper: 2,380 USD
BB Lower: 2,320 USD
BB Position: 0.5  ← Middle, neutral
```

## 2️⃣ Pivot Points (Support/Resistance)

What: Levels based on yesterday's high, low, close

Features created:
- pivot_prev - Yesterday's pivot
- r1_prev - Yesterday's resistance 1
- s1_prev - Yesterday's support 1
- high_prev - Yesterday's high
- low_prev - Yesterday's low
- dist_pivot - Distance to pivot (%)
- dist_r1 - Distance to resistance (%)
- dist_s1 - Distance to support (%)
- inside_prev_range - Is price inside yesterday's range?
- breakout_high - Did we break yesterday's high?
- breakout_low - Did we break yesterday's low?

How to read:
```
Price > R1 = Strong bullish (broke resistance)
Price < S1 = Strong bearish (broke support)
Inside range = Consolidation
Breakout = New trend starting
```

Example:
```
Yesterday: High=2,350 Low=2,300 Close=2,320
Today price: 2,360
Result: breakout_high = 1 (true), bullish!
```

## 3️⃣ Session Encoding (Time of Day)

What: Which trading session is active

Features created:
- session_asia - 00:00-08:00 NY time (1 if active, else 0)
- session_london - 08:00-17:00 NY time
- session_ny_pm - 17:00-21:00 NY time
- day_of_week - 0=Monday, 4=Friday
- is_monday - Is it Monday? (1 or 0)
- is_friday - Is it Friday? (1 or 0)

How to read:
```
session_london = 1  ← London active, high volume
session_ny_pm = 1   ← NY afternoon, lower volume
is_friday = 1       ← Friday, watch for weekend gap
```

Why it matters:
- London session = Most volume, big moves
- Asia session = Quiet, range-bound
- Friday afternoon = Traders close positions

## 4️⃣ Spread Features

What: Trading cost info

Features created:
- spread_pct - Spread as % of price
- spread_ma_20 - 20-bar average spread
- spread_ratio - Current / average spread

How to read:
```
Spread Ratio > 1.5 = Expensive to trade (skip)
Spread Ratio < 0.8 = Cheap to trade (good)
```

Example:
```
Current spread: 2 USD
20-bar average: 1.5 USD
Spread ratio: 1.33  ← Slightly expensive
```

## 5️⃣ Lag Features (Historical Data)

What: Previous bars' data (for tree models)

Features created:
- close_lag_1 - Price 1 bar ago
- close_lag_2 - Price 2 bars ago
- close_lag_3, close_lag_5, close_lag_10
- returns_lag_1 - Return 1 bar ago (%)
- returns_lag_2, returns_lag_3, returns_lag_5, returns_lag_10
- high_lag_1, high_lag_2, etc.
- low_lag_1, low_lag_2, etc.
- volume_lag_1, volume_lag_2, volume_lag_3

How to read:
```
Returns_lag_1 = 0.5%  ← Previous bar went up 0.5%
Returns_lag_5 = -2%   ← 5 bars ago dropped 2%
```

Why: Tree models need explicit history (unlike LSTM)

Implementation note: Uses .shift() to prevent lookahead bias. No future information leaks into features.

## 6️⃣ Price Distances (Relative Position)

What: Where current price is relative to indicators

Features created:
- close_dist_ema_20 - Distance from 20 EMA (%)
- close_dist_ema_50 - Distance from 50 EMA (%)
- close_dist_ema_200 - Distance from 200 EMA (%)

How to read:
```
Dist = 0.02  ← Price is 2% above EMA (extended)
Dist = -0.01 ← Price is 1% below EMA (cheap)
Dist = 0.00  ← Price at EMA (fair value)
```

Example:
```
Price: 2,350 USD
EMA_20: 2,320 USD
Distance: (2350-2320)/2320 = 0.0129 = 1.29%
```

## 📋 Complete Feature List Example

For one candle (timestamp = 2023-06-15 14:00):

```
┌────────────────────────────────────────┐
│  OHLCV Data                            │
├────────────────────────────────────────┤
│  open:         1,950.50 USD            │
│  high:         1,955.00 USD            │
│  low:          1,948.20 USD            │
│  close:        1,952.30 USD            │
│  volume:       12,500                  │
└────────────────────────────────────────┘

┌────────────────────────────────────────┐
│  Technical Indicators                  │
├────────────────────────────────────────┤
│  ema_20:       1,948.50 USD            │
│  ema_50:       1,940.20 USD            │
│  ema_200:      1,850.00 USD            │
│  rsi_14:       62.5                    │
│  macd:         2.3                     │
│  macd_signal:  1.8                     │
│  macd_hist:    0.5                     │
│  atr_14:       8.50 USD                │
│  bb_upper:     1,965.00 USD            │
│  bb_lower:     1,931.00 USD            │
│  bb_position:  0.62                    │
└────────────────────────────────────────┘

┌────────────────────────────────────────┐
│  Pivot Points                          │
├────────────────────────────────────────┤
│  pivot_prev:   1,945.00 USD            │
│  r1_prev:      1,960.00 USD            │
│  s1_prev:      1,930.00 USD            │
│  dist_pivot:   0.0037 (0.37%)          │
│  breakout_high: 0 (false)              │
└────────────────────────────────────────┘

┌────────────────────────────────────────┐
│  Session and Time                      │
├────────────────────────────────────────┤
│  session_asia:   0                     │
│  session_london: 1                   │  ← London open, high volume
│  session_ny_pm:  0                     │
│  day_of_week:    3 (Thursday)          │
│  is_friday:      0                     │
└────────────────────────────────────────┘

┌────────────────────────────────────────┐
│  Lag Features                          │
├────────────────────────────────────────┤
│  close_lag_1:    1,948.80 USD          │
│  returns_lag_1:  0.0018 (0.18%)       │
│  close_lag_5:    1,935.00 USD          │
│  returns_lag_5:  0.0089 (0.89%)       │
└────────────────────────────────────────┘

┌────────────────────────────────────────┐
│  Price Distances                       │
├────────────────────────────────────────┤
│  close_dist_ema_20:  0.0019 (0.19%)    │
│  close_dist_ema_50:  0.0062 (0.62%)    │
└────────────────────────────────────────┘
```

Total: ~45 features per candle

## 🎯 How the Model Uses These

### LightGBM (Tree Model)

```
IF rsi_14 < 30 AND close_dist_ema_20 < -0.01:
    → BUY (oversold bounce)
    
IF macd_hist < 0 AND breakout_high = 1:
    → SELL (momentum down + resistance hit)
```

### LSTM (Neural Network)

```
Sees: Last 60 bars of [Open, High, Low, Close, Volume]
Learns: Patterns like "3 higher highs + volume increase"
Outputs: Probability of up/down/sideways
```

Note: LSTM receives raw OHLCV sequences, not the engineered features above. The LSTM learns patterns from raw price action.

### Stacking (Combination)

```
LightGBM says: 70% chance UP
LSTM says: 45% chance UP
Meta-learner decides: 60% chance UP (weighted average)
```

## 📊 Feature Importance (Actual Results)

Based on SHAP analysis from March 2026 run:

Feature importance distributed across many features with no single dominant predictor. This indicates:
- No strong individual signal exists
- Model combines multiple weak signals
- Predictions rely on ensemble effect

Current finding: Feature importance does not show clear predictive patterns, consistent with 50.6% directional accuracy.

## ⚙️ Where to Change Features

Edit config.toml:

```toml
[features]
# Add more EMAs
ema_periods = [10, 20, 50, 100, 200]

# Change RSI period
rsi_period = 21  # Instead of 14

# More lags
lag_periods = [1, 2, 3, 5, 10, 20]

# Disable some features
use_pivots = false    # Skip pivot features
use_session = false   # Skip time features
```

## 🔒 Data Leakage Prevention in Features

All features use proper lagging:

1. Lag features: Uses .shift() - no future info
2. Pivot points: Uses yesterday's data with .shift(1)
3. Moving averages: Calculated from past data only
4. Session features: Derived from timestamp only

Triple-barrier labels use forward-looking bars (naturally), but features do not leak future information.

## 🎓 Tips for Thesis

### What to Write

In Methodology Chapter:
```
Features are categorized into six groups:
1. Technical indicators (EMA, RSI, MACD, ATR, BB)
2. Pivot-based support/resistance
3. Session encoding for time-of-day effects
4. Spread cost information
5. Lagged price and return features
6. Relative distance from moving averages

Total: 45 features per observation.
```

In Results Chapter:
```
SHAP analysis reveals distributed feature importance
with no dominant predictor. This suggests the model
relies on ensemble combination of multiple weak signals
rather than strong individual predictors.
```

---

Next: See Config.md to change feature settings!
