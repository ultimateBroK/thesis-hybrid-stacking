# Configuration Tuning Guide

> Practical guidance for achieving best results from the hybrid GRU + LightGBM model.

---

## Overview

This guide helps you adjust `config.toml` for different testing scenarios, timeframes, and goals. The pipeline has many parameters, but only a few have the biggest impact on results. This guide tells you **which parameters to tune first** and **what ranges to try**.

---

## 1. High-Impact Parameters

These parameters have the most effect on your backtest results. Adjust these before touching anything else.

### `horizon_bars` (labels)

| What it does | Maximum time window for a trade, in bars |
|--------------|----------------------------------------|
| **Default** | 24 |
| **Range** | 12–48 |
| **Effect** | Shorter = faster trades, more signals, higher costs. Longer = swing trading, fewer signals, larger moves. |

- **12–16 bars**: Scalping style. More trades, more costs, catches small moves.
- **24 bars** (default for 1H): Covers ~1–2 trading days. Balanced.
- **36–48 bars**: Swing style. Fewer trades, needs bigger moves to hit TP/SL.

> For XAU/USD on 1H, `horizon_bars = 24` is a good starting point. Increase to 36–48 for a more conservative, swing-trading approach.

---

### `confidence_threshold` (backtest)

| What it does | Minimum predicted probability before taking a trade |
|--------------|---------------------------------------------------|
| **Default** | 0.50 |
| **Range** | 0.0–0.9 |
| **Effect** | Higher = fewer but more confident trades. Lower = more signals, more noise. |

- **0.0**: Trade on ALL signals (ignore model confidence).
- **0.40–0.50**: Moderate confidence required (current default).
- **0.50** (default): Only trade when model is at least 50% confident.
- **0.65–0.75**: Conservative — only high-confidence trades. Good for live trading.

> If your backtest has **too few trades** (< 100), try lowering to 0.40. If you see **too many small losses**, try raising to 0.60–0.65.
>
> **Lot scaling:** Position size scales from `min_lots` (0.01) at threshold confidence up to `max_lots` (0.5) at 1.0 confidence. This means high-conviction trades get 50x the capital of low-conviction trades.

---

### `atr_stop_multiplier` (backtest)

| What it does | Stop-loss distance as a multiple of ATR |
|--------------|---------------------------------------|
| **Default** | 2.0 (must match `[labels] atr_sl_multiplier`) |
| **Range** | 0.5–3.0 |
| **Effect** | Lower = tighter stops, more stopped out, smaller losses per trade. Higher = wider stops, room for volatility, larger losses. |

- **0.5–1.0**: Very tight. Good for calm markets, but gets stopped out often in volatile periods.
- **1.5–2.0**: Wider. Gives price room during volatile periods. XAUUSD typical range.
- **2.0** (default): Balanced for gold — matches the 2xATR barrier used in labeling.
- **2.5–3.0**: Very wide. Fewer stop-outs but larger losses per stop.

> **Barrier alignment constraint:** `atr_stop_multiplier` must equal `[labels] atr_sl_multiplier` (both 2.0 by default). Changing one without the other creates a mismatch between training targets and execution risk, degrading OOS performance.

> XAU/USD is volatile. During gold's bull run (2024–2026), ATR multipliers of 2.0 work well — wider stops are needed to survive gold's intraday swings.

---

### `atr_tp_multiplier` (backtest)

| What it does | Take-profit distance as a multiple of ATR (0 = disabled) |
|--------------|-------------------------------------------------------|
| **Default** | 2.0 |
| **Range** | 0–4.0 |
| **Effect** | Controls profit target. Lower = quicker exits, higher = let winners run. |

- **0**: No take-profit. Trades exit only on stop-loss or end of `horizon_bars`.
- **1.5**: Quick exits. Good for scalping, lower risk-reward.
- **2.0** (default): Balanced — TP is 2× the stop distance.
- **2.5–3.0**: Let winners run. Higher risk-reward, fewer trades hit TP.

> A common pattern is symmetric barriers: `atr_tp_multiplier = atr_stop_multiplier = 2.0`. With TP and SL both at 2xATR, the model learns from balanced labels where wins and losses are equally sized. The 2:1 reward-to-risk pattern is not applicable with current barrier-aligned design.

---

### `sequence_length` (gru)

| What it does | How many past bars the GRU reads before predicting |
|--------------|---------------------------------------------------|
| **Default** | 48 |
| **Range** | 24–96 |
| **Effect** | Shorter = faster training, less context. Longer = more temporal context, slower training, risk of overfitting. |

- **24–32 bars**: Fast training, good for initial experiments.
- **48 bars** (default): ~2 trading days of context. Good balance.
- **64–96 bars**: Longer memory. Risk of overfitting on small datasets.

> For 1H timeframe, `sequence_length = 48` (2 days) is standard. For Daily, try 20–30. For 30min, try 96+.

---

### `hidden_size` (gru)

| What it does | Size of the GRU's internal memory (hidden states) |
|--------------|--------------------------------------------------|
| **Default** | 64 |
| **Range** | 32–256 |
| **Effect** | Larger = more capacity to learn complex patterns, but slower and more risk of overfitting. |

- **32–64**: Small datasets (< 2 years) or fast experiments.
- **64** (default): Stable capacity for 5+ years of data without over-amplifying noisy regimes.
- **192–256**: Very large capacity. Needs strong regularization (dropout 0.3+) and early stopping.

> With 5+ years of XAU/USD data and 20 input features, `hidden_size = 64` is the stable default. The latest 128-hidden regression run degraded OOS class signal, so larger sizes should be treated as experiments.
>
> **Objective default:** GRU trains with multiclass focal loss. Regression is available for experiments only.

---

## 2. Timeframe-Specific Guidance

Different timeframes require different parameter values.

### For 30min Timeframe

| Parameter | Value | Why |
|-----------|-------|-----|
| `horizon_bars` | 48 | ~2 trading days of context |
| `sequence_length` | 96 | Covers weekend gaps better |
| `atr_stop_multiplier` | 1.5–2.0 | More noise at 30min, wider stops |
| `atr_tp_multiplier` | 2.5 | Need larger targets to cover costs |
| `confidence_threshold` | 0.65 | Filter false signals |

### For 1H Timeframe (default)

| Parameter | Value | Why |
|-----------|-------|-----|
| `horizon_bars` | 24 | ~1–2 trading days |
| `sequence_length` | 48 | ~2 trading days of context |
| `atr_stop_multiplier` | 2.0 | Must match label barriers for gold volatility |
| `atr_tp_multiplier` | 2.0 | Symmetric TP/SL (2:1 not applicable with symmetric barriers) |
| `confidence_threshold` | 0.50 | Default trade filter; sizing remains fixed |

### For 4H Timeframe

| Parameter | Value | Why |
|-----------|-------|-----|
| `horizon_bars` | 12–18 | ~2–3 trading days |
| `sequence_length` | 24–36 | ~4–6 trading days |
| `atr_stop_multiplier` | 1.0–1.5 | Less noise at 4H |
| `atr_tp_multiplier` | 2.0–2.5 | Let trades develop |
| `confidence_threshold` | 0.55 | Still need enough trades |

### For Daily Timeframe

| Parameter | Value | Why |
|-----------|-------|-----|
| `horizon_bars` | 5–10 | ~1–2 weeks |
| `sequence_length` | 20–30 | ~1 month |
| `atr_stop_multiplier` | 2.0–3.0 | Let trades develop |
| `atr_tp_multiplier` | 3.0–4.0 | Larger moves expected |
| `confidence_threshold` | 0.50 | Fewer bars, still need signals |

---

## 3. Walk-Forward Validation (`[validation]`)

Walk-forward validation is critical for time-series models. These parameters control how the pipeline splits data into train/test windows. Window sizes are **bar-based** row counts, not guaranteed calendar durations.

| Parameter | Default | What it does |
|-----------|---------|-------------|
| `method` | `"sliding"` | `"sliding"` walk-forward or `"static"` fixed split |
| `train_window_bars` | 17520 (~2yr) | Training data per window |
| `test_window_bars` | 4380 (~6mo) | Out-of-sample period per window |
| `step_bars` | 4380 | Step between consecutive windows |
| `purge_bars` | 48 | Bars removed at train/test boundary |
| `embargo_bars` | 50 | Additional gap after purge |
| `min_train_bars` | 10000 | Minimum training bars to produce a window |
| `oof_ensemble` | `true` | Aggregate OOF predictions across windows |

### How to adjust

- **`train_window_bars`**: Default is 17520 (~2 years). Increase for more stable long-regime training; decrease only if you lack data. Ensure it stays at least `min_train_bars` (10000).
- **`test_window_bars`**: Shorter (2190 ≈ 3mo) = more windows, more robust evaluation. Longer (4380 ≈ 6mo) = more stable per-window estimates.
- **`step_bars`**: Set equal to `test_window_bars` for non-overlapping windows. Set lower (e.g., 2190) for overlapping windows — gives more test periods but slower.
- **`purge_bars` / `embargo_bars`**: Anti-leakage gaps. Keep `purge_bars ≥ 48` and `embargo_bars ≥ 50` for 1H data. Increase purge if average trade holding time exceeds 48 bars.

### Common patterns

```toml
# Fast iteration — single static split
[validation]
method = "static"

# Thorough walk-forward validation
[validation]
method = "sliding"
train_window_bars = 17520
test_window_bars = 4380
step_bars = 4380
purge_bars = 48
embargo_bars = 50
```

---

## 4. Backtest Parameters (`[backtest]`)

All backtest parameters with their defaults and guidance:

| Parameter | Default | What it does |
|-----------|---------|-------------|
| `initial_capital` | 10000.0 | Starting account balance |
| `leverage` | 10 | Margin = 1/leverage |
| `spread_ticks` | 35 | Bid-ask spread in ticks |
| `slippage_ticks` | 5 | Execution slippage in ticks |
| `commission_per_lot` | 10.0 | Commission per standard lot |
| `atr_stop_multiplier` | 2.0 | Stop-loss ATR multiplier (must match `[labels] atr_sl_multiplier`) |
| `atr_tp_multiplier` | 2.0 | Take-profit ATR multiplier (must match `[labels] atr_tp_multiplier`; 0 = disabled) |
| `lots_per_trade` | 0.01 | Fixed conservative lot size after confidence filtering |
| `min_lots` | 0.01 | Minimum lot safety bound |
| `max_lots` | 0.5 | Maximum lot safety cap |
| `confidence_threshold` | 0.50 | Minimum probability to trade (0 = disabled) |
| `min_bars_between_trades` | 6 | **Trade cooldown**: minimum bars between position exit and next entry |
| `max_drawdown_cutoff` | 0.30 | Circuit breaker: stop if equity drops 30% from peak |
| `dd_cooldown_bars` | 12 | Bars to pause after drawdown cutoff breach |
| `max_open_positions` | 1 | Maximum simultaneous open positions |
| `daily_loss_limit` | 0.03 | Stop trading for the day after -3% equity drawdown |

### Risk management parameters

- **`min_bars_between_trades`**: Trade cooldown reduces overtrading and correlation between consecutive trades. Increase (12–24) for swing trading. Decrease (2–3) for higher frequency. At 6, a minimum 6-hour gap between trades on 1H data.
- **`dd_cooldown_bars`**: After the drawdown circuit breaker triggers, the system pauses for this many bars. Increase (24–48) for more cautious recovery. Decrease (6) to resume faster.
- **`max_open_positions`**: Keep at 1 for single-instrument backtesting. The pipeline supports higher values for multi-instrument setups.
- **`daily_loss_limit`**: Set to 0.02 for conservative daily risk caps. Set to 0.05 for more tolerance. Set to 0.0 to disable.

### Position Sizing

Confidence filters entries only. Position size remains fixed at `lots_per_trade`
and is clamped to `[min_lots, max_lots]`.

Why: confidence-scaled lots amplified wrong high-confidence predictions in the latest
OOS run. Do not re-enable lot amplification until fixed 0.01-lot backtests are
profitable and statistically stable.

---

## 5. Dataset-Size Guidance

The amount of training data affects which parameters are safe to use.

### Small Dataset (< 2 years)

| Parameter | Adjustment | Why |
|-----------|------------|-----|
| `sequence_length` | Reduce to 24–32 | Less data to fill long sequences |
| `hidden_size` | Reduce to 32–64 | Fewer parameters to prevent overfitting |
| `correlation_threshold` | Increase to 0.85 | Keep more features |
| `min_child_samples` | Increase to 300+ | More conservative splits |
| `batch_size` | Reduce to 128 | Smaller batches for small datasets |
| `train_window_bars` | Reduce to 17520 (~2yr) | Match available data |

### Medium Dataset (2–5 years)

| Parameter | Adjustment | Why |
|-----------|------------|-----|
| `sequence_length` | 48 (default) | Good balance |
| `hidden_size` | 64–128 (default) | Good balance |
| `correlation_threshold` | 0.75 (default) | Default filtering |
| `batch_size` | 256 (default) | Standard |

### Large Dataset (> 5 years)

| Parameter | Adjustment | Why |
|-----------|------------|-----|
| `sequence_length` | 48–64 | Can afford longer context |
| `hidden_size` | 128–192 | More capacity for rich data |
| `correlation_threshold` | 0.70–0.75 | Can afford aggressive filtering |
| `batch_size` | 256–512 | Larger batches for stable gradients |

---

## 6. Profiles for Different Goals

Choose a profile based on your goal and risk tolerance.

### Conservative Profile

> Fewer trades, lower drawdown, suitable for **live trading** or **small accounts**.

```toml
[backtest]
confidence_threshold = 0.65
atr_stop_multiplier = 2.0
atr_tp_multiplier = 2.0
min_bars_between_trades = 12
max_drawdown_cutoff = 0.20
dd_cooldown_bars = 24
daily_loss_limit = 0.02

[labels]
horizon_bars = 36
```

**Expected behavior**: 30–50% fewer trades, lower max drawdown, smoother equity curve. Higher cooldown (12 bars) prevents overtrading.

---

### Balanced Profile (Default)

> Default settings that work well for XAU/USD 1H with 5+ years of data.

```toml
[backtest]
confidence_threshold = 0.50        # Only trade with >50% conviction
atr_stop_multiplier = 2.0          # Must match [labels] atr_sl_multiplier
atr_tp_multiplier = 2.0            # Must match [labels] atr_tp_multiplier
lots_per_trade = 0.01              # Fixed conservative size
min_lots = 0.01                    # Safety floor
max_lots = 0.5                     # Safety cap
min_bars_between_trades = 6        # Trade cooldown
max_drawdown_cutoff = 0.30
dd_cooldown_bars = 12
daily_loss_limit = 0.03

[labels]
horizon_bars = 24
```

**Expected behavior**: Conservative risk while measuring whether the model has real edge. Default recommendation.

---

### Aggressive Profile

> More trades, higher potential return, higher drawdown risk. For **larger accounts** with **higher risk tolerance**.

```toml
[backtest]
confidence_threshold = 0.40
atr_stop_multiplier = 2.0
atr_tp_multiplier = 2.0
min_bars_between_trades = 3
max_drawdown_cutoff = 0.40
dd_cooldown_bars = 6
daily_loss_limit = 0.05

[labels]
horizon_bars = 16
```

**Expected behavior**: 50–80% more trades, higher return potential but also higher drawdown. Lower cooldown (3 bars) allows faster re-entry. Monitor closely.

---

## 7. Tuning Priority Order

When optimizing, adjust parameters in this order. Each step builds on the previous.

1. **`horizon_bars`** — Most impact on label quality. Start here.
2. **`confidence_threshold`** — Controls trade frequency. Adjust for your account size.
3. **`atr_stop_multiplier` / `atr_tp_multiplier`** — Direct impact on risk/reward ratio.
4. **`sequence_length`** — GRU context. Adjust for your timeframe.
5. **`hidden_size`** — GRU capacity. Adjust for your dataset size.
6. **`correlation_threshold`** — Feature set. Adjust if feature count is too low/high.
7. **`train_window_bars` / `test_window_bars`** — Walk-forward window sizing.
8. **LightGBM params** — Fine-tune after above are stable (`max_depth`, `min_child_samples`, `learning_rate`).

---

## 8. Common Pitfalls

Avoid these mistakes:

| Mistake | Why It's Bad | Fix |
|---------|--------------|-----|
| `horizon_bars` too short (5–10) | Labels become noise — price has no time to reach TP/SL | Minimum 12 for 1H, 16+ for 30min |
| `confidence_threshold` too low (0.3) | Too many false signals, poor win rate | Minimum 0.5, prefer 0.55+ |
| `atr_stop_multiplier` too tight (0.5) | Constantly stopped out by normal volatility | Minimum 0.8, prefer 1.0+ |
| `sequence_length` too long (100+) | GRU overfits to training data | Maximum 64 for small data, 96 for large |
| `correlation_threshold` too high (0.95) | Redundant features add noise, slow training | Keep at 0.75 or lower |
| `hidden_size` too large | GRU memorizes or loses OOS class signal | Keep 64 default; test 128+ only with OOS validation |
| `purge_bars` / `embargo_bars` too low | Label leakage from train into test | Keep purge ≥ 48, embargo ≥ 50 for 1H; increase purge if trades hold longer |
| `max_drawdown_cutoff` too low (0.10) | Backtest stops too early, not enough trades | Minimum 0.20, prefer 0.30 |

---

## 9. Quick Reference Tables

### ATR Multiplier by Market Volatility

| Market Condition | `atr_stop_multiplier` | `atr_tp_multiplier` |
|------------------|----------------------|---------------------|
| Calm (low ATR) | 0.8–1.0 | 1.5–2.0 |
| Normal | 1.0 (default) | 2.0 (default) |
| Volatile (high ATR) | 1.5–2.5 | 2.5–3.5 |

### Confidence Threshold by Account Size

| Account Size | `confidence_threshold` | `daily_loss_limit` |
|--------------|----------------------|-------------------|
| < $5,000 | 0.65–0.70 | 0.02 |
| $5,000–$20,000 | 0.55–0.65 | 0.03 |
| > $20,000 | 0.50–0.55 | 0.03–0.05 |

### GRU Parameters by Timeframe

| Timeframe | `sequence_length` | `batch_size` | `hidden_size` |
|-----------|-------------------|-------------|--------------|
| 30min | 96 | 256 | 32 |
| 1H | 48 | 256 | 32 |
| 4H | 24–36 | 256 | 32 |
| Daily | 20–30 | 256 | 16–24 |

---

## 10. Monitoring and Iteration

After running a backtest:

1. **Check trade count**: < 100 trades = not statistically meaningful. Adjust `confidence_threshold` or `horizon_bars`.
2. **Check max drawdown**: > 25% is risky for most traders. Increase `confidence_threshold` or widen `atr_stop_multiplier`.
3. **Check win rate**: Should be above 40% with good risk-reward. Profit factor matters more.
4. **Check profit factor**: Above 1.5 is good. Above 2.0 is excellent but be suspicious of > 3.0 (possible overfitting).
5. **Compare to buy & hold**: Your alpha (excess return over buy & hold) should be positive.

---

## 11. Example Tuning Workflow

```mermaid
flowchart TD
    A["Run with defaults"] --> B["Check trade count"]
    B --> |"Too few"| C["Lower confidence_threshold to 0.50"]
    B --> |"Too many"| D["Raise confidence_threshold to 0.65"]
    C --> E["Run backtest"]
    D --> E
    E --> F["Check max drawdown"]
    F --> |"Too high"| G["Raise atr_stop_multiplier to 1.5"]
    F --> |"OK"| H["Check profit factor"]
    H --> |"Below 1.5"| I["Try horizon_bars = 36"]
    H --> |"Above 1.5"| J["Tuning complete<br/>Log results"]
    I --> K["Run backtest"]
    K --> J
    G --> L["Run backtest"]
    L --> H
```

---

## Summary

| Parameter | Default | Range | Key Use |
|-----------|---------|-------|---------|
| `horizon_bars` | 24 | 12–48 | Trade duration |
| `confidence_threshold` | 0.55 | 0.0–0.9 | Trade frequency |
| `atr_stop_multiplier` | 1.0 | 0.5–3.0 | Stop-loss distance |
| `atr_tp_multiplier` | 2.0 | 0–4.0 | Take-profit distance |
| `sequence_length` | 48 | 24–96 | GRU context |
| `hidden_size` | 32 | 16–64 | GRU capacity |
| `batch_size` | 256 | 64–512 | GRU training |
| `correlation_threshold` | 0.75 | 0.5–0.95 | Feature selection |
| `train_window_bars` | 17520 | 10000–35040 | Walk-forward train size |
| `test_window_bars` | 4380 | 2190–8760 | Walk-forward test size |
| `max_drawdown_cutoff` | 0.30 | 0.10–0.50 | Circuit breaker |
| `dd_cooldown_bars` | 12 | 6–48 | Post-drawdown pause |
| `daily_loss_limit` | 0.03 | 0.0–0.05 | Daily risk cap |
