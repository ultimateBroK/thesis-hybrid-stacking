# Evaluation: How to Read Your Results

## 🎯 What You'll Get

After running, check results/thesis_report.md and results/backtest_results.json.

## 📊 Current Results (After Data Leakage Fix)

Fixed LSTM data leakage on March 28, 2026. Results now show:

```
Total Trades:       791
Win Rate:           68.2%
Total Return:       1446%
Sharpe Ratio:       3.86
Max Drawdown:       15.9%
Directional Accuracy: 50.6%
```

IMPORTANT: These returns are NOT from prediction skill. They result from volatility harvesting during gold's massive 2024-2026 bull run.

## 📊 Key Metrics Explained

### 1. Total Return
What: How much money you made

Example:
- Start with: 100,000 USD
- End with: 1,546,000 USD
- Total Return: 1446%

Why it's unrealistic:
- Gold price increased 117% from 2024-2026 (2065 to 4494 USD)
- Model predicts Short 85% of the time but captures volatile pullbacks
- Always-in-market reversing strategy with max 2.0 lots position size
- 791 trades averaging 991 pips each over 24.7 hours

### 2. Win Rate
What: Percent of winning trades

Example:
- 791 total trades
- 539 wins
- Win Rate: 68.2%

Why it's high:
- Volatile market with large price swings
- Strategy captures intraday volatility
- Not predictive skill, just market conditions

### 3. Profit Factor
What: Gross profit / Gross loss

Example:
- Total wins: 15,000 USD
- Total losses: 10,000 USD
- Profit Factor: 1.5

Good if: > 1.2
Great if: > 1.5
Bad if: < 1.0 (losing money!)

### 4. Sharpe Ratio
What: Return vs risk. Higher = better risk-adjusted return

Example:
- Sharpe = 3.86
- Means: Exceptional returns for risk taken

Good if: > 1.0
Great if: > 2.0
Excellent: > 3.0
Note: Sharpe > 3.0 is suspicious for retail trading strategies

### 5. Max Drawdown
What: Biggest drop from peak to trough

Example:
- Peak: 1,100,000 USD
- Lowest: 926,000 USD
- Max DD: 15.9%

Good if: < 15%
Acceptable: < 20%
Dangerous: > 30%

### 6. Directional Accuracy
What: Percent of correct direction predictions (Long vs Short)

Current: 50.6%

This is the critical metric:
- 50.6% = Barely better than coin flip
- Profits come from volatility, not prediction skill
- Model profits from being in market during large moves

## ⚠️ Why These Returns Are Unrealistic

### Market Context
Gold price action 2024-2026:
- Jan 2024: 2065 USD/oz
- Mar 2026: 4494 USD/oz
- Total increase: +117%

### Strategy Behavior
- Predicts Short: 85% of the time
- Predicts Long: 15% of the time
- Always in market (no flat positions)
- Reverses position on every signal change

### How It Makes Money
The strategy profits from volatility harvesting:
1. Gold makes large intraday swings (avg 991 pips per trade)
2. Strategy captures pullbacks during uptrend
3. Position sizing (2.0 lots max) amplifies returns
4. 791 trades compound over 26 months

### Data Leakage Investigation
We fixed one source of leakage (LSTM normalization) but returns remain high.

Before fix: 1622% return
After fix: 1446% return (-10.8% reduction)

This confirms leakage was present but not the main cause.

### Root Cause
The returns are mathematically possible because:
- Gold's extreme volatility (2024-2026 bull run)
- High leverage (100:1) amplifies gains
- Always-in-market strategy compounds returns
- Large average trade size (991 pips)

BUT this is not replicable:
- Requires identical market conditions
- 50.6% directional accuracy means luck-dependent
- Will lose money in ranging markets
- Cannot predict next regime

## 📈 Reading the Numbers

### Example: Current Strategy (Fixed Leakage)

```
Total Return:     1446%  ✗ Unrealistic (market-dependent)
Win Rate:         68.2%  ✓ Good but volatility-driven
Profit Factor:    2.24   ✓ Good
Sharpe Ratio:     3.86   ✗ Suspicious (>3.0)
Max Drawdown:     15.9%  ✓ Good control
Directional Acc:  50.6%  ✗ No prediction skill
```

Verdict: Strategy exploits volatility, not predictive ability.

## 🎯 What to Look For in Your Thesis

### Minimum Acceptable
```
Total Return  > 0% (positive)
Max Drawdown  < 20%
Directional Accuracy > 52% (better than random)
```

### Good Results (Realistic)
```
Total Return  > 15%
Max Drawdown  < 15%
Directional Accuracy > 55%
Profit Factor > 1.3
Sharpe Ratio  > 1.0 (and < 2.0)
```

### Red Flags (Concerning)
```
Total Return > 100%        ← Market regime exploitation
Max Drawdown > 50%         ← Will blow up account
Win Rate > 65%             ← Suspicious in volatile market
Sharpe Ratio > 3.0         ← Unrealistic for retail
Directional Accuracy < 52% ← No predictive edge
```

If you see these: Strategy relies on market conditions, not skill.

## 📊 Understanding Label Distribution

In the logs, you'll see:

```
Label distribution:
  -1 (Short):  18,500 (35.6%)
   0 (Hold):    8,200 (15.8%)
   1 (Long):   25,300 (48.6%)
```

Current prediction distribution:
```
  -1 (Short):  85%
   0 (Hold):   0%
   1 (Long):  15%
```

Imbalance shows model learned: always bet against trend (and let volatility work).

## 📈 Backtest vs Reality

### What's Included
Your backtest includes:
- Spread costs (2 pips)
- Slippage (1 pip)
- Leverage (100:1)
- Realistic position sizing

### What's NOT Included
- Latency (execution speed)
- Broker rejection
- Psychological factors
- Changing market regimes

Rule: Real trading = Backtest × 0.3 to 0.5 for high-return strategies

Example:
- Backtest: 1446% return
- Reality: ~430-720% return (if conditions repeat)
- More likely: Strategy fails when regime changes

## 🔍 SHAP Analysis (Feature Importance)

Look at results/shap_summary.png:

Current findings:
- Feature importance distributed across many features
- No single dominant predictor
- Confirms: No strong predictive signal found

## ✅ Checklist for Good Results

After running, verify:
- [ ] Total Return is positive
- [ ] Max Drawdown is < 20%
- [ ] Directional Accuracy > 52%
- [ ] Sharpe Ratio between 1.0-2.0
- [ ] Feature importance shows predictive signals
- [ ] Test on different market regimes

## 🎓 For Your Thesis Defense

### What Professors Ask

Q: "Why is return so high?"
A: "1446% return reflects gold's 117% price increase from 2024-2026 combined with volatility harvesting. Directional accuracy is only 50.6%, indicating profits come from market volatility, not prediction skill."

Q: "Is this realistic?"
A: "The strategy exploits a specific market regime (gold bull run with high volatility). Returns are mathematically possible but not replicable. The 50.6% directional accuracy shows no genuine predictive edge."

Q: "What does 50.6% accuracy mean?"
A: "It means the model barely beats random guessing (50%). Profits come from being always in market during volatile conditions, not from correctly predicting direction."

Q: "Did you check for data leakage?"
A: "Yes. Found and fixed LSTM normalization leakage. Returns dropped from 1622% to 1446% (-10.8%), confirming leakage was present but not the main cause of high returns."

## 📊 Comparing to Benchmarks

### Buy-and-Hold Gold (2024-2026)
```
Gold price:  2065 → 4494 USD
Return:      117%
Max DD:      18% (pullbacks during uptrend)
Sharpe:      ~1.2
```

### Your Strategy
```
Return:      1446%
With:        Same drawdown periods
But:         50.6% directional accuracy
```

Interpretation: Strategy levered up gold's volatility, not gold's trend.

### Realistic Expectation
If market regime changes to ranging:
- Return: -20% to +10%
- Win rate: 45-55%
- Max DD: >30%

## ⚠️ Key Finding for Thesis

The 1446% return is not evidence of a working strategy. It is evidence that:
1. Gold had an exceptional bull run (2024-2026)
2. Always-in-market strategies profit from volatility
3. High leverage amplifies returns (and would amplify losses)
4. 50.6% directional accuracy shows no predictive skill

Recommendation: Present this as "volatility harvesting during regime change" not as "predictive trading strategy."

---

Next: See Config.md to understand parameter effects.
