# Understanding Your Results

> A relaxed, beginner-friendly guide to reading the numbers, charts, and reports your pipeline produces. No stress — we will walk through everything.

---

## Where to Find Results

After running the pipeline, everything lives in a session folder:

```
results/latest/              ← Always points to your most recent run
├── reports/
│   ├── thesis_report.md     ← Start here — your full report
│   ├── shap_summary.png     ← Which features matter most
│   ├── confidence_histogram.png  ← How sure the model is
│   └── model_disagreement.png    ← When models disagree
├── backtest/
│   ├── backtest_results.json ← All trading metrics
│   └── trades_detail.csv     ← Every single trade
└── predictions/
    └── *.parquet             ← Raw prediction data
```

---

## 1. Reading the Thesis Report

Open `results/latest/reports/thesis_report.md` in any Markdown viewer or text editor.

The report has these sections:
- **Executive Summary** — The most important numbers, all in one place
- **Data Configuration** — What data was used
- **Model Configuration** — How the models were set up
- **Backtest Results** — How the trading simulation performed
- **Conclusions** — What it all means

---

## 2. The Metrics — What Each One Means

### Predictive Metrics (How well the model predicts)

| Metric | Simple Explanation | Good Range |
|--------|-------------------|------------|
| **Accuracy** | What percentage of predictions were correct | Above 50% is decent (random is 33% for 3 classes) |
| **Macro F1** | Average score across all classes (balances Long/Hold/Short) | Above 0.40 is reasonable |
| **Precision** (per class) | When the model says "Long," how often is it right? | Higher is better |
| **Recall** (per class) | Of all actual "Long" signals, how many did the model find? | Higher is better |

> **Chill note:** In financial prediction, accuracy above 50% for 3 classes is already meaningful. Do not expect 90% — that would be suspicious (probably data leakage).

### Trading Metrics (How much money the strategy makes)

| Metric | Simple Explanation | What to Look For |
|--------|-------------------|-----------------|
| **Total Return** | How much your starting capital grew (or shrank) | Positive is good. Negative means the strategy lost money. |
| **Sharpe Ratio** | Return per unit of risk. Higher = smoother profits | Above 0.5 is decent. Above 1.0 is good. Above 2.0 is excellent. |
| **Sortino Ratio** | Like Sharpe, but only counts downside risk | Should be higher than Sharpe (because upside volatility is OK) |
| **Max Drawdown** | The biggest peak-to-trough loss | Smaller is better. -20% means at one point you were down 20% from your peak. |
| **Calmar Ratio** | Annual return divided by max drawdown | Above 0.5 is reasonable |
| **Win Rate** | What percentage of trades were profitable | Above 50% is good, but not required (you can be profitable with 40% if winners are bigger than losers) |
| **Profit Factor** | Total profits / Total losses | Above 1.0 means profitable. Above 1.5 is good. Above 2.0 is great. |
| **Avg Trade** | Average profit per trade in dollars | Positive = good |
| **Number of Trades** | How many trades the strategy took | Too few (< 30) = unreliable stats. Too many might mean over-trading. |

---

## 3. Understanding the Charts

### SHAP Summary Plot (`shap_summary.png`)

This chart shows **which features the LightGBM model relies on most**.

- **Y-axis:** Feature names, ranked by importance (most important at top)
- **X-axis:** SHAP value — how much each feature pushes the prediction
- **Color:** Red = high feature value, Blue = low feature value
- **Spread:** Wider spread = more important feature

**How to read it:**
1. Features at the top are the most important
2. Red dots on the right mean "high values of this feature push toward one class"
3. A mix of red and blue on both sides means the feature has complex relationships

### Confidence Histogram (`confidence_histogram.png`)

Shows **how confident the model is** in its predictions.

- **X-axis:** Confidence level (0.0 to 1.0)
- **Y-axis:** Number of predictions at each confidence level

**What you want to see:**
- A peak on the **right side** (high confidence) = the model is sure about many predictions
- A long tail on the **left** (low confidence) = some predictions are uncertain (these become "Hold" signals with the 0.6 threshold)

### Model Disagreement (`model_disagreement.png`)

Shows **when LightGBM and LSTM disagree** with each other.

- Some disagreement is **normal and healthy** — that is why we use both models
- If they always agree, stacking would not add value
- The meta-learner learns which model to trust in different situations

---

## 4. Reading the Backtest Results

### The JSON File

Open `backtest_results.json` to see all metrics in detail:

```json
{
  "total_return": 0.15,        // 15% profit
  "sharpe_ratio": 1.2,         // Good risk-adjusted return
  "max_drawdown": -0.12,       // Worst peak-to-trough loss was 12%
  "win_rate": 0.55,            // 55% of trades were profitable
  "profit_factor": 1.8,        // Total wins / Total losses
  "total_trades": 150,         // Number of trades taken
  "avg_trade_pnl": 100.0       // Average $100 profit per trade
}
```

### The Trade Log

Open `trades_detail.csv` in a spreadsheet or text editor. Each row is one trade:

| Column | Meaning |
|--------|---------|
| `entry_time` | When the trade was opened |
| `exit_time` | When the trade was closed |
| `direction` | Long (buy) or Short (sell) |
| `entry_price` | Price when the trade started |
| `exit_price` | Price when the trade ended |
| `position_size` | How many lots were traded |
| `pnl` | Profit or loss for this trade |
| `exit_reason` | Why the trade closed (TP hit, SL hit, timeout, etc.) |

---

## 5. What Makes a "Good" Result?

Here is a simple checklist:

| Check | What to Look For |
|-------|-----------------|
| Positive total return | The strategy makes money overall |
| Sharpe > 0.5 | The returns are not just from luck |
| Max drawdown < 30% | The worst loss period is survivable |
| Profit factor > 1.2 | Winners are bigger than losers |
| More than 30 trades | The sample size is meaningful |
| Reasonable win rate | 40-60% is typical for trend-following |
| Model accuracy > 45% | Better than random for 3 classes |

---

## 6. Common Scenarios

### "My Sharpe ratio is negative"

This means the strategy loses money more often than it gains. Possible causes:
- The model might be overfitting to the training data
- Try adjusting the confidence threshold (increase to 0.7 to be more selective)
- Check if the data range is appropriate

### "Win rate is 35% but total return is positive"

This is actually fine! It means your winning trades are larger than your losing trades. This is a common pattern in trend-following strategies.

### "Too many Hold signals"

The confidence threshold is filtering out uncertain predictions. This is by design — it means the model is being careful. You can lower the threshold to get more trades, but they might be lower quality.

### "Very few trades"

Possible reasons:
- The confidence threshold is too high (try lowering from 0.6 to 0.5)
- The model is being very selective
- The test period might be in a choppy market

### "Model accuracy is 50% — is that bad?"

For 3 classes (Long/Hold/Short), random guessing gives 33% accuracy. So 50% is **significantly better than random**. In financial markets, even small edges are valuable when applied consistently.

---

## 7. Comparing Runs

Each run creates a separate session folder, so you can compare different configurations:

```bash
# List all sessions
ls results/

# Compare two reports
diff results/XAUUSD_1H_20260406_102913/reports/thesis_report.md \
     results/XAUUSD_1H_20260407_143022/reports/thesis_report.md
```

**Things to compare across runs:**
- Sharpe ratio (higher = better risk-adjusted return)
- Max drawdown (smaller = less risk)
- Total return (higher = more profit)
- Number of trades (similar = consistent behavior)
- Win rate (similar = stable strategy)

---

## 8. Quick Sanity Checks

Before trusting any results, check these:

1. **No data leakage:** The test set should be chronologically after training. Check the dates in the report.
2. **Reasonable metrics:** If Sharpe > 5.0 or accuracy > 80%, something is probably wrong (likely data leakage or overfitting).
3. **Enough trades:** At least 30 trades in the backtest for statistical significance.
4. **Trading costs included:** The backtest already includes spread (2 pips) and slippage (1 pip), so the results are realistic.
5. **Hold signals exist:** If every prediction is Long or Short, the model might not have a proper Hold mechanism.

> **Remember:** Past performance does not guarantee future results. The backtest simulates historical trading — real trading has additional factors like execution delays and market impact.
