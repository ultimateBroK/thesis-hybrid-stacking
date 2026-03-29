# Quickstart: Run the Project in 5 Minutes

## 🎯 Goal

Run the full pipeline: ticks → features → models → backtest → report

## ⚡ Quick Commands

```bash
# 1. Go to project folder
cd /home/ultimatebrok/Downloads/thesis

# 2. Install dependencies (first time only)
pixi install

# 3. Run everything
python main.py

# Done! Check results/ folder
```

## 📋 Step-by-Step (Detailed)

### Step 1: Check Your Data

Make sure tick data exists:
```bash
ls data/raw/XAUUSD/
# Should show: 2018-01.parquet, 2018-02.parquet, ..., 2026-03.parquet
```

### Step 2: Run Full Pipeline

```bash
python main.py
```

You will see:
```
======================================================================
Hybrid Stacking (LSTM + LightGBM) - XAU/USD H1 Trading Signals
Bachelor's Thesis - Thuy Loi University
Student: Nguyen Duc Hieu | Advisor: Hoang Quoc Dung
======================================================================
Loading configuration from: config.toml
Data range: 2018-01-01 to 2026-03-31
Train: 2018-01-01 → 2022-12-31
Val: 2023-01-01 → 2023-12-31
Test: 2024-01-01 → 2026-03-31
======================================================================
STAGE DATA
======================================================================
Found 99 tick files to process
Processing tick files: 100%|████████| 99/99
Saved OHLCV data: 52,000 candles
...
Pipeline completed successfully!
Results: results/thesis_report.md
```

### Step 3: Check Results

```bash
# View report
cat results/thesis_report.md

# View JSON metrics
cat results/backtest_results.json

# View plots
ls results/
# shap_summary.png, feature_importance.png, equity_curve.png
```

## 🎮 Common Commands

### Run Only One Stage

```bash
# Only process data (ticks → OHLCV)
python main.py --stage data

# Only train LightGBM
python main.py --stage lightgbm

# Only run backtest
python main.py --stage backtest
```

### Force Re-run (Ignore Cache)

```bash
# Re-run everything (ignore saved files)
python main.py --force

# Re-run only features stage
python main.py --stage features --force
```

### Change Settings

```bash
# Use 8 CPU cores instead of all
python main.py --jobs 8

# Different random seed
python main.py --seed 123

# Custom config file
python main.py --config my_config.toml
```

## 🔧 Environment Variables

Change settings without editing files:

```bash
# Change timeframe to 30 minutes
export THESIS_DATA__TIMEFRAME=30m
python main.py

# Use fewer Optuna trials (faster)
export THESIS_MODELS__TREE__OPTUNA_TRIALS=50
python main.py

# Reduce LSTM epochs
export THESIS_MODELS__LSTM__EPOCHS=50
python main.py
```

## ⏱️ Expected Runtime

| Stage | Time | Why |
|-------|------|-----|
| Data (ticks→OHLCV) | 5-10 min | 300M ticks to process |
| Features | 2-3 min | Technical indicators |
| Labels | 1-2 min | Triple-barrier loop |
| Split | 30 sec | Simple filtering |
| LightGBM | 15-30 min | Optuna tuning (100 trials) |
| LSTM | 20-40 min | PyTorch training (100 epochs) |
| Stacking | 2-3 min | Meta-learner |
| Backtest | 1 min | Simulation |
| Report | 2-3 min | SHAP analysis |
| TOTAL | ~1-2 hours | First run |

After first run: 2-5 minutes (cached)

## 🐛 Common Issues

### Issue: "No module named 'thesis'"

Fix: Make sure you're in the project root
```bash
cd /home/ultimatebrok/Downloads/thesis
python main.py
```

### Issue: "LightGBM not installed"

Fix: Install with pixi
```bash
pixi add lightgbm
```

### Issue: "Out of memory"

Fix: Reduce parallel jobs
```bash
python main.py --jobs 4
```

Or edit config.toml:
```toml
[workflow]
n_jobs = 4  # Instead of -1 (all cores)
```

### Issue: "No tick files found"

Fix: Check data location
```bash
ls data/raw/XAUUSD/
# If empty, download Dukascopy data first
```

## 📁 Output Files

After running, you'll have:

```
data/processed/
├── ohlcv.parquet          # H1 candles
├── features.parquet       # Technical indicators
├── labels.parquet         # Triple-barrier labels
├── train.parquet          # Training data (2018-2022)
├── val.parquet            # Validation data (2023)
└── test.parquet           # Test data (2024-2026)

data/predictions/
├── lightgbm_oof.parquet   # LightGBM predictions
├── lstm_oof.parquet       # LSTM predictions
└── final_predictions.parquet  # Stacking output

models/
├── lightgbm_model.pkl     # Trained LightGBM
├── lstm_model.pt          # Trained LSTM
├── lstm_norm_stats.npz    # LSTM normalization stats (NEW)
└── stacking_meta_learner.pkl  # Meta-learner

results/
├── thesis_report.md       # Main report (READ THIS!)
├── backtest_results.json  # Raw metrics
├── trades_detail.csv      # Individual trade list
├── shap_summary.png       # Feature importance
└── feature_importance.png # LGBM importance
```

## 📊 Data Leakage Fix (March 2026)

LSTM now saves normalization statistics from training:

```bash
# Check normalization stats exist
ls models/lstm_norm_stats.npz

# If missing, re-run LSTM stage
python main.py --stage lstm --force
```

This prevents test set statistics from leaking into predictions.

## 🎓 Tips for Understanding Results

### Data Split
- Train: 2018-2022 (60%) - Mixed market conditions
- Val: 2023 (15%) - Used for stacking
- Test: 2024-2026 (25%) - Gold bull run period

### Expected Metrics
After data leakage fix (March 2026):
```
Total Trades: 791
Win Rate: 68.2%
Total Return: 1446%
Sharpe Ratio: 3.86
Max Drawdown: 15.9%
Directional Accuracy: 50.6%
```

Important: 1446% return reflects gold's 2024-2026 bull run, not prediction skill. Directional accuracy of 50.6% shows the model barely beats random guessing.

### Interpreting High Returns
The strategy profits from volatility harvesting, not directional prediction:
- Gold increased 117% from 2024-2026
- Strategy is always in market (85% Short predictions)
- Captures volatile pullbacks during uptrend
- High leverage (100:1) amplifies returns

## ✅ Success Checklist

After running, you should see:
- [ ] data/processed/ohlcv.parquet exists
- [ ] data/processed/features.parquet exists
- [ ] data/processed/labels.parquet exists
- [ ] models/lightgbm_model.pkl exists
- [ ] models/lstm_model.pt exists
- [ ] models/lstm_norm_stats.npz exists (data leakage fix)
- [ ] results/thesis_report.md exists
- [ ] Report shows metrics for all stages

---

Questions? See Architecture.md for details
