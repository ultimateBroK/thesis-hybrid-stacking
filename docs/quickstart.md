# Quickstart

> Step-by-step guide to get the project running from zero to results.

---

## Prerequisites

| Requirement | Version | How to Check |
|-------------|---------|-------------|
| **Pixi** (package manager) | Latest | `pixi --version` |
| **Python** | 3.13+ | Managed by Pixi (no need to install separately) |
| **Git** | Any | `git --version` |
| **Disk space** | ~15 GB | For raw data + models + results |

> **Note:** Pixi handles all Python dependencies automatically. You do **not** need to install Python, PyTorch, or any library yourself.

---

## Step 1: Get the Code

```bash
# Clone the repository (or download and extract the ZIP)
git clone <repository-url>
cd thesis
```

---

## Step 2: Install Dependencies

```bash
# Pixi will download and install everything
pixi install
```

This installs all required libraries: Polars, PyTorch, LightGBM, scikit-learn, Optuna, SHAP, TA-Lib, and more. The first run may take a few minutes.

---

## Step 3: Download Raw Data

The project uses **tick-level gold price data** from Dukascopy (a forex data provider). You need to download it before running the pipeline.

```bash
# Download all available data (2018-01 to current month)
pixi run data

# Or download for a specific year/month
python data_download.py --symbol XAUUSD --start-year 2024 --start-month 1
```

**What happens:**
- Tick data files are saved to `data/raw/XAUUSD/` as monthly parquet files
- Each file covers one month of trading
- Total download size: ~10 GB for the full 2018-2026 range
- Download takes 10-30 minutes depending on your internet speed

> **Tip:** For a quick test, download just 1-2 years of data:
> ```bash
> python data_download.py --symbol XAUUSD --start-year 2025 --start-month 1
> ```

For detailed download instructions, see `docs/guide/Download_Guide.md`.

---

## Step 4: Configure the Pipeline

All settings are in **`config.toml`** at the project root. For your first run, the defaults should work fine. Here is what you might want to check:

```toml
# config.toml — key settings for your first run

[data]
timeframe = "1H"                  # 1-hour candles
start_date = "2018-01-01"         # Data range start
end_date = "2026-03-31"           # Data range end

[splitting]
train_start = "2018-01-01"        # Training period
train_end = "2022-12-31 23:59:59"
val_start = "2023-01-01"          # Validation period
val_end = "2023-12-31 23:59:59"
test_start = "2024-01-01"         # Test period (out-of-sample)
test_end = "2026-03-31 23:59:59"

[models.lstm]
epochs = 50                        # Max training epochs
device = "cpu"                     # Use "cuda" if you have a GPU

[models.tree]
use_optuna = true                  # Auto-tune hyperparameters
optuna_trials = 100                # Number of tuning trials

[workflow]
random_seed = 42                   # For reproducibility
```

> **GPU users:** Change `device = "cuda"` in `[models.lstm]` to speed up LSTM training significantly.

---

## Step 5: Run the Full Pipeline

```bash
# Run the complete pipeline (all 9 stages)
pixi run workflow
```

This command runs all stages in order:

```
📊 Stage 1: Prepare     → Tick data → OHLCV H1 candles
🔧 Stage 2: Features    → Add 20+ technical indicators
🏷️ Stage 3: Labels      → Triple-Barrier labeling
✂️ Stage 4: Split       → Train / Val / Test split
🌳 Stage 5: LightGBM    → Train gradient-boosted tree model
🧠 Stage 6: LSTM        → Train neural network
🥞 Stage 7: Stacking    → Combine both models
📈 Stage 8: Backtest    → Simulate CFD trading
📄 Stage 9: Report      → Generate thesis report
```

**Expected runtime:**
| Stage | CPU | GPU |
|-------|-----|-----|
| Prepare | 2-5 min | 2-5 min |
| Features | 1-2 min | 1-2 min |
| Labels | 1-2 min | 1-2 min |
| LightGBM (with Optuna) | 30-60 min | 30-60 min |
| LSTM | 20-40 min | 5-10 min |
| Stacking | 2-5 min | 2-5 min |
| Backtest | 1-2 min | 1-2 min |
| Report | 1-2 min | 1-2 min |

---

## Step 6: View the Results

Results are saved in a session folder under `results/`:

```bash
# The 'latest' symlink always points to the most recent session
ls results/latest/
```

### Key Output Files

| File | What It Contains |
|------|-----------------|
| `reports/thesis_report.md` | Full thesis report with metrics and analysis |
| `reports/shap_summary.png` | Feature importance chart |
| `reports/confidence_histogram.png` | How confident the model is in its predictions |
| `backtest/backtest_results.json` | All trading metrics (Sharpe, drawdown, etc.) |
| `backtest/trades_detail.csv` | Trade-by-trade log |
| `config/config_snapshot.toml` | Exact config used for this run |

### Read the Report

```bash
# View the report in your terminal
cat results/latest/reports/thesis_report.md

# Or open it in a Markdown viewer / VS Code
code results/latest/reports/thesis_report.md
```

---

## Step 7: Run Individual Stages

You can also run specific stages instead of the full pipeline. Edit `config.toml`:

```toml
[workflow]
run_data_pipeline = true
run_feature_engineering = true
run_label_generation = true
run_data_splitting = true
run_lightgbm = true
run_lstm = false          # Skip LSTM
run_stacking = false      # Skip stacking
run_backtest = false      # Skip backtest
run_reporting = false     # Skip reporting
```

Then run:

```bash
pixi run workflow
```

Or use the `--force` flag to re-run everything from scratch:

```bash
pixi run force
```

---

## Step 8: Run Tests

```bash
# Run all tests
pixi run test

# Run only unit tests
pixi run pytest tests/unit/

# Run a specific test file
pixi run pytest tests/unit/test_features/test_engineering.py

# Run a specific test function
pixi run pytest tests/unit/test_labels/test_triple_barrier.py::test_label_values

# Run with markers
pixi run pytest -m "not slow"         # Exclude slow tests
pixi run pytest -m unit                # Only unit tests
pixi run pytest -m integration         # Only integration tests
```

---

## Common Commands Reference

| Command | What It Does |
|---------|-------------|
| `pixi run workflow` | Run the full pipeline |
| `pixi run force` | Run the full pipeline from scratch |
| `pixi run data` | Download latest month of data |
| `pixi run test` | Run all tests |
| `pixi run lint` | Check code style |
| `pixi run format` | Auto-format code |
| `pixi run clean-cache` | Remove split/label cache files |
| `pixi run clean-all` | Remove all generated files |
| `pixi run dashboard` | Open terminal dashboard |

---

## Troubleshooting

### "No module named thesis"
Make sure you ran `pixi install` and are using `pixi run` commands.

### "FileNotFoundError: data/raw/XAUUSD"
You need to download data first. Run `pixi run data` or see `docs/guide/Download_Guide.md`.

### LSTM training is very slow
If you have a GPU, set `device = "cuda"` in `[models.lstm]`. Otherwise, reduce `epochs` to `20` or `sequence_length` to `60` for a quicker test.

### Optuna takes too long
Reduce the number of trials:
```toml
[models.tree]
optuna_trials = 20    # Instead of 100
```

### Out of memory
Try reducing the LSTM batch size or sequence length:
```toml
[models.lstm]
batch_size = 64       # Instead of 128
sequence_length = 60  # Instead of 120
```

---

## Using the Jupyter Notebook

The project includes a Jupyter notebook for interactive exploration:

```bash
# Start Jupyter Lab
pixi run jupyter lab

# Then open main.ipynb
```

The notebook provides a cell-by-cell interface to run pipeline stages and visualize results interactively.
