# Architecture

> How this project is built — a high-level overview for readers who want to understand the system before diving into code.

***

## 1. What This Project Does

This project predicts **trading signals for Gold (XAU/USD)** on the 1-hour timeframe. It uses a **hybrid stacking** approach that combines two different machine-learning models:

| Model                                 | What It Sees                                    | Why It Helps                                          |
| ------------------------------------- | ----------------------------------------------- | ----------------------------------------------------- |
| **LSTM** (neural network)             | Sequences of 120 candles (5 days of OHLCV data) | Good at finding patterns over time                    |
| **LightGBM** (gradient-boosted trees) | 20+ technical indicator features                | Good at finding patterns across many features at once |

A **meta-learner** (on top) takes the predictions from both models and makes a final decision: **Long** (buy), **Short** (sell), or **Hold** (do nothing).

***

## 2. The Big Picture

```
Raw Tick Data (Dukascopy)
        |
        v
  [1. Prepare]  →  Tick data → OHLCV H1 candles
        |
        v
  [2. Features]  →  Add 20+ technical indicators
        |
        v
  [3. Labels]   →  Triple-Barrier labeling (Long / Hold / Short)
        |
        v
  [4. Split]    →  Train (2018-2022) / Val (2023) / Test (2024-2026)
        |              with purge & embargo to prevent data leakage
        |
   +----+----+
   |         |
   v         v
[5. LightGBM]  [6. LSTM]     ← Base models train independently
   |         |
   +----+----+
        |
        v
  [7. Stacking]  →  Meta-learner combines both models
        |
        v
  [8. Backtest]  →  Simulate CFD trades with realistic costs
        |
        v
  [9. Report]    →  Markdown report + SHAP plots + metrics
```

***

## 3. Project Structure

```
thesis/
├── config.toml          # All settings in one file
├── main.py              # CLI entry point
├── data_download.py     # Download raw tick data
├── data/
│   ├── raw/XAUUSD/      # Monthly tick files from Dukascopy
│   └── processed/       # Generated parquet files (OHLCV, features, labels, splits)
├── src/thesis/          # Main source code
│   ├── config/          # Load and validate config.toml
│   ├── data/            # Tick-to-OHLCV conversion, data splitting
│   ├── features/        # Technical indicator engineering
│   ├── labels/          # Triple-Barrier labeling
│   ├── models/          # LSTM, LightGBM, stacking meta-learner
│   ├── pipeline/        # Stage orchestration + session management
│   ├── backtest/        # CFD trading simulator
│   ├── reporting/       # Thesis report generation with SHAP
│   └── validation/      # Label, math, and pipeline integrity checks
├── tests/               # Unit and integration tests
├── scripts/             # TUI dashboard
├── results/             # Session-based output folders
└── docs/                # Documentation (this folder)
```

***

## 4. Key Modules Explained

### 4.1 Config (`src/thesis/config/`)

Reads `config.toml` and converts it into Python dataclasses. Every setting — from data paths to model hyperparameters — lives in one file. You can also override any setting with environment variables using the prefix `THESIS_`.

**Key classes:**

- `DataConfig` — raw data path, timeframe, timezone
- `SplittingConfig` — train/val/test dates, purge/embargo bars
- `FeaturesConfig` — which indicators to compute, periods, thresholds
- `LabelsConfig` — triple-barrier parameters (ATR multipliers, horizon)
- `TreeModelConfig` — LightGBM hyperparameters and Optuna settings
- `LSTMModelConfig` — LSTM architecture, training epochs, learning rate
- `StackingConfig` — meta-learner type and calibration method
- `BacktestConfig` — spread, slippage, leverage, risk management
- `Config` — combines all of the above

### 4.2 Data (`src/thesis/data/`)

| File               | Purpose                                                                                                                                         |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `tick_to_ohlcv.py` | Converts raw tick data into 1-hour candles (OHLCV). Validates data quality (gaps, zero volume, price spikes).                                   |
| `splitting.py`     | Splits data into train/val/test sets. Uses **purge bars** (gap between sets) and **embargo bars** (buffer after train) to prevent data leakage. |

### 4.3 Features (`src/thesis/features/`)

Generates 20+ features grouped into categories:

| Category             | Examples                                                     |
| -------------------- | ------------------------------------------------------------ |
| Technical indicators | EMA (34, 89), RSI (14), MACD (12,26,9), ATR (14)             |
| Pivot points         | Daily pivot, R1, S1                                          |
| Session encoding     | Asia (00-08 UTC), London (08-17 UTC), NY PM (17-21 UTC)      |
| Lag features         | Close price at t-1, t-2, t-3, t-5, t-10                      |
| Spread features      | Bid-ask spread metrics                                       |
| Microstructure       | Candlestick patterns (engulfing, doji, hammer), volume delta |

Highly correlated features (above 0.90) are automatically removed.

### 4.4 Labels (`src/thesis/labels/`)

Uses the **Triple-Barrier Method** to assign labels:

- **Take Profit (TP)** = Close + 1.5 x ATR
- **Stop Loss (SL)** = Close - 1.5 x ATR
- **Horizon** = 20 bars (maximum wait time)

For each bar, the system checks future prices bar-by-bar:

- If price hits TP first → label is **Long (+1)**
- If price hits SL first → label is **Short (-1)**
- If neither is hit within 20 bars → label is **Hold (0)**
- If TP and SL are hit at the same time → label is **Hold (0)** (conservative)

### 4.5 Models (`src/thesis/models/`)

| File                  | Purpose                                                                                                                                                                                           |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lightgbm_model.py`   | Trains a LightGBM classifier. Can use **Optuna** (up to 100 trials) for automatic hyperparameter tuning. Uses class weights to handle imbalanced labels.                                          |
| `lstm_model.py`       | Trains a PyTorch LSTM neural network. Input: 120-bar sequences of OHLCV data. Uses early stopping with patience=10.                                                                               |
| `stacking.py`         | Combines LightGBM and LSTM predictions using a **meta-learner** (default: logistic regression). Uses time-aware cross-validation to avoid leakage. Applies a confidence threshold (default: 0.6). |
| `cross_validation.py` | Walk-forward cross-validation that respects time ordering. Supports purge and embargo gaps.                                                                                                       |

### 4.6 Backtest (`src/thesis/backtest/`)

A CFD (Contract for Difference) trading simulator that models realistic trading conditions:

- **Spread:** 2 pips
- **Slippage:** 1 pip
- **Leverage:** 50:1
- **Risk per trade:** 1% of capital
- **Risk management:** margin call at 50%, stop-out at 20%, max daily loss 5%, trailing stop

### 4.7 Reporting (`src/thesis/reporting/`)

Generates a Markdown thesis report including:

- Executive summary with key metrics
- Data and model configuration
- Backtest results with equity curve
- SHAP analysis for feature importance
- Model disagreement analysis (when LightGBM and LSTM disagree)
- Confidence histogram

### 4.8 Validation (`src/thesis/validation/`)

| File                    | Purpose                                                                                          |
| ----------------------- | ------------------------------------------------------------------------------------------------ |
| `label_validation.py`   | Checks for lookahead bias in labels (i.e., the model should not see future data during training) |
| `math_consistency.py`   | Validates backtest math: Kelly criterion, expected return, Calmar ratio                          |
| `pipeline_integrity.py` | Verifies that all pipeline outputs are consistent and complete                                   |

***

## 5. Data Flow

```
Raw ticks (.parquet)
    │
    ▼  tick_to_ohlcv.py
OHLCV H1 candles
    │
    ▼  engineering.py
Feature-enriched OHLCV (20+ columns)
    │
    ├──► triple_barrier.py → Labels (Long/Hold/Short)
    │
    ▼  splitting.py
Train / Val / Test splits (with purge + embargo)
    │
    ├──► LightGBM ──► probabilities (3 classes)
    │
    ├──► LSTM ──────► probabilities (3 classes)
    │
    ▼  stacking.py
Meta-learner combines → final predictions
    │
    ▼  cfd_simulator.py
Backtest results (trades, equity curve, metrics)
    │
    ▼  thesis_report.py
Markdown report + charts
```

***

## 6. Session-Based Output

Each pipeline run creates a session folder:

```
results/XAUUSD_1H_20260406_102913/
├── config/
│   ├── config_snapshot.toml    # Exact config used for this run
│   └── session_info.json       # Timestamps, durations, seed
├── models/
│   ├── lightgbm_model.pkl      # Trained LightGBM model
│   ├── lstm_model.pt           # Trained LSTM weights
│   ├── lstm_norm_stats.npz     # Normalization statistics
│   └── stacking_meta_learner.pkl  # Trained meta-learner
├── predictions/
│   ├── lightgbm_oof.parquet    # LightGBM out-of-fold predictions
│   ├── lstm_oof.parquet        # LSTM out-of-fold predictions
│   ├── stacking_predictions.parquet  # Final stacked predictions
│   └── final_predictions.parquet     # All predictions combined
├── reports/
│   ├── thesis_report.md        # Full thesis report
│   ├── thesis_report.json      # Metrics in machine-readable format
│   ├── shap_summary.png        # Feature importance plot
│   ├── confidence_histogram.png # Prediction confidence chart
│   └── model_disagreement.png  # Where models disagree
├── backtest/
│   ├── backtest_results.json   # Full backtest metrics
│   └── trades_detail.csv       # Trade-by-trade log
└── logs/
    └── pipeline_20260406_102913.log  # Detailed execution log
```

A `results/latest` symlink always points to the most recent session.

***

## 7. Design Decisions

| Decision                         | Reason                                                                 |
| -------------------------------- | ---------------------------------------------------------------------- |
| **Polars** instead of Pandas     | Faster data processing, better memory use                              |
| **TOML** for configuration       | Simple, readable, one file for everything                              |
| **Session folders**              | Every run is isolated and reproducible                                 |
| **Purge + Embargo**              | Prevents data leakage in time-series splitting                         |
| **Walk-Forward CV**              | Respects time ordering, simulates real deployment                      |
| **Confidence threshold (0.6)**   | Avoids low-confidence trades, reduces noise                            |
| **Triple-Barrier labeling**      | Better than fixed-pip labels because it adapts to volatility (via ATR) |
| **Stacking (not simple voting)** | Learns the optimal combination of model predictions                    |

***

## 8. Dependencies

| Library              | Purpose                                     |
| -------------------- | ------------------------------------------- |
| Polars               | Fast data manipulation                      |
| NumPy                | Numerical operations                        |
| PyTorch              | LSTM neural network                         |
| LightGBM             | Gradient-boosted tree model                 |
| scikit-learn         | Stacking meta-learner, metrics              |
| Optuna               | Automatic hyperparameter tuning             |
| SHAP                 | Model interpretability (feature importance) |
| TA-Lib               | Technical indicator calculations            |
| Matplotlib / Seaborn | Charts and plots                            |
| pytest               | Testing                                     |
| Ruff                 | Code formatting and linting                 |
| Textual              | Terminal dashboard                          |

