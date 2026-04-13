# Documentation Hub

> Welcome to the documentation for **Hybrid Stacking (LSTM + LightGBM) for XAU/USD Trading Signal Prediction**.

---

## About This Project

This project predicts trading signals for Gold (XAU/USD) on the 1-hour timeframe using a hybrid stacking approach that combines two machine learning models:

- **LightGBM** — finds patterns across technical indicators
- **LSTM** — finds patterns in price sequences over time
- **Meta-Learner** — combines both models for a final prediction

**Three signals:** Long (buy), Short (sell), or Hold (wait).

---

## Documentation Sections

| Document | Description | When to Read |
|----------|-------------|-------------|
| [**Architecture**](architecture.md) | How the project is built — modules, data flow, design decisions | First — to understand the system |
| [**Quickstart**](quickstart.md) | Step-by-step guide to install, configure, and run the pipeline | When you want to run it |
| [**Evaluation**](evaluation.md) | How to interpret results, metrics, and charts | After your first pipeline run |
| [**Configuration**](configuration.md) | Features overview + detailed parameter tuning guide | When you want to optimize |
| [**Roadmap**](roadmap.md) | Completed tasks and planned improvements | To see project status |
| [**Glossary**](glossary.md) | Simple definitions for all technical terms | Anytime you see an unfamiliar term |

---

## Quick Links

### Getting Started
1. Read [Architecture](architecture.md) to understand the system
2. Follow [Quickstart](quickstart.md) to run your first pipeline
3. Check [Evaluation](evaluation.md) to understand the results

### Optimizing Results
1. Read the **Features** section in [Configuration](configuration.md)
2. Adjust parameters in `config.toml` using the **Configuration Guide**
3. Compare runs to find the best settings

### Understanding Terms
- Open [Glossary](glossary.md) for any technical term you are not sure about

---

## Project Structure

```
thesis/
├── config.toml          # All settings
├── main.py              # CLI entry point
├── data/
│   ├── raw/XAUUSD/      # Downloaded tick data
│   └── processed/       # Generated parquet files
├── src/thesis/          # Source code
│   ├── config/          # Configuration loader
│   ├── data/            # Data processing and splitting
│   ├── features/        # Technical indicators
│   ├── labels/          # Triple-Barrier labeling
│   ├── models/          # LSTM, LightGBM, Stacking
│   ├── pipeline/        # Stage orchestration
│   ├── backtest/        # CFD trading simulator
│   ├── reporting/       # Report generation
│   └── validation/      # Integrity checks
├── tests/               # Test suite
├── results/             # Session-based output
└── docs/                # This documentation
```

---

## Additional Resources

| Resource | Location |
|----------|----------|
| Data Download Guide | `docs/guide/Download_Guide.md` |
| Git Commit Guide | `docs/guide/Git_Commit_Guide.md` |
| Implementation Plan | `docs/planning/implementation_plan.md` |
| Project Details | `docs/planning/detail.md` |
| Agent Instructions | `AGENTS.md` (project root) |
