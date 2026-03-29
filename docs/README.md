# Project Documentation Guide

Welcome! This folder contains everything you need to understand and use this thesis project.

## 📚 Documentation Map

Start here based on what you need:

| If you want to... | Read this |
|-------------------|-----------|
| Run the project quickly | Quickstart.md |
| Understand how it works | Architecture.md |
| Know what features exist | Features.md |
| Change settings/numbers | Config.md |
| Read the results | Evaluation.md |
| See what's done/todo | Todo.md |

## 🎯 Quick Links

- Main code: src/thesis/
- Config file: config.toml
- Run command: python main.py
- Results: results/ folder

## 📖 Reading Order (Recommended)

For beginners:
1. Quickstart.md - Get it running
2. Architecture.md - Understand the flow
3. Evaluation.md - Read your results

For researchers:
1. Architecture.md - Deep dive
2. Features.md - What signals are used
3. Config.md - Tune parameters
4. Evaluation.md - Interpret results

For developers:
1. Todo.md - What's implemented
2. Architecture.md - Code structure
3. Features.md - Feature engineering

## 🔬 Recent Updates (March 2026)

### Data Leakage Fix
Fixed LSTM normalization data leakage on March 28, 2026:
- LSTM now saves training statistics (models/lstm_norm_stats.npz)
- Test predictions use training stats (not test stats)
- Returns dropped from 1622% to 1446% (-10.8%)

### Key Finding
1446% return is not from prediction skill. It results from:
- Gold's 117% price increase (2024-2026 bull run)
- Volatility harvesting strategy
- Always-in-market approach
- Directional accuracy only 50.6% (barely better than random)

See Evaluation.md for detailed analysis.

## 🎓 Project Info

- Student: Nguyen Duc Hieu (2151061192)
- Advisor: Hoang Quoc Dung
- University: Thuy Loi University
- Topic: Hybrid Stacking (LSTM + LightGBM) for XAU/USD Trading

## 🚀 One-Line Summary

Raw ticks → Features → Labels → Train LGBM + LSTM → Stack → Backtest → Report

---

All docs use simple English. Examples included!
