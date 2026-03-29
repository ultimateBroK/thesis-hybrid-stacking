# Commit Message Style Guide

This project follows the [Conventional Commits](https://www.conventionalcommits.org/) specification.

## Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

## Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style (formatting, semicolons, etc.)
- `refactor`: Code refactoring (no functional change)
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks, dependencies
- `ci`: CI/CD configuration changes
- `build`: Build system or external dependency changes

## Scopes

- `data`: Data pipeline, OHLCV conversion
- `features`: Feature engineering
- `labels`: Triple-barrier labeling
- `models`: LSTM, LightGBM, stacking models
- `backtest`: Backtesting engine
- `reporting`: Report generation
- `config`: Configuration management
- `pipeline`: Workflow orchestration
- `deps`: Dependencies (pixi, pip)
- `docs`: Documentation
- `readme`: README.md updates
- `github`: GitHub Actions, templates

## Examples

### Feature Addition
```
feat(models): add hyperparameter optimization for LightGBM using Optuna

Implemented Optuna-based hyperparameter tuning for LightGBM classifier.
- Added study configuration with 100 trials
- Optimized: max_depth, learning_rate, n_estimators
- Reduced overfitting by 15% on validation set

Closes #42
```

### Bug Fix
```
fix(data): correct timezone handling in tick_to_ohlcv conversion

Fixed issue where timestamps were not properly converted to UTC.
- Added explicit timezone normalization
- Updated test cases to verify timezone handling
- Added warning for ambiguous timestamps

Fixes #38
```

### Documentation
```
docs(readme): update installation instructions for Pixi workflow

- Added Pixi installation steps
- Updated environment setup section
- Included troubleshooting guide
```

### Refactoring
```
refactor(features): extract technical indicator calculation to separate module

Moved indicator calculations from engineering.py to indicators.py
- Created new indicators submodule
- Improved code organization
- No functional changes

See: Architecture decision record #3
```

### Performance
```
perf(pipeline): parallelize feature engineering with Polars

- Leveraged Polars multiprocessing
- Reduced feature engineering time from 45s to 12s
- Added n_jobs configuration option

Benchmarks: See results/performance-benchmark-2026-03-29.md
```

### Test Addition
```
test(backtest): add unit tests for CFD simulation edge cases

- Test zero spread scenario
- Test maximum leverage limits
- Test margin call conditions
- Achieved 95% code coverage for backtest module
```

### Dependency Update
```
chore(deps): update PyTorch to 2.10.0 for CUDA 12 support

Updated dependencies:
- pytorch: 2.9.0 → 2.10.0
- torchvision: 0.20.0 → 0.21.0

BREAKING CHANGE: Requires CUDA 12 compatible GPU
```

## Anti-Patterns to Avoid

❌ Bad:
```
fix: fixed stuff
WIP
asdfasdf
fixed bug in thing
```

✅ Good:
```
fix(labels): resolve off-by-one error in triple-barrier horizon

Corrected horizon calculation that was excluding final bar.
Tests updated to verify correct barrier placement.

Fixes #51
```

## Pre-commit Hook

Install the pre-commit hook to automatically validate commit messages:

```bash
pixi add pre-commit
pre-commit install
```

## Tools

### Validate Commit Messages Locally

```bash
# Check last commit message
git log -1 --pretty=%B | grep -E "^(feat|fix|docs|style|refactor|perf|test|chore|ci|build)\("

# Interactive rebase to edit old commits
git rebase -i HEAD~5
```

### VS Code Extension

Install the "Conventional Commits" extension for VS Code to help write compliant messages.
