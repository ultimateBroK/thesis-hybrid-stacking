# Git Commit Guide - Hybrid Stacking Thesis

## Quick Reference Card

### Before You Start (One-Time Setup)

```bash
# Set your identity (matches your GitHub account)
git config --global user.name "Hieu Nguyen"
git config --global user.email "hieuteo03@gmail.com"

# Verify settings
git config --list | grep -E "(user.name|user.email)"
```

---

## Standard Commit Workflow

### Step 1: Check Status
```bash
# See what files changed
git status

# See detailed changes
git diff

# See staged changes (after git add)
git diff --cached
```

### Step 2: Stage Changes
```bash
# Stage specific file
git add filename.py

# Stage multiple files
git add file1.py file2.py

# Stage all changes (use with caution)
git add .

# Stage only modified files (not new untracked)
git add -u
```

### Step 3: Commit
```bash
# Commit with message
git commit -m "type(scope): description

Detailed explanation here (optional)"
```

### Step 4: Push to GitHub
```bash
# Push current branch
git push origin main

# Or if on feature branch
git push origin branch-name
```

---

## Commit Message Format (Conventional Commits)

### Structure
```
type(scope): subject

[optional body]

[optional footer]
```

### Types for Thesis Project

| Type | Use When | Example |
|------|----------|---------|
| `feat` | Add new feature/model | `feat(models): add attention mechanism to LSTM` |
| `fix` | Fix bugs/errors | `fix(data): handle missing values in feature engineering` |
| `docs` | Documentation changes | `docs: update Architecture.md with pipeline diagram` |
| `refactor` | Code restructuring | `refactor(models): simplify cross-validation logic` |
| `test` | Add/modify tests | `test: add unit tests for triple barrier labeling` |
| `chore` | Maintenance tasks | `chore(deps): update requirements.txt` |
| `perf` | Performance improvements | `perf(features): optimize technical indicator calculations` |

### Scope Guidelines
- `data` - Data pipeline (splitting, features, labels)
- `models` - ML models (lightgbm, lstm, stacking)
- `backtest` - Backtesting engine
- `reporting` - Reports and visualization
- `config` - Configuration files
- `ci` - GitHub Actions/workflows
- `docs` - Documentation

### Examples

```bash
# Feature addition
git commit -m "feat(models): implement early stopping for LightGBM

Add early_stopping_rounds parameter to prevent overfitting.
Monitors validation metric and stops when no improvement."

# Bug fix
git commit -m "fix(labels): correct triple barrier calculation

Volatility calculation was using wrong window size.
Changed from 20-day to 10-day lookback as per paper."

# Documentation
git commit -m "docs: add experiment log for 2026-03-29

Record results of LSTM hyperparameter tuning.
Best config: 50 epochs, 0.001 lr, 128 batch."

# Configuration update
git commit -m "chore(config): extend training date range

Update end_date from 2025-01 to 2025-12
for additional 12 months of training data."
```

---

## Common Scenarios

### Scenario 1: Fix a Bug You Just Found

```bash
# Check what changed
git status
git diff

# Stage the fix
git add src/thesis/models/lstm_model.py

# Commit with context
git commit -m "fix(models): resolve shape mismatch in LSTM forward pass

Input tensor shape (batch, seq, features) was not matching
expected (seq, batch, features). Added permute() operation.

Error was: RuntimeError: mat1 and mat2 shapes cannot be multiplied"

# Push
git push origin main
```

### Scenario 2: Add New Feature Across Multiple Files

```bash
# Check all changes
git status

# Stage related files
git add src/thesis/features/new_indicator.py
git add src/thesis/config/loader.py
git add tests/test_new_indicator.py

# Write comprehensive commit message
git commit -m "feat(features): add RSI momentum indicator

Implement Relative Strength Index calculation with:
- Configurable lookback period (default: 14)
- Overbought/oversold level detection
- Integration with feature pipeline

Files changed:
- new_indicator.py: core implementation
- loader.py: add RSI configuration schema
- test_new_indicator.py: unit tests with 95% coverage"

# Push
git push origin main
```

### Scenario 3: Update Documentation Only

```bash
# Stage doc files
git add docs/Architecture.md
git add README.md

# Simple commit for docs
git commit -m "docs: update Architecture.md with data flow diagram

Add Mermaid diagram showing:
- Raw tick → OHLCV transformation
- Feature engineering pipeline
- Model training flow
- Backtesting process"

git push origin main
```

### Scenario 4: Revert a Bad Commit (Emergency)

```bash
# See recent commits
git log --oneline -5

# Revert specific commit (creates new commit)
git revert abc1234

# Push the revert
git push origin main

# Or reset last commit (if not pushed yet)
git reset --soft HEAD~1
# Make fixes, then commit again
```

---

## Best Practices for Thesis Project

### 1. Commit Often, Push Regularly
```bash
# Good: Small, focused commits
git commit -m "feat(data): implement tick aggregation for 1H bars"

# Bad: Giant commit with everything
git commit -m "lots of changes"  # ❌ Don't do this
```

### 2. Write Meaningful Messages
```bash
# Good: Explains WHAT and WHY
git commit -m "fix(labels): adjust triple barrier thresholds

Lower volatility multiplier from 2.0 to 1.5 to reduce
false positive signals during high volatility periods."

# Bad: Vague, no context
git commit -m "fix stuff"  # ❌ Don't do this
```

### 3. Verify Before Pushing
```bash
# Review your changes
git diff --cached

# Check files being committed
git status

# If something wrong, unstage
git reset HEAD filename.py

# Or amend last commit (if not pushed)
git commit --amend -m "new message"
```

### 4. Keep Main Branch Clean
```bash
# Never push broken code to main
# Test locally first:
pixi run test  # or run your tests
python main.py  # verify pipeline works

# Then push
git push origin main
```

---

## Troubleshooting

### Problem: Wrong Author in Commit

```bash
# Check current author
git log -1 --format="%an <%ae>"

# If wrong, fix it:
git config user.name "Hieu Nguyen"
git config user.email "hieuteo03@gmail.com"

# Amend last commit with new author
git commit --amend --author="Hieu Nguyen <hieuteo03@gmail.com>" --no-edit

# Push with force (only for unpushed commits!)
git push origin main --force-with-lease
```

### Problem: Accidentally Committed Sensitive Data

```bash
# Don't push! Remove from last commit
git reset --soft HEAD~1

# Add file to .gitignore
echo "secret.txt" >> .gitignore
git add .gitignore

# Stage and commit again (without sensitive file)
git add .
git commit -m "feat: add new feature"
```

### Problem: Merge Conflict

```bash
# When pulling/pushing shows conflict

# 1. See conflicted files
git status

# 2. Edit files to resolve (look for <<<<<<< markers)

# 3. Mark as resolved
git add resolved_file.py

# 4. Complete merge
git commit -m "merge: resolve conflict in model training"
```

---

## Quick Commands Cheat Sheet

| Task | Command |
|------|---------|
| See status | `git status` |
| Stage file | `git add file.py` |
| Stage all | `git add .` |
| Unstage file | `git reset HEAD file.py` |
| Commit | `git commit -m "message"` |
| Push | `git push origin main` |
| Pull updates | `git pull origin main` |
| See log | `git log --oneline -10` |
| See changes | `git diff` |
| See staged | `git diff --cached` |
| Undo last commit (keep changes) | `git reset --soft HEAD~1` |
| Amend commit | `git commit --amend -m "new msg"` |
| Discard changes | `git checkout -- file.py` |

---

## Daily Workflow Example

```bash
# 1. Start work - pull latest changes
git pull origin main

# 2. Make changes to files...
#    (edit src/thesis/models/lstm_model.py)

# 3. Check what changed
git status
git diff

# 4. Stage your changes
git add src/thesis/models/lstm_model.py

# 5. Commit with good message
git commit -m "feat(models): add dropout regularization to LSTM

Add 0.2 dropout between LSTM layers to prevent overfitting.
Validation accuracy improved from 0.72 to 0.78."

# 6. Push to GitHub
git push origin main

# 7. Verify on GitHub
#    Check: https://github.com/ultimateBroK/thesis-hybrid-stacking/commits/main
```

---

## Emergency Contacts / Help

- **GitHub Actions Status**: https://github.com/ultimateBroK/thesis-hybrid-stacking/actions
- **Repository URL**: https://github.com/ultimateBroK/thesis-hybrid-stacking
- **Pro Git Book**: https://git-scm.com/book/en/v2

---

*Created: 2026-03-29*  
*For: Hybrid Stacking Thesis - XAU/USD Trading System*