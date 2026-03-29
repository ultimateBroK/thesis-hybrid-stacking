# GitHub Actions Guide - Hybrid Stacking Thesis

## Table of Contents
1. [What are GitHub Actions?](#what-are-github-actions)
2. [Workflows Overview](#workflows-overview)
3. [CI Pipeline (ci.yml)](#ci-pipeline-ciyml)
4. [Develop Branch (develop.yml)](#develop-branch-develyml)
5. [Release Automation (release.yml)](#release-automation-releaseyml)
6. [How Triggers Work](#how-triggers-work)
7. [Reading Workflow Results](#reading-workflow-results)
8. [Troubleshooting](#troubleshooting)
9. [Common Tasks](#common-tasks)

---

## What are GitHub Actions?

GitHub Actions is a **CI/CD (Continuous Integration/Continuous Deployment)** platform that automatically runs tasks when you push code to GitHub.

### Why we use it:
- ✅ **Automated Testing** - Runs tests every time you push
- ✅ **Code Quality** - Checks linting, formatting, security
- ✅ **Release Automation** - Creates releases automatically
- ✅ **Early Error Detection** - Catches issues before they reach main

### Where to see them:
- **GitHub Repository** → **Actions** tab: https://github.com/ultimateBroK/thesis-hybrid-stacking/actions
- **Pull Requests** - Checks appear at the bottom
- **Commits** - Status icons next to each commit

---

## Workflows Overview

This repository has **3 workflows**:

| Workflow | File | When It Runs | Purpose |
|----------|------|--------------|---------|
| **CI Pipeline** | `ci.yml` | Every push (except main/develop), every PR | Code quality, tests, security |
| **Develop Branch** | `develop.yml` | Every push to `develop` branch | Integration tests, staging checks |
| **Release** | `release.yml` | Every push to `main` branch | Create GitHub releases, deploy artifacts |

---

## CI Pipeline (ci.yml)

### Purpose
Runs on every code change to ensure quality before merging.

### When It Triggers
```yaml
on:
  push:
    branches-ignore:  # Run on all branches EXCEPT:
      - main
      - develop
  pull_request:       # Also run on PRs to:
    branches:
      - develop
      - main
```

### Jobs (5 parallel checks)

#### 1. Lint & Format Check
**What it does:**
- Runs Ruff linter to check code style
- Verifies code formatting
- Validates commit messages (on PRs only)

**Failure means:** Your code has style issues or isn't formatted properly

**How to fix:**
```bash
pixi run lint      # See issues
pixi run format    # Auto-fix formatting
```

#### 2. Test Suite
**What it does:**
- Creates sample test files
- Runs pytest with coverage
- Uploads coverage to Codecov

**Failure means:** Tests are failing or coverage is too low

**How to fix:**
```bash
pixi run test      # Run tests locally
```

#### 3. Security Scan
**What it does:**
- Scans for hardcoded passwords/API keys
- Checks for secrets in code
- Validates .gitignore compliance

**Failure means:** You committed sensitive data

**How to fix:**
- Remove secrets from code
- Use environment variables instead
- Add sensitive files to .gitignore

#### 4. Code Quality
**What it does:**
- Runs Vulture to find dead code
- Generates code statistics

**Warnings are OK** - Some unused code is expected in research projects

#### 5. Pipeline Validation
**What it does:**
- Loads config.toml to verify it's valid
- Checks all required directories exist
- Verifies critical files are present

**Failure means:** Project structure is broken or config is invalid

---

## Develop Branch (develop.yml)

### Purpose
Comprehensive testing for the `develop` branch before merging to main.

### When It Triggers
```yaml
on:
  push:
    branches:
      - develop  # Only when pushing to develop branch
```

### Jobs

#### 1. Integration Tests
**What it does:**
- Creates minimal test data
- Tests all module imports
- Validates individual pipeline stages

**Tests these imports:**
```python
from thesis.data import tick_to_ohlcv
from thesis.features import engineering
from thesis.labels import triple_barrier
from thesis.models import lightgbm_model, lstm_model
```

#### 2. Staging Deployment Check
**What it does:**
- Verifies main.py is present
- Checks required Pixi tasks exist
- Validates documentation completeness

#### 3. Notify Develop Push
**What it does:**
- Logs notification (could integrate with Slack/Discord in production)

---

## Release Automation (release.yml)

### Purpose
Automatically creates GitHub releases when code is pushed to main.

### When It Triggers
```yaml
on:
  push:
    branches:
      - main  # Only on main branch pushes
```

### Jobs

#### 1. Final Validation
**What it does:**
- Runs complete test suite
- Validates version from `__init__.py`
- Verifies documentation exists
- Creates validation report

#### 2. Create GitHub Release
**What it does:**
- Gets version from code
- Generates changelog from commits
- Creates Git tag (e.g., `v0.1.0`)
- Creates GitHub Release with:
  - Release notes
  - Installation instructions
  - Feature list
  - Links to documentation

#### 3. Deploy Artifacts
**What it does:**
- Packages source code
- Creates `thesis-source.tar.gz`
- Uploads to GitHub Release

#### 4. Notify Release
**What it does:**
- Logs release summary
- Shows workflow status
- Provides release URL

---

## How Triggers Work

### Visual Flow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Push to        │     │  Push to        │     │  Push to        │
│  feature/xyz    │────▶│  develop        │────▶│  main           │
│  (any branch)   │     │  branch         │     │  branch         │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  ci.yml         │     │  develop.yml    │     │  release.yml    │
│  - Lint         │     │  - Integration  │     │  - Validation   │
│  - Test         │     │    tests        │     │  - Create tag   │
│  - Security     │     │  - Staging      │     │  - GitHub       │
│  - Quality      │     │    checks       │     │    Release      │
│  - Validation   │     │                 │     │  - Deploy       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Trigger Examples

#### Example 1: Working on Feature Branch
```bash
# You push to feature/new-model
git push origin feature/new-model

# What runs: ci.yml
# - Code quality checks
# - Tests
# - Security scan
```

#### Example 2: Creating Pull Request
```bash
# You create PR from feature/new-model → develop

# What runs: ci.yml (on PR)
# Same checks as above
# Plus conventional commit message validation
```

#### Example 3: Merging to Develop
```bash
# PR merged to develop branch

# What runs: develop.yml
# - Integration tests
# - Module import checks
# - Staging validation
```

#### Example 4: Releasing to Main
```bash
# PR merged to main branch

# What runs: release.yml
# - Final validation
# - Creates GitHub Release
# - Generates changelog
# - Uploads artifacts
```

---

## Reading Workflow Results

### Where to Find Results

#### Option 1: GitHub Website
1. Go to: https://github.com/ultimateBroK/thesis-hybrid-stacking/actions
2. Click on the workflow run
3. See all jobs and their status

#### Option 2: Commit Status Icons
- ✅ Green check = All passed
- ❌ Red X = Something failed
- 🟡 Yellow circle = Still running

#### Option 3: Pull Request Checks
- At bottom of PR, see "All checks have passed"
- Click "Show all checks" to see details

### Understanding Job Status

```
Workflow: CI Pipeline
├── ✅ Lint & Format Check      (Passed)
├── ❌ Test Suite               (Failed - pytest error)
├── ✅ Security Scan            (Passed)
├── ⚠️ Code Quality             (Passed with warnings)
└── ✅ Pipeline Validation      (Passed)
```

**Red X on any job = Whole workflow failed**

### Reading Logs

When a job fails:
1. Click on the failed job name
2. Expand the failed step
3. Read the error message

**Common error patterns:**
```
# Python syntax error
SyntaxError: invalid syntax

# Import error
ModuleNotFoundError: No module named 'thesis'

# Test failure
FAILED tests/test_placeholder.py::test_placeholder - AssertionError

# Lint error
src/models.py:42:1: E302 expected 2 blank lines
```

---

## Troubleshooting

### Problem: Workflow Not Starting

**Check:**
```bash
# Did you push to correct branch?
git branch

# Is workflow file valid?
git status  # Check if .github/workflows/ files changed

# Check Actions tab for errors
```

**Fix:**
- Make sure you're pushing to a branch (not just local commits)
- Check if workflow file was modified accidentally

---

### Problem: Lint Check Fails

**Error:**
```
Run pixi run lint
src/models.py:42:1: E302 expected 2 blank lines
Error: Process completed with exit code 1.
```

**Fix:**
```bash
# Auto-fix formatting
pixi run format

# Run lint again to verify
pixi run lint

# Stage and commit fixes
git add .
git commit -m "style: fix formatting issues"
git push origin main
```

---

### Problem: Tests Fail

**Error:**
```
FAILED tests/test_placeholder.py::test_import_thesis - ImportError
```

**Fix:**
```bash
# Run tests locally first
pixi run test

# If import errors, check:
python -c "import sys; sys.path.insert(0, 'src'); import thesis"

# Fix any issues, then commit
```

---

### Problem: Security Scan Finds Secrets

**Error:**
```
ERROR: Potential hardcoded password found
src/config.py: password = "secret123"
```

**Fix:**
```bash
# Remove hardcoded secrets
# Use environment variables instead:
# password = os.getenv("DB_PASSWORD")

# Add to .env (which is gitignored)
echo "DB_PASSWORD=secret123" >> .env

# Commit fix
git add src/config.py
git commit -m "fix(config): remove hardcoded password"
git push origin main
```

---

### Problem: Release Not Created

**Symptom:** Pushed to main but no GitHub Release

**Check:**
1. Go to Actions tab → Release workflow
2. Check if workflow ran
3. Look for errors in "Create GitHub Release" step

**Common causes:**
- Tag already exists with different commit
- Missing permissions
- Version mismatch

**Fix:**
```bash
# Check if tag exists
git tag -l

# Delete and recreate if needed
git tag -d v0.1.0
git push origin :refs/tags/v0.1.0

# Update version in code if needed
# Then push again
```

---

### Problem: "Node.js 20 Deprecated" Warning

**Status:** ✅ Already fixed in this repo

All workflows updated to:
- `actions/checkout@v6` (Node.js 24 compatible)
- `prefix-dev/setup-pixi@v0.9.4`
- `codecov/codecov-action@v5`
- `softprops/action-gh-release@v2`

---

## Common Tasks

### Task 1: See All Recent Workflow Runs

**GitHub:**
1. Go to https://github.com/ultimateBroK/thesis-hybrid-stacking/actions
2. See list of all runs
3. Filter by workflow type (CI, Develop, Release)

**Command Line:**
```bash
# Can't see directly from CLI, but can see commit status
git log --oneline --all | head -10
```

---

### Task 2: Re-run Failed Workflow

**GitHub:**
1. Go to Actions tab
2. Click on failed workflow
3. Click "Re-run jobs" button (top right)
4. Choose "Re-run all jobs" or "Re-run failed jobs"

---

### Task 3: Cancel Running Workflow

**GitHub:**
1. Go to Actions tab
2. Click on running workflow (yellow icon)
3. Click "Cancel workflow" button

---

### Task 4: Check Specific Job Details

**GitHub:**
1. Click on workflow run
2. Click on job name (e.g., "Lint & Format Check")
3. See all steps
4. Click on failed step to expand logs

---

### Task 5: Disable Workflow Temporarily

**GitHub:**
1. Go to Actions tab
2. Click on workflow name (e.g., "CI Pipeline")
3. Click "..." menu (top right)
4. Select "Disable workflow"

To re-enable, do the same and select "Enable workflow"

---

## Workflow File Locations

```
.github/
└── workflows/
    ├── ci.yml      # CI Pipeline
    ├── develop.yml # Develop branch
    └── release.yml # Release automation
```

**To modify workflows:**
1. Edit files in `.github/workflows/`
2. Commit and push
3. Changes take effect immediately on next trigger

---

## Key Environment Variables

Workflows use these variables:

| Variable | Meaning | Example |
|----------|---------|---------|
| `github.repository` | Full repo name | `ultimateBroK/thesis-hybrid-stacking` |
| `github.ref` | Branch/tag | `refs/heads/main` |
| `github.sha` | Commit hash | `b9c0297...` |
| `github.actor` | Who triggered | `ultimateBroK` |
| `secrets.CODECOV_TOKEN` | Secret token | (hidden) |

---

## Summary

### Quick Checklist

Before pushing code:
- [ ] Code runs locally: `python main.py`
- [ ] Tests pass: `pixi run test`
- [ ] Linting passes: `pixi run lint`
- [ ] Formatting OK: `pixi run format --check`

After pushing:
- [ ] Check Actions tab for status
- [ ] Fix any failures before merging
- [ ] Verify releases created (for main branch)

### Emergency Contacts

- **Actions Status:** https://github.com/ultimateBroK/thesis-hybrid-stacking/actions
- **Releases:** https://github.com/ultimateBroK/thesis-hybrid-stacking/releases
- **Documentation:** This guide + README.md

---

*Created: 2026-03-29*  
*For: Hybrid Stacking Thesis - GitHub Actions Documentation*