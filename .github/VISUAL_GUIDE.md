# Git Workflow - Visual Guide

## 🌳 Branch Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                         MAIN (stable)                            │
│  v1.0.0        v1.1.0        v1.2.0        v2.0.0               │
│    ●────────────●─────────────●─────────────●                   │
│         ↗              ↗             ↗                          │
└────────┼──────────────┼─────────────┼───────────────────────────┘
         │              │             │
         │    develop (integration)   │                           │
         └────●──────────●─────────────┘                           │
              │          │                                         │
              │    ┌─────┴─────────────────────────────────────┐   │
              │    │         FEATURE BRANCHES                  │   │
              │    │                                           │   │
              ├────┼─── feature/data-pipeline                  │   │
              │    │                                           │   │
              ├────┼─────── feature/model-training             │   │
              │    │                                           │   │
              ├────┼─────────── feature/backtesting            │   │
              │    │                                           │   │
              │    └─── feature/reporting                      │   │
              │                                                 │
              └─── bugfix/*                                     │
              └─── hotfix/* (emergency fixes)                  │
```

---

## 🔄 Complete Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DEVELOPER WORKFLOW                           │
└─────────────────────────────────────────────────────────────────────┘

    START
      │
      ▼
┌──────────────────┐
│ Checkout develop │
│ git pull origin  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Create feature   │
│ branch           │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Develop feature  │
│ + Commit often   │
│ Use conventional │
│ commit messages  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Sync with        │
│ develop (rebase) │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Run local checks │
│ ✓ lint           │
│ ✓ format         │
│ ✓ test           │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Push to GitHub   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Create PR        │
│ Fill template    │
└────────┬─────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│              GITHUB ACTIONS (Automatic)                 │
│  ┌─────────────────────────────────────────────────┐   │
│  │ CI PIPELINE                                     │   │
│  │  • Lint & Format Check                          │   │
│  │  • Test Suite                                   │   │
│  │  • Security Scan                                │   │
│  │  • Code Quality                                 │   │
│  │  • Pipeline Validation                          │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────┐
│ Code Review      │
│ by team members  │
└────────┬─────────┘
         │
    ┌────┴────┐
    │Approved?│
    └────┬────┘
         │
    ┌────┴───────┐
    │ No         │ Yes
    ▼            ▼
┌─────────┐  ┌──────────────────┐
│ Address │  │ Merge to develop │
│ feedback│  │ (--no-ff)        │
└────┬────┘  └────────┬─────────┘
     │                │
     └────────────────┘
              │
              ▼
       ┌──────────────────┐
       │ Delete feature   │
       │ branch           │
       └────────┬─────────┘
                │
                ▼
         ┌──────────────────┐
         │ Continue work... │
         └──────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│                     RELEASE TO MAIN                                 │
└─────────────────────────────────────────────────────────────────────┘

develop                                    main
  │                                          │
  │         [PR: develop → main]             │
  │─────────────────────────────────────────>│
  │                                          │
  │         [CI: Final Validation]           │
  │         • All tests pass                 │
  │         • Documentation check            │
  │         • Version verification           │
  │                                          │
  │         [Create Release]                 │
  │         • Tag version (v1.2.3)           │
  │         • Generate changelog             │
  │         • Publish artifacts              │
  │                                          │
  │         [Deploy]                         │
  │         • Production deployment          │
  │         • Update documentation           │
  │         • Notify stakeholders            │
  ▼                                          ▼
```

---

## 📊 CI/CD Pipeline Flow

```
┌────────────────────────────────────────────────────────────────────┐
│                    CI/CD PIPELINE OVERVIEW                         │
└────────────────────────────────────────────────────────────────────┘

Trigger: Push to feature branch or PR creation
│
▼
┌────────────────────────────────────────────────────────────────┐
│ STAGE 1: LINT & FORMAT                                         │
├────────────────────────────────────────────────────────────────┤
│ • Ruff linter checks                                           │
│ • Code formatting validation                                   │
│ • Commit message convention check                              │
└────────────────────────────────────────────────────────────────┘
│
▼
┌────────────────────────────────────────────────────────────────┐
│ STAGE 2: TEST SUITE                                            │
├────────────────────────────────────────────────────────────────┤
│ • Unit tests (pytest)                                          │
│ • Coverage reporting                                           │
│ • Integration tests                                            │
└────────────────────────────────────────────────────────────────┘
│
▼
┌────────────────────────────────────────────────────────────────┐
│ STAGE 3: SECURITY SCAN                                         │
├────────────────────────────────────────────────────────────────┤
│ • Secret detection                                             │
│ • Credential scanning                                          │
│ • Dependency vulnerability check                               │
│ • .gitignore compliance                                        │
└────────────────────────────────────────────────────────────────┘
│
▼
┌────────────────────────────────────────────────────────────────┐
│ STAGE 4: CODE QUALITY                                          │
├────────────────────────────────────────────────────────────────┤
│ • Dead code detection (Vulture)                                │
│ • Code statistics                                              │
│ • Complexity analysis                                          │
└────────────────────────────────────────────────────────────────┘
│
▼
┌────────────────────────────────────────────────────────────────┐
│ STAGE 5: PIPELINE VALIDATION                                   │
├────────────────────────────────────────────────────────────────┤
│ • Configuration loading test                                   │
│ • Module import validation                                     │
│ • Project structure verification                               │
└────────────────────────────────────────────────────────────────┘
│
▼
✓ All checks passed - Ready for merge


Trigger: Push to develop
│
▼
┌────────────────────────────────────────────────────────────────┐
│ DEVELOP INTEGRATION PIPELINE                                   │
├────────────────────────────────────────────────────────────────┤
│ • Integration tests                                            │
│ • Module compatibility checks                                  │
│ • Staging deployment readiness                                 │
│ • Notification to team                                         │
└────────────────────────────────────────────────────────────────┘
│
▼
✓ Staging environment updated


Trigger: Push to main
│
▼
┌────────────────────────────────────────────────────────────────┐
│ RELEASE PIPELINE                                               │
├────────────────────────────────────────────────────────────────┤
│ 1. Final Validation                                            │
│    • Complete test suite                                       │
│    • Documentation check                                       │
│    • Version tag verification                                  │
│                                                                │
│ 2. Create GitHub Release                                       │
│    • Generate changelog                                        │
│    • Create version tag (v1.2.3)                               │
│    • Publish release notes                                     │
│                                                                │
│ 3. Deploy Artifacts                                            │
│    • Package source code                                       │
│    • Upload release assets                                     │
│    • Update documentation site                                 │
│                                                                │
│ 4. Notification                                                │
│    • Release announcement                                      │
│    • Stakeholder notification                                  │
└────────────────────────────────────────────────────────────────┘
│
▼
✓ Production release published
```

---

## 🎯 Decision Points

```
                    Should this go to main directly?
                              │
                    ┌─────────┴─────────┐
                    │                   │
                   YES                 NO
                    │                   │
                    ▼                   ▼
          ┌─────────────────┐   ┌──────────────────┐
          │ Is it critical? │   │ Create feature   │
          │ (security bug,  │   │ branch from      │
          │  production     │   │ develop          │
          │  down)          │   │                  │
          └────────┬────────┘   └────────┬─────────┘
                   │                     │
          ┌────────┴────────┐            │
          │                 │            │
         YES               NO            │
          │                 │            │
          ▼                 ▼            │
┌─────────────────┐ ┌──────────────┐    │
│ Create hotfix/  │ │ Follow       │    │
│ branch from     │ │ feature      │    │
│ main            │ │ branch       │    │
│                 │ │ workflow     │    │
│ Merge to both   │ │              │    │
│ main & develop  │ │              │    │
└─────────────────┘ └──────────────┘    │
                                        │
                                        ▼
                              ┌──────────────────┐
                              │ After completion │
                              │                  │
                              │ Merge to develop │
                              │                  │
                              │ Test thoroughly  │
                              │                  │
                              │ Create PR to     │
                              │ main when ready  │
                              │ for release      │
                              └──────────────────┘
```

---

## 📋 Commit Message Examples

```
✅ GOOD COMMIT MESSAGES:

feat(models): add hyperparameter optimization for LightGBM using Optuna

  Implemented Optuna-based tuning with 100 trials.
  Optimized max_depth, learning_rate, n_estimators.
  Reduced overfitting by 15% on validation set.

  Closes #42

---

fix(data): correct timezone handling in tick_to_ohlcv conversion

  Fixed issue where timestamps were not properly converted to UTC.
  Added explicit timezone normalization and test cases.

  Fixes #38

---

refactor(features): extract technical indicator calculation to separate module

  Moved indicator calculations to new indicators.py file.
  Improved code organization without functional changes.

  See: Architecture decision record #3


❌ BAD COMMIT MESSAGES (DON'T DO THIS):

fix: fixed stuff
WIP
asdfasdf
fixed bug in thing
updated code
minor changes
```

---

## 🔐 Branch Protection Setup

```
GitHub Repository Settings → Branches → Add branch protection rule

┌─────────────────────────────────────────────────────────────┐
│ PROTECT: main                                                │
├─────────────────────────────────────────────────────────────┤
│ ☑ Require a pull request before merging                     │
│   ☑ Require approvals: 1                                    │
│   ☑ Dismiss stale approvals when new commits pushed         │
│                                                             │
│ ☑ Require status checks to pass before merging              │
│   ☑ Required status checks:                                 │
│      • Lint & Format Check                                  │
│      • Test Suite                                           │
│      • Security Scan                                        │
│      • Final Validation                                     │
│                                                             │
│ ☑ Require branches to be up to date before merging          │
│ ☑ Include administrators                                    │
│                                                             │
│ ☑ Restrict who can push to matching branches                │
│   (Configure allowed users/teams)                           │
│                                                             │
│ ☒ Allow force pushes (LEAVE UNCHECKED)                      │
│ ☒ Allow deletions (LEAVE UNCHECKED)                         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ PROTECT: develop                                             │
├─────────────────────────────────────────────────────────────┤
│ ☑ Require status checks to pass before merging              │
│   ☑ Required status checks:                                 │
│      • Lint & Format Check                                  │
│      • Test Suite                                           │
│      • Integration Tests                                    │
│                                                             │
│ ☑ Require branches to be up to date before merging          │
│ ☑ Disallow force pushes                                     │
│                                                             │
│ ☒ Allow force pushes (LEAVE UNCHECKED)                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 File Organization

```
thesis/
│
├── .github/                          # GitHub configuration
│   ├── README_GIT_WORKFLOW.md        # ← START HERE (executive summary)
│   ├── GIT_WORKFLOW.md               # Complete workflow guide
│   ├── IMPLEMENTATION_PLAN.md        # Implementation tracking
│   ├── COMMIT_CONVENTION.md          # Commit message guide
│   ├── PULL_REQUEST_TEMPLATE.md      # PR template
│   │
│   └── workflows/                    # GitHub Actions
│       ├── ci.yml                    # Pre-merge CI checks
│       ├── develop.yml               # Develop integration
│       └── release.yml               # Release pipeline
│
├── src/thesis/                       # Source code
│   ├── data/                         # Data pipeline
│   ├── features/                     # Feature engineering
│   ├── labels/                       # Triple-barrier labeling
│   ├── models/                       # ML models
│   ├── backtest/                     # Backtesting engine
│   ├── reporting/                    # Report generation
│   ├── config/                       # Configuration loader
│   └── pipeline/                     # Workflow orchestrator
│
├── tests/                            # Test suite (create this)
│   ├── unit/                         # Unit tests
│   ├── integration/                  # Integration tests
│   └── conftest.py                   # Pytest configuration
│
├── data/                             # Data directories
│   ├── raw/                          # Raw tick data (.gitignore)
│   ├── processed/                    # Processed OHLCV (.gitignore)
│   └── predictions/                  # Model outputs (.gitignore)
│
├── models/                           # Trained models
│   └── .gitkeep                      # Keep directory in git
│
├── results/                          # Results & visualizations
│   ├── backtest_results.json
│   ├── thesis_report.json
│   ├── thesis_report.md
│   └── trades_detail.csv
│
├── docs/                             # Documentation
│   ├── Architecture.md
│   ├── Features.md
│   ├── Evaluation.md
│   ├── Quickstart.md
│   └── README.md
│
├── logs/                             # Log files (.gitignore)
│
├── main.py                           # Entry point
├── main.ipynb                        # Jupyter notebook
├── pixi.toml                         # Environment & tasks
├── config.toml                       # Pipeline configuration
├── pixi.lock                         # Dependency lock file
├── .envrc                            # Direnv configuration
├── .gitignore                        # Git ignore rules
└── README.md                         # Project README
```

---

## ✅ Quick Start Checklist

```
□ 1. Read .github/README_GIT_WORKFLOW.md (you are here!)

□ 2. Review all documentation files:
  □ .github/GIT_WORKFLOW.md
  □ .github/COMMIT_CONVENTION.md
  □ .github/IMPLEMENTATION_PLAN.md

□ 3. Initial git setup:
  □ git checkout main
  □ git add .github/
  □ git commit -m "ci: establish Git workflow"
  □ git push -u origin main

□ 4. Configure GitHub settings (manual):
  □ Set up branch protection for main
  □ Set up branch protection for develop
  □ Verify workflows are enabled

□ 5. Test the workflow:
  □ Create test feature branch
  □ Make a small change
  □ Create PR
  □ Verify CI checks run
  □ Merge to develop
  □ Verify develop workflow runs

□ 6. Team onboarding:
  □ Share workflow documentation
  □ Review commit conventions
  □ Practice creating PRs
  □ Establish review rotation
```

---

## 🎓 Learning Resources

### Required Reading
- [Conventional Commits](https://www.conventionalcommits.org/) - 10 min
- [Git Flow](https://nvie.com/posts/a-successful-git-branching-model/) - 15 min
- [GitHub Actions Docs](https://docs.github.com/en/actions) - Browse

### Optional Deep Dives
- [Keep a Changelog](https://keepachangelog.com/)
- [Semantic Versioning](https://semver.org/)
- [GitHub Flow](https://guides.github.com/introduction/flow/)

---

**Quick Reference**: When in doubt, check `.github/README_GIT_WORKFLOW.md`

**Questions?**: Review the implementation plan or consult with your advisor

**Status**: 🚀 Ready to implement!
