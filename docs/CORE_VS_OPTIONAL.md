# Core vs optional modules

This split matches the thesis batch pipeline (`pixi run workflow`) vs research and UI extras.

## Core (thesis-critical path)

These modules implement stages **0–6** and are required for reproducible experiments and the written thesis:

| Area | Path | Role |
|------|------|------|
| Config | [`config.toml`](../config.toml), [`src/thesis/config.py`](../src/thesis/config.py) | Single TOML + dataclass loader |
| Session paths | [`src/thesis/session_paths.py`](../src/thesis/session_paths.py) | Session-scoped artifact paths (CLI + dashboard) |
| Orchestration | [`src/thesis/pipeline.py`](../src/thesis/pipeline.py), [`main.py`](../main.py) | Stage runner + CLI |
| Stage 0 | [`src/thesis/agg/`](../src/thesis/agg/) | Tick → OHLCV |
| Stage 1 | [`src/thesis/features/`](../src/thesis/features/) | Technical indicators |
| Stage 2 | [`src/thesis/labeling/`](../src/thesis/labeling/) | Triple-barrier labels |
| Stage 3 | [`src/thesis/splitting/`](../src/thesis/splitting/) | Train/val/test, purge/embargo, correlation filter |
| Stage 4 | [`src/thesis/gru/`](../src/thesis/gru/), [`src/thesis/hybrid/`](../src/thesis/hybrid/) (except interpret) | GRU extractor + LightGBM hybrid |
| Stage 5 | [`src/thesis/backtest/`](../src/thesis/backtest/) | CFD simulation |
| Stage 6 | [`src/thesis/report/`](../src/thesis/report/), [`src/thesis/plots/`](../src/thesis/plots/) | Markdown report + static matplotlib charts |
| Shared constants | [`src/thesis/constants.py`](../src/thesis/constants.py) | `EXCLUDE_COLS`, chart palette |
| Console UI | [`src/thesis/ui.py`](../src/thesis/ui.py) | Rich logging helpers |

## Optional (not required for the core pipeline)

These add comparison studies, interactivity, or heavier interpretation—they are omitted from the default coverage gate where noted:

| Area | Path | Role |
|------|------|------|
| Ablation | [`src/thesis/ablation.py`](../src/thesis/ablation.py) | LGBM-only / GRU-only / hybrid comparison (`pixi run ablation`) |
| SHAP / FI extras | [`src/thesis/hybrid/interpret.py`](../src/thesis/hybrid/interpret.py) | SHAP + JSON feature importance (invoked from hybrid training) |
| Interactive charts | [`src/thesis/charts/`](../src/thesis/charts/) | Pyecharts builders for Streamlit |
| Dashboard | [`src/thesis/dashboard/`](../src/thesis/dashboard/) | `pixi run streamlit` explorer |
| Data download | [`scripts/data_download.py`](../scripts/data_download.py) | Dukascopy tick download (`pixi run data`) |

**Note:** `hybrid/interpret.py` runs during stage 4 when training completes; it is “optional” in the sense that the architectural story of the thesis is the hybrid stack, while SHAP depth is supporting analysis.

## Artifact ownership

- **Global (default):** processed parquets under `paths.data_processed` / `paths.*_data` as set in `config.toml`.
- **Per session:** models, predictions, backtest JSON, report markdown, and session logs under `results/{SYMBOL}_{TF}_{timestamp}/`, applied via [`session_paths.configure_session_paths`](../src/thesis/session_paths.py).

See also [ARCHITECTURE.md](ARCHITECTURE.md) and [CONFIGURATION.md](CONFIGURATION.md).
