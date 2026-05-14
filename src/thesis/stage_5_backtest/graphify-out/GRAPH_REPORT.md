# Graph Report - src/thesis/stage_5_backtest  (2026-05-15)

## Corpus Check
- 5 files · ~2,373 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 47 nodes · 76 edges · 7 communities
- Extraction: 75% EXTRACTED · 25% INFERRED · 0% AMBIGUOUS · INFERRED: 19 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `37d94694`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]

## God Nodes (most connected - your core abstractions)
1. `MLSignalStrategy` - 9 edges
2. `run_backtest_manual()` - 8 edges
3. `compute_backtest()` - 7 edges
4. `_persist_backtest_results()` - 7 edges
5. `_run_fractional_backtest()` - 6 edges
6. `run_backtest_from_data()` - 6 edges
7. `_prepare_df()` - 5 edges
8. `_normalize_stats()` - 4 edges
9. `_trades_to_list()` - 4 edges
10. `_load_backtest_data()` - 4 edges

## Surprising Connections (you probably didn't know these)
- `_persist_backtest_results()` --calls--> `_log_core_backtest_metrics()`  [INFERRED]
  simulation.py → persistence.py
- `compute_backtest()` --calls--> `_normalize_stats()`  [INFERRED]
  simulation.py → persistence.py
- `run_backtest_manual()` --calls--> `_normalize_stats()`  [INFERRED]
  simulation.py → persistence.py
- `compute_backtest()` --calls--> `_trades_to_list()`  [INFERRED]
  simulation.py → persistence.py
- `run_backtest_manual()` --calls--> `_trades_to_list()`  [INFERRED]
  simulation.py → persistence.py

## Communities (7 total, 0 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.25
Nodes (5): MLSignalStrategy, Evaluate latest model signal, place orders if appropriate., Trade on ML signals with ATR stops and simple risk gates.      Signal shift: pre, Register indicators and initialise risk-management state., Strategy

### Community 1 - "Community 1"
Cohesion: 0.36
Nodes (9): _compute_spread_rate(), _create_fractional_backtest(), _make_commission_fn(), _prepare_df(), Stage 5 backtest helpers (prep + runner)., _run_fractional_backtest(), _validate_backtest_merge(), Run backtest with manual params (no Config required).      Designed for dashboar (+1 more)

### Community 2 - "Community 2"
Cohesion: 0.39
Nodes (7): _log_core_backtest_metrics(), Stage 5 backtest outputs (metrics, trades, charts)., _save_bokeh_chart(), _save_equity_curve_csv(), _save_json_results(), _save_trade_details_csv(), _persist_backtest_results()

### Community 3 - "Community 3"
Cohesion: 0.33
Nodes (4): CFD backtest simulation package., _calendar_day(), ML signal trading strategy used in stage 5 backtests., Return market date anchored to 5PM New York (FX/CFD market close).      UTC conv

### Community 4 - "Community 4"
Cohesion: 0.5
Nodes (4): _normalize_stats(), _trades_to_list(), Run backtest using in-memory Polars DataFrames.      Args:         test_df: Mark, run_backtest_from_data()

### Community 5 - "Community 5"
Cohesion: 0.67
Nodes (3): _apply_oos_date_filter(), _load_backtest_data(), CFD backtest simulation via backtesting.py.  SL/TP ATR multipliers must align wi

### Community 6 - "Community 6"
Cohesion: 0.5
Nodes (4): compute_backtest(), Load data, run FractionalBacktest, return normalized results.      No files writ, Run full CFD backtest from files in config.      Walk-forward: joins OOF predict, run_backtest()

## Knowledge Gaps
- **13 isolated node(s):** `CFD backtest simulation package.`, `Stage 5 backtest outputs (metrics, trades, charts).`, `Stage 5 backtest helpers (prep + runner).`, `CFD backtest simulation via backtesting.py.  SL/TP ATR multipliers must align wi`, `Load data, run FractionalBacktest, return normalized results.      No files writ` (+8 more)
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `MLSignalStrategy` connect `Community 0` to `Community 3`?**
  _High betweenness centrality (0.355) - this node is a cross-community bridge._
- **Why does `run_backtest_manual()` connect `Community 1` to `Community 4`, `Community 5`?**
  _High betweenness centrality (0.208) - this node is a cross-community bridge._
- **Why does `_persist_backtest_results()` connect `Community 2` to `Community 5`, `Community 6`?**
  _High betweenness centrality (0.162) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `run_backtest_manual()` (e.g. with `_prepare_df()` and `_compute_spread_rate()`) actually correct?**
  _`run_backtest_manual()` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `compute_backtest()` (e.g. with `_run_fractional_backtest()` and `_normalize_stats()`) actually correct?**
  _`compute_backtest()` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `_persist_backtest_results()` (e.g. with `_save_json_results()` and `_save_trade_details_csv()`) actually correct?**
  _`_persist_backtest_results()` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `_run_fractional_backtest()` (e.g. with `compute_backtest()` and `run_backtest_from_data()`) actually correct?**
  _`_run_fractional_backtest()` has 2 INFERRED edges - model-reasoned connections that need verification._