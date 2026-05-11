# Graph Report - src  (2026-05-11)

## Corpus Check
- Corpus is ~37,639 words - fits in a single context window. You may not need a graph.

## Summary
- 737 nodes · 1021 edges · 44 communities (30 shown, 14 thin omitted)
- Extraction: 82% EXTRACTED · 18% INFERRED · 0% AMBIGUOUS · INFERRED: 185 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_LGBM Training Pipeline|LGBM Training Pipeline]]
- [[_COMMUNITY_Feature Engineering|Feature Engineering]]
- [[_COMMUNITY_Data Processing & Quality|Data Processing & Quality]]
- [[_COMMUNITY_Model Evaluation Metrics|Model Evaluation Metrics]]
- [[_COMMUNITY_Walk-Forward Validation|Walk-Forward Validation]]
- [[_COMMUNITY_Backtest Engine|Backtest Engine]]
- [[_COMMUNITY_Configuration Management|Configuration Management]]
- [[_COMMUNITY_Report Generation|Report Generation]]
- [[_COMMUNITY_Label Generation|Label Generation]]
- [[_COMMUNITY_Pipeline Orchestration|Pipeline Orchestration]]
- [[_COMMUNITY_Feature Registry|Feature Registry]]
- [[_COMMUNITY_Trading Strategy|Trading Strategy]]
- [[_COMMUNITY_Report Data Sections|Report Data Sections]]
- [[_COMMUNITY_Chart Data Layer|Chart Data Layer]]
- [[_COMMUNITY_Session Management|Session Management]]
- [[_COMMUNITY_Backtest Visualization|Backtest Visualization]]
- [[_COMMUNITY_Calibration Metrics|Calibration Metrics]]
- [[_COMMUNITY_Benchmark Comparison|Benchmark Comparison]]
- [[_COMMUNITY_Table Rendering|Table Rendering]]
- [[_COMMUNITY_Model Visualization|Model Visualization]]
- [[_COMMUNITY_Dashboard Reports|Dashboard Reports]]
- [[_COMMUNITY_Assessment Sections|Assessment Sections]]
- [[_COMMUNITY_Model Quality Assessment|Model Quality Assessment]]
- [[_COMMUNITY_Zone-Based Metrics|Zone-Based Metrics]]
- [[_COMMUNITY_Markdown Formatting|Markdown Formatting]]
- [[_COMMUNITY_Issue Reporting|Issue Reporting]]
- [[_COMMUNITY_Dashboard UI|Dashboard UI]]
- [[_COMMUNITY_Metric Cards|Metric Cards]]
- [[_COMMUNITY_OOF Analysis|OOF Analysis]]
- [[_COMMUNITY_Backtest Issues|Backtest Issues]]
- [[_COMMUNITY_Stage 1 Init|Stage 1 Init]]
- [[_COMMUNITY_Stage 3 Init|Stage 3 Init]]
- [[_COMMUNITY_Stage 4 Init|Stage 4 Init]]
- [[_COMMUNITY_Walk-Forward Init|Walk-Forward Init]]
- [[_COMMUNITY_LGBM Init|LGBM Init]]
- [[_COMMUNITY_Reporting Sections Init|Reporting Sections Init]]
- [[_COMMUNITY_Charts Init|Charts Init]]
- [[_COMMUNITY_Charts Shared|Charts Shared]]
- [[_COMMUNITY_Dashboard Init|Dashboard Init]]
- [[_COMMUNITY_Shared Init|Shared Init]]
- [[_COMMUNITY_UI Rationale|UI Rationale]]
- [[_COMMUNITY_Schemas Rationale|Schemas Rationale]]
- [[_COMMUNITY_Schemas Rationale 2|Schemas Rationale 2]]
- [[_COMMUNITY_Schemas Rationale 3|Schemas Rationale 3]]

## God Nodes (most connected - your core abstractions)
1. `generate_features()` - 21 edges
2. `_build_markdown()` - 20 edges
3. `compute_all_classification_metrics()` - 17 edges
4. `_train_and_predict_stacking_window()` - 16 edges
5. `_tbl_row()` - 16 edges
6. `_train_and_predict_static_window()` - 13 edges
7. `generate_labels()` - 12 edges
8. `train_stacking_walk_forward()` - 12 edges
9. `run_backtest()` - 12 edges
10. `render_backtest_section()` - 11 edges

## Surprising Connections (you probably didn't know these)
- `_run_backtest_with_barrier_guard()` --calls--> `run_backtest()`  [INFERRED]
  pipeline.py → stage_5_backtest/simulation.py
- `run_pipeline()` --calls--> `train_walk_forward()`  [INFERRED]
  pipeline.py → stage_4_training/walk_forward/dispatcher.py
- `generate_features()` --calls--> `build_feature_output_cols()`  [INFERRED]
  stage_2_features/engineering.py → shared/feature_registry.py
- `generate_features()` --calls--> `get_static_feature_cols()`  [INFERRED]
  stage_2_features/engineering.py → shared/feature_registry.py
- `_zone()` --calls--> `_get_metric_zone()`  [INFERRED]
  stage_6_reporting/tables.py → shared/zones.py

## Communities (44 total, 14 thin omitted)

### Community 0 - "LGBM Training Pipeline"
Cohesion: 0.05
Nodes (65): _build_interaction_constraints(), _compute_class_weights(), _compute_distribution_shift_weights(), _filter_validation_to_seen_classes(), LightGBM utilities for tabular walk-forward training., Drop validation rows whose class is absent from the training fold.      LightGBM, Train LightGBM with fixed hyperparameters.      Args:         X_train: Training, Wrap a NumPy matrix as a pandas DataFrame.      Args:         X: Feature matrix (+57 more)

### Community 1 - "Feature Engineering"
Cohesion: 0.05
Nodes (53): _drop_warmup_rows(), generate_features(), Feature engineering — production pipeline for price-action features.  The produc, Raise on empty, unsorted, or duplicate timestamps; log gap stats., Drop rows with null/non-finite model-facing features., Pandera + timestamp/uniqueness/null checks on the feature dataset., Write a JSON sidecar listing feature column names., Generate and persist feature-enriched OHLCV bars. (+45 more)

### Community 2 - "Data Processing & Quality"
Cohesion: 0.05
Nodes (51): Project-wide constants shared across pipeline stages.  Single source of truth fo, Parse a config timeframe string into milliseconds.      Supports ``H`` (hours),, timeframe_to_ms(), check_candle_quality(), check_gap_report(), check_ohlcv_consistency(), check_outlier_returns(), classify_calendar_gaps() (+43 more)

### Community 3 - "Model Evaluation Metrics"
Cohesion: 0.05
Nodes (54): always_predict_class(), compute_baseline_metrics(), majority_class_baseline(), naive_direction(), random_baseline(), Baseline prediction strategies for walk-forward comparison.  All baselines opera, Predict the direction of the previous bar's return (persistence).      Maps: ret, Return predictions that always equal *class_label*. (+46 more)

### Community 4 - "Walk-Forward Validation"
Cohesion: 0.05
Nodes (46): Save sorted model feature importances to JSON.      Args:         model: Fitted, _save_feature_importance(), apply_event_time_purge(), apply_purge_embargo(), generate_windows(), log_windows(), Bar-based walk-forward sliding window validation with purge and embargo.  Genera, Adjust a raw window to account for purge and embargo gaps.      * **Purge** remo (+38 more)

### Community 5 - "Backtest Engine"
Cohesion: 0.08
Nodes (35): _log_core_backtest_metrics(), _normalize_stats(), Stage 5 backtest outputs (metrics, trades, charts)., Convert a backtesting.py trades DataFrame to a JSON-serializable list.      Each, Save backtest results (metrics + trades) as JSON., Save per-trade records as CSV., Save equity curve as CSV with running peak and drawdown.      Equity curve is tr, Save Bokeh HTML chart for the backtest. (+27 more)

### Community 6 - "Configuration Management"
Cohesion: 0.07
Nodes (37): _apply_section(), BacktestConfig, Config, DataConfig, FeaturesConfig, get_config(), LabelsConfig, LGBMConfig (+29 more)

### Community 7 - "Report Generation"
Cohesion: 0.08
Nodes (28): _equity_series_from_closed_trades(), _load_feature_importance(), _plot_equity_curve(), _plot_feature_importance(), Chart rendering helpers for the thesis report.  Contains equity-curve and featur, Timestamps and equity from closed-trade PnL (not mark-to-market)., Render and save an equity curve image from trade history.      Args:         tra, Load feature-importance JSON from session report outputs.      Args:         con (+20 more)

### Community 8 - "Label Generation"
Cohesion: 0.1
Nodes (27): compute_average_uniqueness(), compute_event_end(), _compute_labels(), _filter_censored(), generate_labels(), _load_inputs(), _log_atr_stats(), _log_distribution() (+19 more)

### Community 9 - "Pipeline Orchestration"
Cohesion: 0.11
Nodes (20): Shared lightweight CLI UI helpers without Rich dependency., Minimal console facade compatible with previous Rich call sites., Log plain text messages., Log a visual separator line., Print a stage banner with concise log output.      Args:         stage: Stage nu, Print a skip line and logger message.      Args:         stage: Stage number (1-, SimpleConsole, stage_header() (+12 more)

### Community 10 - "Feature Registry"
Cohesion: 0.11
Nodes (21): build_exclude_cols(), build_feature_output_cols(), build_label_output_cols(), get_label_helper_cols(), get_static_feature_cols(), Single source of truth for feature column lists across pipeline stages., Return the static (non-sequential) feature columns from config., Return helper columns used during label construction (e.g. ATR). (+13 more)

### Community 11 - "Trading Strategy"
Cohesion: 0.12
Nodes (13): CFD backtest simulation package., _calendar_day(), MLSignalStrategy, ML signal trading strategy used in stage 5 backtests., Floor ATR to ``max(atr, self.min_atr)``., Update peak equity, drawdown tracking, and daily loss tracking.          The dra, Check all risk gates before opening a new position., Return fixed position size after confidence filtering.          Scaling lots by (+5 more)

### Community 12 - "Report Data Sections"
Cohesion: 0.15
Nodes (19): _load_label_distribution(), Data-quality and methodology section renderers.  Renderers for data quality, lab, Render the Label Design & Methodology explanation section., Render the Validation Methodology section (walk-forward, purge/embargo)., Render auxiliary regression metrics section (MAE/RMSE/R²) if available., Compute class distribution from the labels parquet file., Render the Data Quality analysis section from the JSON sidecar., _render_auxiliary_regression_section() (+11 more)

### Community 13 - "Chart Data Layer"
Cohesion: 0.13
Nodes (18): build_candlestick_chart(), build_correlation_heatmap(), build_feature_distribution_chart(), build_feature_distributions_chart(), build_label_distribution_chart(), _downsample_ohlcv(), _get_feature_cols(), Data exploration chart builders. (+10 more)

### Community 14 - "Session Management"
Cohesion: 0.12
Nodes (17): load_session_data(), Session artifact loading for dashboard charts., Load session artifacts required by interactive chart builders.      Args:, find_sessions(), load_config(), parse_session_meta(), Session discovery, parsing, and loading helpers., Discover available session directories under ``results/``. (+9 more)

### Community 15 - "Backtest Visualization"
Cohesion: 0.14
Nodes (16): build_duration_pnl_scatter(), build_equity_drawdown_chart(), build_monthly_returns_heatmap(), build_pnl_histogram_chart(), build_rolling_sharpe_chart(), _compute_monthly_returns(), Backtest chart builders., Build a histogram of winning and losing trade PnL values.      Args:         tra (+8 more)

### Community 16 - "Calibration Metrics"
Cohesion: 0.17
Nodes (15): brier_score(), calibration_reliability_data(), compute_all_calibration_metrics(), confidence_bins_accuracy(), expected_calibration_error(), log_loss(), Probability calibration metrics: ECE, Brier, log-loss, confidence bins., Return bin centers, accuracies, and counts for calibration curve plotting. (+7 more)

### Community 17 - "Benchmark Comparison"
Cohesion: 0.2
Nodes (15): _annualized_sharpe(), compute_benchmark_comparison(), _compute_random_strategy(), _equity_curve_from_bar_returns(), _load_close_prices_for_benchmark(), _max_drawdown_pct(), _model_label(), Benchmark comparison helpers — naive strategies vs model.  Provides equity-curve (+7 more)

### Community 18 - "Table Rendering"
Cohesion: 0.2
Nodes (13): _accuracy_table(), _backtest_metrics_table(), _calibration_summary_text(), _compute_ece_numpy(), _exec_table(), Markdown table builders and verdict logic for the thesis report.  Each function, Compute a one-paragraph calibration reliability note.      Reads predicted proba, Key ML-first metrics with application-demo metrics second.      Args:         L: (+5 more)

### Community 19 - "Model Visualization"
Cohesion: 0.18
Nodes (12): build_confidence_distribution_chart(), build_confusion_matrix_chart(), build_feature_importance_chart(), build_prediction_distribution_chart(), Model performance chart builders., Build a horizontal top-N feature-importance chart.      Args:         fi: Mappin, Build a normalized confusion-matrix heatmap for 3-class labels.      Args:, Build actual-vs-predicted label distribution bars. (+4 more)

### Community 20 - "Dashboard Reports"
Cohesion: 0.16
Nodes (12): Reports section renderer., Render the reports page with thesis markdown and artifact visuals., render_reports_section(), date_only(), Shared chart renderer and config/trade summary helpers., Render a pyecharts chart into the Streamlit app., Return the date part of a config timestamp string., Hide report sections duplicated by dashboard-native charts. (+4 more)

### Community 21 - "Assessment Sections"
Cohesion: 0.22
Nodes (9): _get_zone_info(), Return (emoji, zone_label, recommended_range) for a backtest metric., _compute_avg_win_loss_ratio(), Backtest metric zones, baseline comparison, and verdict section renderers.  Cont, Render baseline strategy comparison using the _baselines module., Compute average win / average loss ratio from trade records., Render backtest metric quality zones with emoji indicators., _render_baseline_comparison_section() (+1 more)

### Community 22 - "Model Quality Assessment"
Cohesion: 0.24
Nodes (9): _assess_model_quality(), _assess_trading_edge(), _derive_recommendation(), Assessment helpers and constants for backtest verdict logic.  Contains quality t, Classify ML quality into POOR / FAIR / GOOD with a short reason., Classify trading edge into NEGATIVE / MARGINAL / POSITIVE., Produce a deployment recommendation from model quality + trading edge., Append synthesized verdict (model quality + trading edge + rec). (+1 more)

### Community 23 - "Zone-Based Metrics"
Cohesion: 0.24
Nodes (9): Render a metric card with colour-coded zone indicator., render_zoned_metric(), _get_metric_zone(), _is_extreme_value(), Metric zone definitions for CFD backtest benchmarks.  Provides zone classificati, Check if a metric value is extreme and return threshold info.      Args:, Return (color_name, zone_label, recommendation) for a given metric.      Zone la, Zone emoji for a metric value. (+1 more)

### Community 24 - "Markdown Formatting"
Cohesion: 0.25
Nodes (8): _fmt_dollar(), _fmt_f2(), _fmt_pct(), Shared Markdown formatting helpers for stage 6 reporting.  This module centraliz, _backtest_params_table(), _benchmark_comparison_table(), Backtest simulation parameters., Compare benchmarks against the configured model architecture.

### Community 25 - "Issue Reporting"
Cohesion: 0.25
Nodes (8): _identify_primary_issue(), Return the single most critical issue description, or None., Append one-paragraph ML quality assessment to markdown lines., Append primary issue identification and application demo summary., _render_ml_quality_paragraph(), _render_primary_issue(), _exec_verdict(), One-paragraph ML-first overall assessment with synthesized verdict.      Delegat

### Community 26 - "Dashboard UI"
Cohesion: 0.25
Nodes (6): main(), Dashboard entry point — sidebar, navigation, and section dispatch., Render the Streamlit dashboard with session selection and navigation., Training section renderer., Render tabular training history and pipeline logs., render_training_section()

### Community 27 - "Metric Cards"
Cohesion: 0.33
Nodes (5): Metric card renderers and CSS constants for the dashboard., Render a styled metric card with gradient background and accent border., render_metric_card(), Render compact direction counts and PnL without low-value charts., render_trade_direction_summary()

### Community 28 - "OOF Analysis"
Cohesion: 0.5
Nodes (3): OOF vs OOS generalization check section renderer.  Renders the out-of-fold vs ou, Render OOF vs OOS comparison section with side-by-side metrics table., _render_oof_vs_oos_section()

### Community 29 - "Backtest Issues"
Cohesion: 0.5
Nodes (4): Render sorted issues and recommendations into markdown lines., _render_issues(), _issues_list(), High-signal issues and recommendations from report metrics.      Only the most c

## Knowledge Gaps
- **366 isolated node(s):** `Thesis ML pipeline — top-level public API surface.  This module re-exports the m`, `Pipeline orchestration for the sequential thesis workflow.  Runs data preparatio`, `Compute an 8-char SHA-256 fingerprint of config sections relevant to a stage.`, `Resolve the effective cache check path based on invalidation strategy.      Args`, `Execute a pipeline stage with cache checking.      Checks the workflow flag and` (+361 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **14 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `_get_metric_zone()` connect `Zone-Based Metrics` to `Table Rendering`?**
  _High betweenness centrality (0.052) - this node is a cross-community bridge._
- **Why does `render_zoned_metric()` connect `Zone-Based Metrics` to `Metric Cards`, `Backtest Visualization`?**
  _High betweenness centrality (0.051) - this node is a cross-community bridge._
- **Why does `render_backtest_section()` connect `Backtest Visualization` to `Dashboard UI`, `Metric Cards`, `Dashboard Reports`, `Zone-Based Metrics`?**
  _High betweenness centrality (0.050) - this node is a cross-community bridge._
- **Are the 15 inferred relationships involving `generate_features()` (e.g. with `_add_atr()` and `_add_context_features()`) actually correct?**
  _`generate_features()` has 15 INFERRED edges - model-reasoned connections that need verification._
- **Are the 17 inferred relationships involving `_build_markdown()` (e.g. with `_model_label()` and `_exec_table()`) actually correct?**
  _`_build_markdown()` has 17 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `_train_and_predict_stacking_window()` (e.g. with `_select_static_feature_cols()` and `fit_static_feature_pipeline()`) actually correct?**
  _`_train_and_predict_stacking_window()` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 14 inferred relationships involving `_tbl_row()` (e.g. with `_exec_table()` and `_config_table()`) actually correct?**
  _`_tbl_row()` has 14 INFERRED edges - model-reasoned connections that need verification._