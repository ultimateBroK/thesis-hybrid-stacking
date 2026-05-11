# Graph Report - src  (2026-05-11)

## Corpus Check
- Corpus is ~47,711 words - fits in a single context window. You may not need a graph.

## Summary
- 846 nodes · 1188 edges · 54 communities (39 shown, 15 thin omitted)
- Extraction: 81% EXTRACTED · 19% INFERRED · 0% AMBIGUOUS · INFERRED: 229 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Feature Engineering|Feature Engineering]]
- [[_COMMUNITY_Baseline Models|Baseline Models]]
- [[_COMMUNITY_Data Quality Constants|Data Quality Constants]]
- [[_COMMUNITY_GRU Data Calibration|GRU Data Calibration]]
- [[_COMMUNITY_Configuration Schema|Configuration Schema]]
- [[_COMMUNITY_Backtest Persistence|Backtest Persistence]]
- [[_COMMUNITY_GRU Inference Validation|GRU Inference Validation]]
- [[_COMMUNITY_Benchmark Comparison|Benchmark Comparison]]
- [[_COMMUNITY_Metric Zones Dashboard|Metric Zones Dashboard]]
- [[_COMMUNITY_Pipeline CLI UI|Pipeline CLI UI]]
- [[_COMMUNITY_Triple Barrier Labels|Triple Barrier Labels]]
- [[_COMMUNITY_Feature Registry|Feature Registry]]
- [[_COMMUNITY_Walk Forward Artifacts|Walk Forward Artifacts]]
- [[_COMMUNITY_Trading Strategy|Trading Strategy]]
- [[_COMMUNITY_Walk Forward Utils|Walk Forward Utils]]
- [[_COMMUNITY_GRU Architecture|GRU Architecture]]
- [[_COMMUNITY_Data Charts|Data Charts]]
- [[_COMMUNITY_Session Loading|Session Loading]]
- [[_COMMUNITY_Static Window Training|Static Window Training]]
- [[_COMMUNITY_Hybrid Training|Hybrid Training]]
- [[_COMMUNITY_Backtest Charts|Backtest Charts]]
- [[_COMMUNITY_Markdown Formatting|Markdown Formatting]]
- [[_COMMUNITY_Calibration Metrics|Calibration Metrics]]
- [[_COMMUNITY_Report Data Sections|Report Data Sections]]
- [[_COMMUNITY_Report Tables|Report Tables]]
- [[_COMMUNITY_Model Charts|Model Charts]]
- [[_COMMUNITY_Backtest Report Sections|Backtest Report Sections]]
- [[_COMMUNITY_Assessment Verdicts|Assessment Verdicts]]
- [[_COMMUNITY_OOF Predictions|OOF Predictions]]
- [[_COMMUNITY_ML Quality Verdict|ML Quality Verdict]]
- [[_COMMUNITY_LGBM Entry Points|LGBM Entry Points]]
- [[_COMMUNITY_GRU Walk Forward|GRU Walk Forward]]
- [[_COMMUNITY_Prediction Saving|Prediction Saving]]
- [[_COMMUNITY_Reports Dashboard|Reports Dashboard]]
- [[_COMMUNITY_Dashboard Metric Cards|Dashboard Metric Cards]]
- [[_COMMUNITY_Chart Shared Helpers|Chart Shared Helpers]]
- [[_COMMUNITY_Training Dashboard|Training Dashboard]]
- [[_COMMUNITY_OOF OOS Report|OOF OOS Report]]
- [[_COMMUNITY_Dashboard Main|Dashboard Main]]
- [[_COMMUNITY_Top Level API|Top Level API]]
- [[_COMMUNITY_Labels Package|Labels Package]]
- [[_COMMUNITY_Training Package|Training Package]]
- [[_COMMUNITY_Walk Forward Package|Walk Forward Package]]
- [[_COMMUNITY_LGBM Package|LGBM Package]]
- [[_COMMUNITY_GRU Package|GRU Package]]
- [[_COMMUNITY_Report Sections Package|Report Sections Package]]
- [[_COMMUNITY_Charts Package|Charts Package]]
- [[_COMMUNITY_Chart Typing|Chart Typing]]
- [[_COMMUNITY_Dashboard Package|Dashboard Package]]
- [[_COMMUNITY_Shared Package|Shared Package]]
- [[_COMMUNITY_Status Context|Status Context]]
- [[_COMMUNITY_OHLCV Validation|OHLCV Validation]]
- [[_COMMUNITY_Feature Validation|Feature Validation]]
- [[_COMMUNITY_Label Validation|Label Validation]]

## God Nodes (most connected - your core abstractions)
1. `generate_features()` - 22 edges
2. `_build_markdown()` - 22 edges
3. `compute_all_classification_metrics()` - 17 edges
4. `_tbl_row()` - 17 edges
5. `train_gru()` - 15 edges
6. `train_model()` - 14 edges
7. `_train_and_predict_static_window()` - 13 edges
8. `generate_labels()` - 12 edges
9. `_wf_build_predict_phase()` - 12 edges
10. `run_backtest()` - 12 edges

## Surprising Connections (you probably didn't know these)
- `_run_backtest_with_barrier_guard()` --calls--> `run_backtest()`  [INFERRED]
  pipeline.py → stage_5_backtest/simulation.py
- `generate_features()` --calls--> `build_feature_output_cols()`  [INFERRED]
  stage_2_features/engineering.py → shared/feature_registry.py
- `generate_features()` --calls--> `get_static_feature_cols()`  [INFERRED]
  stage_2_features/engineering.py → shared/feature_registry.py
- `generate_features()` --calls--> `get_gru_feature_cols()`  [INFERRED]
  stage_2_features/engineering.py → shared/feature_registry.py
- `_zone()` --calls--> `_get_metric_zone()`  [INFERRED]
  stage_6_reporting/tables.py → shared/zones.py

## Communities (54 total, 15 thin omitted)

### Community 0 - "Feature Engineering"
Cohesion: 0.05
Nodes (53): _drop_warmup_rows(), generate_features(), Feature engineering — production pipeline for price-action features.  The produc, Raise on empty, unsorted, or duplicate timestamps; log gap stats., Drop rows with null/non-finite model-facing features., Pandera + timestamp/uniqueness/null checks on the feature dataset., Write a JSON sidecar listing feature column names., Generate and persist feature-enriched OHLCV bars. (+45 more)

### Community 1 - "Baseline Models"
Cohesion: 0.05
Nodes (54): always_predict_class(), compute_baseline_metrics(), majority_class_baseline(), naive_direction(), random_baseline(), Baseline prediction strategies for walk-forward comparison.  All baselines opera, Predict the direction of the previous bar's return (persistence).      Maps: ret, Return predictions that always equal *class_label*. (+46 more)

### Community 2 - "Data Quality Constants"
Cohesion: 0.05
Nodes (51): Project-wide constants shared across pipeline stages.  Single source of truth fo, Parse a config timeframe string into milliseconds.      Supports ``H`` (hours),, timeframe_to_ms(), check_candle_quality(), check_gap_report(), check_ohlcv_consistency(), check_outlier_returns(), classify_calendar_gaps() (+43 more)

### Community 3 - "GRU Data Calibration"
Cohesion: 0.05
Nodes (44): Dataset, _calibrate_model(), _compute_ece(), Temperature-scaling calibration for GRU classifiers.  Collects logits on the val, Compute Expected Calibration Error (ECE).      Partitions predictions into ``n_b, Calibrate classifier probabilities via temperature scaling.      Collects logits, _ensure_log_returns(), _extract_labels() (+36 more)

### Community 4 - "Configuration Schema"
Cohesion: 0.06
Nodes (39): _apply_section(), BacktestConfig, Config, DataConfig, FeaturesConfig, get_config(), GRUConfig, LabelsConfig (+31 more)

### Community 5 - "Backtest Persistence"
Cohesion: 0.08
Nodes (35): _log_core_backtest_metrics(), _normalize_stats(), Stage 5 backtest outputs (metrics, trades, charts)., Convert a backtesting.py trades DataFrame to a JSON-serializable list.      Each, Save backtest results (metrics + trades) as JSON., Save per-trade records as CSV., Save equity curve as CSV with running peak and drawdown.      Equity curve is tr, Save Bokeh HTML chart for the backtest. (+27 more)

### Community 6 - "GRU Inference Validation"
Cohesion: 0.08
Nodes (31): extract_hidden_states(), predict_gru_proba(), GRU inference — hidden-state extraction and probability prediction., Extract final-layer hidden states for a batch of sequences.      When ``mean`` a, Predict class probabilities from a trained GRU backbone + classifier.      Args:, apply_event_time_purge(), apply_purge_embargo(), generate_windows() (+23 more)

### Community 7 - "Benchmark Comparison"
Cohesion: 0.09
Nodes (30): _annualized_sharpe(), compute_benchmark_comparison(), _compute_random_strategy(), _equity_curve_from_bar_returns(), _load_close_prices_for_benchmark(), _max_drawdown_pct(), _model_label(), Benchmark comparison helpers — naive strategies vs model.  Provides equity-curve (+22 more)

### Community 8 - "Metric Zones Dashboard"
Cohesion: 0.08
Nodes (26): Render a metric card with colour-coded zone indicator., render_zoned_metric(), _get_metric_zone(), _is_extreme_value(), Metric zone definitions for CFD backtest benchmarks.  Provides zone classificati, Check if a metric value is extreme and return threshold info.      Args:, Return (color_name, zone_label, recommendation) for a given metric.      Zone la, _equity_series_from_closed_trades() (+18 more)

### Community 9 - "Pipeline CLI UI"
Cohesion: 0.09
Nodes (23): Shared lightweight CLI UI helpers without Rich dependency., Minimal console facade compatible with previous Rich call sites., Log plain text messages., Log a visual separator line., Print a stage banner with concise log output.      Args:         stage: Stage nu, Print a skip line and logger message.      Args:         stage: Stage number (1-, SimpleConsole, stage_header() (+15 more)

### Community 10 - "Triple Barrier Labels"
Cohesion: 0.1
Nodes (27): compute_average_uniqueness(), compute_event_end(), _compute_labels(), _filter_censored(), generate_labels(), _load_inputs(), _log_atr_stats(), _log_distribution() (+19 more)

### Community 11 - "Feature Registry"
Cohesion: 0.1
Nodes (23): build_exclude_cols(), build_feature_output_cols(), build_label_output_cols(), get_gru_feature_cols(), get_label_helper_cols(), get_static_feature_cols(), Single source of truth for feature column lists across pipeline stages., Return the static (non-sequential) feature columns from config. (+15 more)

### Community 12 - "Walk Forward Artifacts"
Cohesion: 0.15
Nodes (22): Persist GRU weights, architecture metadata, and normalization stats.      Args:, save_gru_model(), _build_lgbm_info(), _build_wf_history(), _log_walk_forward_complete(), Walk-forward artifact persistence helpers.  Shared by LGBM-only, GRU-only, and h, Write ``models/training_history.json`` under the session dir if enabled., Write ``reports/walk_forward_history.json`` under the session dir if enabled. (+14 more)

### Community 13 - "Trading Strategy"
Cohesion: 0.12
Nodes (13): CFD backtest simulation package., _calendar_day(), HybridGRUStrategy, HybridGRU trading strategy used in stage 5 backtests., Floor ATR to ``max(atr, self.min_atr)``., Update peak equity, drawdown tracking, and daily loss tracking.          The dra, Check all risk gates before opening a new position., Return fixed position size after confidence filtering.          Scaling lots by (+5 more)

### Community 14 - "Walk Forward Utils"
Cohesion: 0.14
Nodes (19): _add_confidence_columns(), _add_prediction_diagnostics(), _compute_per_class_metrics(), _counts_dict(), _pct_dict(), Shared walk-forward utility functions used by both hybrid and static paths., Convert count dict to rounded percentages., Return start/end timestamps for a window slice. (+11 more)

### Community 15 - "GRU Architecture"
Cohesion: 0.11
Nodes (14): GRUExtractor, GRU architecture components — model and dropout layers.  Separates nn.Module def, Encode a batched sequence into an attention-weighted context vector.          Ar, Variational (locked) dropout — same mask across all timesteps.      Standard dro, Initialise variational dropout with probability *p*., Apply variational dropout.          Args:             x: Tensor with shape ``(ba, GRU-based feature extractor with learned attention pooling.      Encodes a (batc, Initialize the GRU extractor module.          Args:             input_size: Numb (+6 more)

### Community 16 - "Data Charts"
Cohesion: 0.13
Nodes (18): build_candlestick_chart(), build_correlation_heatmap(), build_feature_distribution_chart(), build_feature_distributions_chart(), build_label_distribution_chart(), _downsample_ohlcv(), _get_feature_cols(), Data exploration chart builders. (+10 more)

### Community 17 - "Session Loading"
Cohesion: 0.12
Nodes (17): load_session_data(), Session artifact loading for dashboard charts., Load session artifacts required by interactive chart builders.      Args:, find_sessions(), load_config(), parse_session_meta(), Session discovery, parsing, and loading helpers., Discover available session directories under ``results/``. (+9 more)

### Community 18 - "Static Window Training"
Cohesion: 0.13
Nodes (18): _compute_class_weights(), _compute_distribution_shift_weights(), Compute balanced class weights for multiclass labels.      Args:         y: Targ, Compute per-sample training weights to reduce stale-regime bias.      Compares c, Generate predictions and aligned probability matrix.      Args:         model: T, Build hybrid matrix, train LGBM, predict, return full window result.      Args:, _wf_build_predict_phase(), _wf_format_predictions() (+10 more)

### Community 19 - "Hybrid Training"
Cohesion: 0.16
Nodes (17): Train and evaluate the hybrid GRU + LightGBM model.      This stage trains the G, train_model(), _align_splits_with_sequences(), _build_hybrid_matrix(), _build_interaction_constraints(), _filter_validation_to_seen_classes(), LightGBM utilities, training, and hybrid feature-matrix helpers., Drop validation rows whose class is absent from the training fold.      LightGBM (+9 more)

### Community 20 - "Backtest Charts"
Cohesion: 0.14
Nodes (16): build_duration_pnl_scatter(), build_equity_drawdown_chart(), build_monthly_returns_heatmap(), build_pnl_histogram_chart(), build_rolling_sharpe_chart(), _compute_monthly_returns(), Backtest chart builders., Build a histogram of winning and losing trade PnL values.      Args:         tra (+8 more)

### Community 21 - "Markdown Formatting"
Cohesion: 0.15
Nodes (16): _fmt_dollar(), _fmt_f2(), _fmt_pct(), Shared Markdown formatting helpers for stage 6 reporting.  This module centraliz, Format cells as a markdown table row., _tbl_row(), _backtest_metrics_table(), _backtest_params_table() (+8 more)

### Community 22 - "Calibration Metrics"
Cohesion: 0.17
Nodes (15): brier_score(), calibration_reliability_data(), compute_all_calibration_metrics(), confidence_bins_accuracy(), expected_calibration_error(), log_loss(), Probability calibration metrics: ECE, Brier, log-loss, confidence bins., Return bin centers, accuracies, and counts for calibration curve plotting. (+7 more)

### Community 23 - "Report Data Sections"
Cohesion: 0.19
Nodes (13): _load_label_distribution(), Data-quality and methodology section renderers.  Renderers for data quality, lab, Render the Label Design & Methodology explanation section., Render the Validation Methodology section (walk-forward, purge/embargo)., Render auxiliary regression metrics section (MAE/RMSE/R²) if available., Compute class distribution from the labels parquet file., Render the Data Quality analysis section from the JSON sidecar., _render_auxiliary_regression_section() (+5 more)

### Community 24 - "Report Tables"
Cohesion: 0.19
Nodes (13): _accuracy_table(), _calibration_summary_text(), _compute_ece_numpy(), _exec_table(), _gru_summary(), Markdown table builders and verdict logic for the thesis report.  Each function, Compute a one-paragraph calibration reliability note.      Reads predicted proba, Key ML-first metrics with application-demo metrics second.      Args:         L: (+5 more)

### Community 25 - "Model Charts"
Cohesion: 0.18
Nodes (12): build_confidence_distribution_chart(), build_confusion_matrix_chart(), build_feature_importance_chart(), build_prediction_distribution_chart(), Model performance chart builders., Build a horizontal top-N feature-importance chart.      Args:         fi: Mappin, Build a normalized confusion-matrix heatmap for 3-class labels.      Args:, Build actual-vs-predicted label distribution bars. (+4 more)

### Community 26 - "Backtest Report Sections"
Cohesion: 0.18
Nodes (11): _compute_avg_win_loss_ratio(), Backtest metric zones, baseline comparison, and verdict section renderers.  Cont, Render baseline strategy comparison using the _baselines module., Render sorted issues and recommendations into markdown lines., Compute average win / average loss ratio from trade records., Render backtest metric quality zones with emoji indicators., _render_baseline_comparison_section(), _render_issues() (+3 more)

### Community 27 - "Assessment Verdicts"
Cohesion: 0.2
Nodes (11): _assess_model_quality(), _assess_trading_edge(), _derive_recommendation(), _get_zone_info(), Assessment helpers and constants for backtest verdict logic.  Contains quality t, Classify ML quality into POOR / FAIR / GOOD with a short reason., Classify trading edge into NEGATIVE / MARGINAL / POSITIVE., Produce a deployment recommendation from model quality + trading edge. (+3 more)

### Community 28 - "OOF Predictions"
Cohesion: 0.29
Nodes (8): _collect_oof_predictions(), Build OOF prediction chunk from a single window result.      Args:         resul, _label_suffix(), _one_hot_proba_columns(), _probability_columns(), Return canonical probability-column suffix for a class label., Build one-hot probability columns from predicted class labels., Build canonical probability columns for ``{-1, 0, 1}``.

### Community 29 - "ML Quality Verdict"
Cohesion: 0.25
Nodes (8): _identify_primary_issue(), Return the single most critical issue description, or None., Append one-paragraph ML quality assessment to markdown lines., Append primary issue identification and application demo summary., _render_ml_quality_paragraph(), _render_primary_issue(), _exec_verdict(), One-paragraph ML-first overall assessment with synthesized verdict.      Delegat

### Community 30 - "LGBM Entry Points"
Cohesion: 0.33
Nodes (5): LightGBM training entry points (walk-forward + fixed split)., Train LightGBM with walk-forward validation.      Isolates whether GRU hidden st, Train LightGBM using the fixed train/val/test split.      Args:         config:, train_lgbm_fixed(), train_lgbm_walk_forward()

### Community 31 - "GRU Walk Forward"
Cohesion: 0.4
Nodes (5): GRU-only walk-forward training loop.  Runs the sequence model as a standalone cl, Train GRU with walk-forward validation., Train and evaluate one GRU-only walk-forward window., _run_gru_window(), train_gru_walk_forward()

### Community 32 - "Prediction Saving"
Cohesion: 0.4
Nodes (5): _normalize_label(), Hybrid GRU + LightGBM — static train_model orchestrator., Normalize a class label for probability column naming., Save predictions as Parquet and CSV files., _save_predictions()

### Community 33 - "Reports Dashboard"
Cohesion: 0.33
Nodes (5): Reports section renderer., Render the reports page with thesis markdown and artifact visuals., render_reports_section(), Hide report sections duplicated by dashboard-native charts., trim_generated_visual_sections()

### Community 34 - "Dashboard Metric Cards"
Cohesion: 0.33
Nodes (5): Metric card renderers and CSS constants for the dashboard., Render a styled metric card with gradient background and accent border., render_metric_card(), Render compact direction counts and PnL without low-value charts., render_trade_direction_summary()

### Community 35 - "Chart Shared Helpers"
Cohesion: 0.4
Nodes (5): date_only(), Shared chart renderer and config/trade summary helpers., Return the date part of a config timestamp string., Render compact current experiment settings in the sidebar., render_config_summary()

### Community 36 - "Training Dashboard"
Cohesion: 0.33
Nodes (5): Render a pyecharts chart into the Streamlit app., render_chart(), Training section renderer., Render GRU/LGBM training history and pipeline logs., render_training_section()

### Community 37 - "OOF OOS Report"
Cohesion: 0.5
Nodes (3): OOF vs OOS generalization check section renderer.  Renders the out-of-fold vs ou, Render OOF vs OOS comparison section with side-by-side metrics table., _render_oof_vs_oos_section()

### Community 38 - "Dashboard Main"
Cohesion: 0.5
Nodes (3): main(), Dashboard entry point — sidebar, navigation, and section dispatch., Render the Streamlit dashboard with session selection and navigation.

## Knowledge Gaps
- **420 isolated node(s):** `Thesis ML pipeline — top-level public API surface.  This module re-exports the m`, `Pipeline orchestration for the sequential thesis workflow.  Runs data preparatio`, `Compute an 8-char SHA-256 fingerprint of config sections relevant to a stage.`, `Resolve the effective cache check path based on invalidation strategy.      Args`, `Execute a pipeline stage with cache checking.      Checks the workflow flag and` (+415 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **15 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `_get_metric_zone()` connect `Metric Zones Dashboard` to `Report Tables`?**
  _High betweenness centrality (0.041) - this node is a cross-community bridge._
- **Why does `render_zoned_metric()` connect `Metric Zones Dashboard` to `Dashboard Metric Cards`, `Backtest Charts`?**
  _High betweenness centrality (0.040) - this node is a cross-community bridge._
- **Why does `train_walk_forward()` connect `Pipeline CLI UI` to `GRU Inference Validation`, `LGBM Entry Points`, `GRU Walk Forward`?**
  _High betweenness centrality (0.040) - this node is a cross-community bridge._
- **Are the 16 inferred relationships involving `generate_features()` (e.g. with `_add_atr()` and `_add_context_features()`) actually correct?**
  _`generate_features()` has 16 INFERRED edges - model-reasoned connections that need verification._
- **Are the 19 inferred relationships involving `_build_markdown()` (e.g. with `_model_label()` and `_exec_table()`) actually correct?**
  _`_build_markdown()` has 19 INFERRED edges - model-reasoned connections that need verification._
- **Are the 15 inferred relationships involving `_tbl_row()` (e.g. with `_static_vs_hybrid_comparison()` and `_exec_table()`) actually correct?**
  _`_tbl_row()` has 15 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `train_gru()` (e.g. with `_run_gru_window()` and `_wf_gru_phase()`) actually correct?**
  _`train_gru()` has 9 INFERRED edges - model-reasoned connections that need verification._