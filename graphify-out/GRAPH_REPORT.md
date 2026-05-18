# Graph Report - /home/ultimatebrok/Downloads/thesis/src/thesis  (2026-05-19)

## Corpus Check
- Corpus is ~24,755 words - fits in a single context window. You may not need a graph.

## Summary
- 620 nodes · 841 edges · 41 communities (30 shown, 11 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 83 edges (avg confidence: 0.79)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Baseline Models & Metrics|Baseline Models & Metrics]]
- [[_COMMUNITY_Data Preparation|Data Preparation]]
- [[_COMMUNITY_Feature Engineering|Feature Engineering]]
- [[_COMMUNITY_Model Evaluation|Model Evaluation]]
- [[_COMMUNITY_Configuration System|Configuration System]]
- [[_COMMUNITY_Dashboard & Model Viz|Dashboard & Model Viz]]
- [[_COMMUNITY_Label Generation|Label Generation]]
- [[_COMMUNITY_Backtest Demo|Backtest Demo]]
- [[_COMMUNITY_Pipeline & CLI|Pipeline & CLI]]
- [[_COMMUNITY_Data Loading|Data Loading]]
- [[_COMMUNITY_Technical Indicators|Technical Indicators]]
- [[_COMMUNITY_Chart Data|Chart Data]]
- [[_COMMUNITY_Report Generation|Report Generation]]
- [[_COMMUNITY_Risk & Backtest Viz|Risk & Backtest Viz]]
- [[_COMMUNITY_Figure Export|Figure Export]]
- [[_COMMUNITY_Advanced Feature Blocks|Advanced Feature Blocks]]
- [[_COMMUNITY_Model Stacking|Model Stacking]]
- [[_COMMUNITY_Training Pipeline|Training Pipeline]]
- [[_COMMUNITY_Ensemble Training|Ensemble Training]]
- [[_COMMUNITY_Price Action Features|Price Action Features]]
- [[_COMMUNITY_Temporal Features|Temporal Features]]
- [[_COMMUNITY_Trend & Volatility Indicators|Trend & Volatility Indicators]]
- [[_COMMUNITY_Plotting Utilities|Plotting Utilities]]
- [[_COMMUNITY_Report Dashboard|Report Dashboard]]
- [[_COMMUNITY_Regime Detection|Regime Detection]]
- [[_COMMUNITY_Return & Momentum|Return & Momentum]]
- [[_COMMUNITY_Volatility Features|Volatility Features]]
- [[_COMMUNITY_Probability Alignment|Probability Alignment]]
- [[_COMMUNITY_Stacking Data Prep|Stacking Data Prep]]
- [[_COMMUNITY_Walk-Forward Training|Walk-Forward Training]]
- [[_COMMUNITY_Package Init|Package Init]]
- [[_COMMUNITY_Package Init|Package Init]]
- [[_COMMUNITY_Package Init|Package Init]]
- [[_COMMUNITY_Package Init|Package Init]]
- [[_COMMUNITY_Package Init|Package Init]]
- [[_COMMUNITY_Package Init|Package Init]]
- [[_COMMUNITY_Package Init|Package Init]]
- [[_COMMUNITY_Package Init|Package Init]]
- [[_COMMUNITY_Package Init|Package Init]]
- [[_COMMUNITY_Package Init|Package Init]]
- [[_COMMUNITY_Package Init|Package Init]]

## God Nodes (most connected - your core abstractions)
1. `compute_all_classification_metrics()` - 18 edges
2. `_train_stacking_window()` - 15 edges
3. `build_labels()` - 13 edges
4. `build_features()` - 11 edges
5. `create_model_features()` - 10 edges
6. `generate_report()` - 10 edges
7. `render_evaluation_section()` - 9 edges
8. `export_all_figures()` - 9 edges
9. `_save_results()` - 8 edges
10. `MLSignalStrategy` - 8 edges

## Surprising Connections (you probably didn't know these)
- `_run_dataset_stage()` --calls--> `build_labels()`  [INFERRED]
  pipeline.py → dataset/build_labels.py
- `_run_reporting_stage()` --calls--> `run_backtest_demo()`  [INFERRED]
  pipeline.py → demo/backtest_demo.py
- `_run_reporting_stage()` --calls--> `generate_report()`  [INFERRED]
  pipeline.py → reporting/report.py
- `GapClassification` --uses--> `Config`  [INFERRED]
  data/prepare_dataset.py → shared/config.py
- `WalkForwardWindow` --uses--> `Config`  [INFERRED]
  models/train.py → shared/config.py

## Communities (41 total, 11 thin omitted)

### Community 0 - "Baseline Models & Metrics"
Cohesion: 0.06
Nodes (54): always_class(), compute_metrics(), majority_class(), naive_direction(), _per_class_f1(), random_baseline(), Naive baselines. Sanity floor for model skill., Predict previous bar's direction. Persistence. (+46 more)

### Community 1 - "Data Preparation"
Cohesion: 0.07
Nodes (41): _aggregate_all(), _aggregate_file(), check_gap_report(), classify_calendar_gaps(), _classify_gaps_with_calendar(), _classify_gaps_with_heuristic(), _clip_to_month(), _dedupe_and_filter() (+33 more)

### Community 2 - "Feature Engineering"
Cohesion: 0.07
Nodes (39): build_features(), _drop_warmup_rows(), OHLCV -> enriched features pipeline., Unique timestamps + strictly increasing + no nulls (no Pandera)., Write JSON sidecar with model-facing feature column names., Stage 2 feature orchestration: read -> create -> clean -> write., Load Stage 1 OHLCV+ bars., Keep registered feature, helper, and label-input columns only. (+31 more)

### Community 3 - "Model Evaluation"
Cohesion: 0.07
Nodes (40): _add_confidence_columns(), _add_prediction_diagnostics(), _aggregate_oof_summary(), _build_wf_history(), _counts(), _dates(), _label_suffix(), one_hot_proba() (+32 more)

### Community 4 - "Configuration System"
Cohesion: 0.06
Nodes (39): _apply_section(), BacktestConfig, Config, DataConfig, DataRangeConfig, FeaturesConfig, get_config(), LabelsConfig (+31 more)

### Community 5 - "Dashboard & Model Viz"
Cohesion: 0.07
Nodes (32): build_confidence_distribution_chart(), build_confusion_matrix_chart(), build_feature_importance_chart(), build_model_comparison_chart(), Model performance charts., Horizontal top-N feature importance chart., Normalized confusion matrix heatmap for 3-class labels., Accuracy/Macro-F1 comparison across baseline and model variants. (+24 more)

### Community 6 - "Label Generation"
Cohesion: 0.09
Nodes (30): _attach_label_columns(), build_labels(), _check_unique_timestamps(), _compute_triple_barrier(), _drop_censored_and_nan(), _drop_join_artifacts(), _load_features_and_ohlcv(), _log_atr_stats() (+22 more)

### Community 7 - "Backtest Demo"
Cohesion: 0.09
Nodes (25): _calendar_day(), compute_backtest(), _compute_spread_rate(), _load_backtest_data(), _make_commission_fn(), MLSignalStrategy, _normalize_stats(), _prepare_df() (+17 more)

### Community 8 - "Pipeline & CLI"
Cohesion: 0.10
Nodes (20): Shared lightweight CLI UI helpers without Rich dependency., Minimal console facade compatible with previous Rich call sites., Log plain text messages., Log a visual separator line., Print a stage banner with concise log output., Print a skip line and logger message., SimpleConsole, stage_header() (+12 more)

### Community 9 - "Data Loading"
Cohesion: 0.13
Nodes (18): _is_artifact_file(), load_session_data(), Session artifact loading., Read parquet if available; stale paths should not crash dashboard., Read CSV if available; stale paths should not crash dashboard., True only for concrete files; ignore empty/default directory paths., Load session artifacts for chart builders., _read_csv() (+10 more)

### Community 10 - "Technical Indicators"
Cohesion: 0.10
Nodes (19): add_bollinger_pctb(), add_ema_slope(), add_lagged_features(), add_ohlcv_norm(), add_price_dist_ratio(), add_sma_ratio(), add_stochastic(), add_volume_momentum() (+11 more)

### Community 11 - "Chart Data"
Cohesion: 0.14
Nodes (16): build_candlestick_chart(), build_correlation_heatmap(), build_feature_distribution_chart(), build_label_distribution_chart(), _downsample_ohlcv(), _get_feature_cols(), Data exploration charts., Feature cols — exclude metadata. (+8 more)

### Community 12 - "Report Generation"
Cohesion: 0.19
Nodes (17): build_model_comparison_rows(), _build_model_evaluation(), _build_thesis_report(), _fmt_f2(), _fmt_pct(), generate_report(), load_prediction_stats(), _md_table() (+9 more)

### Community 13 - "Risk & Backtest Viz"
Cohesion: 0.13
Nodes (13): build_equity_drawdown_chart(), Equity curve + drawdown subplot for Application Demo., Application Demo: compact backtest illustration., Render a minimal backtest demo, not a trading-performance dashboard., render_backtest_section(), Metric card renderers: zone-based and plain gradient cards., Zone-coloured card: green/yellow/red border + recommendation text., render_zoned_metric() (+5 more)

### Community 14 - "Figure Export"
Cohesion: 0.20
Nodes (16): export_all_figures(), export_confusion_matrix(), export_equity_curve(), export_feature_importance(), export_label_distribution(), export_model_comparison(), _load_label_counts(), Static report figure exports (Matplotlib).  Renders PNG/SVG charts for thesis re (+8 more)

### Community 15 - "Advanced Feature Blocks"
Cohesion: 0.16
Nodes (13): add_microstructure_features(), add_trend_features(), create_model_features(), Business-level feature blocks for Stage 2., Add trend direction and trend-strength features., Add tick-derived volume and activity features., Create minimal causal model features from OHLCV+ bars., add_ema_crossover() (+5 more)

### Community 16 - "Model Stacking"
Cohesion: 0.18
Nodes (13): _classification_summary(), _filter_unseen_classes(), _fit_safe(), Stacking trainer. Base learners feed meta learner.  Simplified: single-holdout o, Fit LightGBM. Early stop when validation usable., Fit model. Dummy protects single-class folds., Stack base probabilities into meta features., Build model comparison metrics. (+5 more)

### Community 17 - "Training Pipeline"
Cohesion: 0.19
Nodes (11): _add_label_prior_features(), _apply_event_purge(), _apply_purge_embargo(), generate_windows(), log_windows(), Walk-forward training: windows, feature pipeline, targets, loop, dispatch., Log window date ranges., Add leakage-safe label priors. (+3 more)

### Community 18 - "Ensemble Training"
Cohesion: 0.17
Nodes (13): _apply_confidence_threshold(), Gate LONG/SHORT by confidence.      When threshold > 0:         LONG  (1) if P(L, _build_base(), _build_meta(), _compute_class_weights(), Build meta learner. Dummy protects single-class meta labels., Train/predict one stacking window.      Protocol: chronological 80/20 split insi, Balanced weights. Counter class skew. (+5 more)

### Community 19 - "Price Action Features"
Cohesion: 0.17
Nodes (12): add_position_features(), Add price-location features normalized by recent range or ATR., add_close_vs_vwap(), add_pivot_position(), add_price_action(), add_vwap(), _ny_trading_day(), Candle structure: body/wick ratios, gap, consecutive bars, price position. (+4 more)

### Community 20 - "Temporal Features"
Cohesion: 0.22
Nodes (10): add_time_features(), Add session and calendar context features., add_calendar(), add_session_dummies(), add_session_range(), _ensure_utc(), NY/London/Asia session flags from UTC→NY hour., Force UTC timezone if missing. (+2 more)

### Community 21 - "Trend & Volatility Indicators"
Cohesion: 0.20
Nodes (10): add_adx(), add_atr(), add_atr_ratio(), Short/long ATR ratio — volatility regime., max(H-L, |H-C_prev|, |L-C_prev|)., EWM alpha=1/period, no adjust., Wilder ATR + close-normalized ATR.      MUST RUN FIRST — many features divide by, Wilder ADX — trend strength from +DI/-DI convergence. (+2 more)

### Community 22 - "Plotting Utilities"
Cohesion: 0.20
Nodes (9): load_feature_importance(), plot_confusion_matrix(), plot_equity_curve(), plot_feature_importance(), Matplotlib/seaborn chart helpers for thesis reporting., Save equity curve PNG from closed-trade list., Save horizontal bar chart of top-N feature importances., Load feature importance JSON from disk. (+1 more)

### Community 23 - "Report Dashboard"
Cohesion: 0.32
Nodes (7): Report section: markdown reports and downloads only., Render report markdown plus downloadable generated files., _render_downloads(), _render_markdown(), render_reports_section(), Strip 'Visual Evidence' section — duplicated by dashboard charts., trim_generated_visual_sections()

### Community 24 - "Regime Detection"
Cohesion: 0.25
Nodes (8): add_optional_regime_features(), Add regime features when enabled by config., add_regime(), add_trend_regime(), add_volatility_regime(), ATR percentile bucketed 0/1/2 (low/normal/high)., EMA slope sign × ADX level → -2..2 scale., Composite: ADX signal × EMA slope sign, clipped [0, clip_max].

### Community 25 - "Return & Momentum"
Cohesion: 0.29
Nodes (7): add_return_features(), Add return and short-horizon momentum features., add_log_returns(), add_macd(), add_rsi(), Multi-horizon log returns (1h, 4h, 1d + config extras)., MACD histogram + ATR-normalized version.

### Community 26 - "Volatility Features"
Cohesion: 0.33
Nodes (6): add_volatility_features(), Add ATR-based volatility and range features., add_atr_percentile(), add_high_low_range(), Rolling ATR rank — relative volatility within lookback., ATR-normalized rolling high-low range.

### Community 27 - "Probability Alignment"
Cohesion: 0.33
Nodes (6): _align_proba(), proba_columns(), Align probabilities to [-1, 0, 1]., Probability columns in canonical order., _aligned_proba(), Predict probabilities in [-1, 0, 1] order.

### Community 28 - "Stacking Data Prep"
Cohesion: 0.33
Nodes (6): _load_labeled_data(), _prepare_for_stacking(), Load labels parquet. Returns (df, is_regression)., Prepare stacking data. Multiclass only., compute_regression_target(), Add forward-return target for regression.      Tail rows cannot see horizon; mar

### Community 29 - "Walk-Forward Training"
Cohesion: 0.33
Nodes (6): Run stacking walk-forward training., train_stacking_walk_forward(), Run walk-forward hooks.      Args:         config: Pipeline config.         prep, Run Hybrid Stacking walk-forward training., run_walk_forward(), train_walk_forward()

## Knowledge Gaps
- **11 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `_run_reporting_stage()` connect `Pipeline & CLI` to `Report Generation`, `Backtest Demo`?**
  _High betweenness centrality (0.395) - this node is a cross-community bridge._
- **Why does `MLSignalStrategy` connect `Backtest Demo` to `Configuration System`?**
  _High betweenness centrality (0.319) - this node is a cross-community bridge._
- **Why does `run_backtest_demo()` connect `Backtest Demo` to `Pipeline & CLI`?**
  _High betweenness centrality (0.307) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `_train_stacking_window()` (e.g. with `select_static_cols()` and `fit_static_feature_pipeline()`) actually correct?**
  _`_train_stacking_window()` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `build_labels()` (e.g. with `_run_dataset_stage()` and `compute_event_end()`) actually correct?**
  _`build_labels()` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `build_features()` (e.g. with `_run_dataset_stage()` and `create_model_features()`) actually correct?**
  _`build_features()` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Thesis ML pipeline — top-level public API surface.  This module re-exports the m`, `Pipeline: data → dataset → models → reporting.`, `8-char SHA-256 fingerprint of stage-relevant config sections.` to the rest of the system?**
  _298 weakly-connected nodes found - possible documentation gaps or missing edges._