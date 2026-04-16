"""Interactive Streamlit dashboard for thesis visualization.

Launch: pixi run streamlit

Features:
- Multi-page sidebar navigation (Data / Model / Backtest)
- Session selector dropdown (auto-detects results/ sessions)
- Interactive ECharts with data zoom, tooltips, selection
- Dark/light theme toggle
- Metrics summary cards
"""

import json
import logging
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
import streamlit as st
from pyecharts import options as opts
from pyecharts.charts import Bar, Line, Pie

# Ensure src/ is on path for imports
_src = str(Path(__file__).resolve().parent.parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from streamlit_echarts import st_pyecharts  # noqa: E402

from thesis.charts import (  # noqa: E402
    COLORS,
    EXCLUDED_FEATURE_COLS,
    build_candlestick_chart,
    build_confidence_distribution_chart,
    build_confusion_matrix_chart,
    build_correlation_heatmap,
    build_duration_pnl_scatter,
    build_equity_drawdown_chart,
    build_feature_importance_chart,
    build_label_distribution_chart,
    build_monthly_returns_heatmap,
    build_pnl_histogram_chart,
    build_rolling_sharpe_chart,
    load_session_data,
)
from thesis.config import load_config  # noqa: E402

logger = logging.getLogger("thesis.app_streamlit")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_sessions() -> list[Path]:
    """
    Discover available session directories under the local results/ folder.
    
    A session directory is any immediate subdirectory that contains a `config` entry.
    If the `results/` folder does not exist, an empty list is returned.
    The returned list is sorted in descending order by directory name.
    
    Returns:
        list[Path]: Reverse-sorted list of session directory paths; empty if none found.
    """
    results = Path("results")
    if not results.exists():
        return []
    sessions = sorted(
        [p for p in results.iterdir() if p.is_dir() and (p / "config").exists()],
        key=lambda p: p.name,
        reverse=True,
    )
    return sessions


@st.cache_data(ttl=60)
def _load_config(session_dir: str) -> dict:
    """
    Load and return the configuration object and associated session data for the given session directory.
    
    This function is cached for a short duration to reduce I/O overhead.
    
    Parameters:
        session_dir (str): Path to the session directory (e.g., "results/<session_name>") or session directory name.
    
    Returns:
        dict: A mapping with keys:
            - "config": the loaded configuration object with `paths.session_dir` set to `session_dir`.
            - "data": the session data loaded according to the configuration.
    """
    config = load_config()
    config.paths.session_dir = session_dir
    data = load_session_data(config)
    return {"config": config, "data": data}


@st.fragment(run_every=30)
def _session_selector_fragment() -> str | None:
    """
    Render a sidebar session selector and return the chosen session directory name.
    
    Updates Streamlit session state keys "known_sessions" (set of discovered session names) and "selected_session" (the currently selected raw session name). When new sessions are discovered (after the initial load) displays a toast for each new session, provides a refresh button, and shows a caption with instructions to generate sessions.
    
    Returns:
        str | None: The raw session directory name selected by the user, or `None` if no sessions are available.
    """
    sessions = _find_sessions()
    if not sessions:
        return None

    session_names = [s.name for s in sessions]

    # Detect new sessions
    known = st.session_state.get("known_sessions", set())
    current_set = set(session_names)
    new_sessions = current_set - known
    if new_sessions and known:  # Skip toast on first load
        for ns in sorted(new_sessions):
            meta = _parse_session_meta(ns)
            st.toast(f"🆕 New session: {meta['date']} {meta['time']}", icon="📈")
    st.session_state.known_sessions = current_set

    # Format labels with metadata
    session_labels = []
    for name in session_names:
        meta = _parse_session_meta(name)
        session_labels.append(
            f"{meta['date']} {meta['time']} ({meta['symbol']} {meta['timeframe']})"
        )

    # Preserve current selection
    current = st.session_state.get("selected_session")
    if current in session_names:
        idx = session_names.index(current)
    else:
        idx = 0
        st.session_state.selected_session = session_names[0]

    selected_label = st.selectbox(
        "Select session",
        options=session_labels,
        index=idx,
        key="_session_selectbox",
    )
    selected = session_names[session_labels.index(selected_label)]
    st.session_state.selected_session = selected

    # Refresh button
    if st.button("🔄 Refresh", use_container_width=True, key="_refresh_btn"):
        st.rerun()

    st.caption("Run `pixi run workflow` to generate new sessions")
    return selected


def _render_chart(chart: object, height: str = "500px") -> None:
    """
    Render a pyecharts chart into the Streamlit app.
    
    If rendering fails, catches the exception and displays a Streamlit warning with the error message.
    """
    try:
        st_pyecharts(chart, height=height)
    except Exception as e:
        st.warning(f"Chart render failed: {e}")


# ---------------------------------------------------------------------------
# Section Renderers
# ---------------------------------------------------------------------------


def _render_data_section(data: dict, config: object) -> None:
    """
    Render the Data Exploration section of the dashboard, producing charts and controls for OHLCV, feature correlations, label distribution, and per-feature distributions.
    
    Parameters:
        data (dict): Session data container expected to possibly include:
            - "ohlcv": tabular OHLCV rows with a "timestamp" column for candlestick plotting and date-range selection.
            - "features": table of feature columns used for correlation heatmap and per-feature histograms.
            - "labels": optional table containing a "label" column for label-distribution plotting.
        config (object): Session configuration object passed to chart builders (used by the candlestick chart builder).
    """
    st.markdown("> 🏠 Dashboard > **Data Exploration**")
    st.header("Data Exploration")

    ohlcv = data.get("ohlcv")
    if ohlcv is not None:
        st.caption(
            f"{len(ohlcv):,} bars | "
            f"{ohlcv['timestamp'].cast(pl.Utf8).min()} → {ohlcv['timestamp'].cast(pl.Utf8).max()}"
        )
    features = data.get("features")
    labels = data.get("labels")

    if ohlcv is not None and len(ohlcv) > 0:
        st.subheader("Candlestick Chart")

        # Date range selector for lazy loading
        ts_col = ohlcv["timestamp"]
        if ts_col.dtype == pl.Utf8:
            ts_parsed = ts_col.str.to_datetime()
        else:
            ts_parsed = ts_col.cast(pl.Datetime)

        min_dt = ts_parsed.min()
        max_dt = ts_parsed.max()
        total_bars = len(ohlcv)

        if min_dt is not None and max_dt is not None:
            min_date = min_dt.date()
            max_date = max_dt.date()
            default_end = max_date
            default_start = max(min_date, max_date - timedelta(days=180))

            col_range1, col_range2 = st.columns(2)
            with col_range1:
                start_date = st.date_input(
                    "From",
                    value=default_start,
                    min_value=min_date,
                    max_value=max_date,
                    key="_candle_start",
                )
            with col_range2:
                end_date = st.date_input(
                    "To",
                    value=default_end,
                    min_value=min_date,
                    max_value=max_date,
                    key="_candle_end",
                )

            # Filter OHLCV to selected range
            start_str = str(start_date)
            end_str = str(end_date) + " 23:59:59"
            ohlcv_filtered = ohlcv.filter(
                (ts_parsed >= pl.lit(start_str).str.to_datetime())
                & (ts_parsed <= pl.lit(end_str).str.to_datetime())
            )
        else:
            ohlcv_filtered = ohlcv

        if len(ohlcv_filtered) > 0:
            chart, info = build_candlestick_chart(ohlcv_filtered, config)
            _render_chart(chart, height="700px")
            if info["total_bars"] < total_bars:
                st.caption(
                    f"Showing {info['displayed_bars']:,} of {total_bars:,} total bars "
                    f"({len(ohlcv_filtered):,} in selected range)"
                )
            elif info["downsampled"]:
                st.caption(
                    f"Showing {info['displayed_bars']:,} of {info['total_bars']:,} bars "
                    f"(downsampled). Use DataZoom to navigate."
                )
        else:
            st.info("No data in selected date range.")
    else:
        st.info("No OHLCV data available.")

    if features is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Feature Correlation")
            chart = build_correlation_heatmap(features)
            _render_chart(chart, height="600px")
        with col2:
            st.subheader("Label Distribution")
            if labels is not None and "label" in labels.columns:
                chart = build_label_distribution_chart(labels)
                _render_chart(chart, height="500px")
            else:
                st.info("No labels data.")

        st.subheader("Feature Distributions")
        feature_cols = [c for c in features.columns if c not in EXCLUDED_FEATURE_COLS]
        if feature_cols:
            # Build individual Bar charts for each feature using tabs
            tabs = st.tabs(feature_cols)
            for col, tab in zip(feature_cols, tabs):
                with tab:
                    vals = features[col].drop_nulls().to_numpy()
                    if len(vals) > 0:
                        counts, bin_edges = np.histogram(vals, bins=50)
                        bin_centers = [
                            (bin_edges[i] + bin_edges[i + 1]) / 2
                            for i in range(len(counts))
                        ]
                        x_labels = [f"{v:.2f}" for v in bin_centers]
                        bar = (
                            Bar(init_opts=opts.InitOpts(height="400px"))
                            .add_xaxis(x_labels)
                            .add_yaxis(
                                series_name=col,
                                y_axis=counts.tolist(),
                                label_opts=opts.LabelOpts(is_show=False),
                                itemstyle_opts=opts.ItemStyleOpts(
                                    color=COLORS["primary"]
                                ),
                            )
                            .set_global_opts(
                                title_opts=opts.TitleOpts(title=f"Distribution: {col}"),
                                xaxis_opts=opts.AxisOpts(name=col),
                                yaxis_opts=opts.AxisOpts(name="Count"),
                                tooltip_opts=opts.TooltipOpts(trigger="axis"),
                                datazoom_opts=[opts.DataZoomOpts(type_="inside")],
                            )
                        )
                        _render_chart(bar, height="400px")
                    else:
                        st.info(f"No data for {col}")
    else:
        st.info("No features data available.")


def _render_model_section(data: dict) -> None:
    """
    Render the "Model Performance" section with prediction metrics and feature-importance charts.
    
    Renders accuracy and basic test statistics, a confusion matrix and confidence-distribution chart when prediction data is present; renders a feature importance chart when feature-importance data is provided. Displays informational messages when predictions or feature-importance are missing.
    
    Parameters:
        data (dict): Session data dictionary. Expected keys:
            - "predictions": table-like object (e.g., DataFrame) containing "true_label" and "pred_label" columns and any additional fields used by the confidence-distribution chart.
            - "feature_importance": dict or sequence describing feature importances, in the format accepted by build_feature_importance_chart.
    """
    st.markdown("> 🏠 Dashboard > **Model Performance**")
    st.header("Model Performance")

    preds = data.get("predictions")
    fi = data.get("feature_importance", {})

    if preds is not None and len(preds) > 0:
        # Metrics cards
        y_true = preds["true_label"].to_numpy()
        y_pred = preds["pred_label"].to_numpy()
        accuracy = float((y_true == y_pred).mean())

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy:.1%}")
        col2.metric("Test Samples", f"{len(y_true):,}")
        col3.metric("Classes", "3 (Short/Hold/Long)")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Confusion Matrix")
            chart = build_confusion_matrix_chart(y_true, y_pred)
            _render_chart(chart, height="500px")
        with col2:
            st.subheader("Confidence Distribution")
            chart = build_confidence_distribution_chart(preds)
            _render_chart(chart, height="500px")
    else:
        st.info("No predictions data available.")

    if fi:
        st.subheader("Feature Importance")
        chart = build_feature_importance_chart(fi)
        _render_chart(chart, height="600px")
    else:
        st.info("No feature importance data available.")


def _render_backtest_section(data: dict) -> None:
    """
    Render the Backtest Results section including summary metrics, visual analyses, and optional CSV downloads.
    
    This function reads these keys from `data` to drive the UI:
    - `backtest_results`: if missing or falsy, the section shows a "No backtest results available." message and returns early.
    - `trades` (list): per-trade records used to build equity/drawdown, PnL distribution, duration vs PnL, individual trade returns, direction analysis, and rolling metrics.
    - `metrics` (dict): summary numeric metrics displayed in the performance overview and the detailed metrics expander.
    - `session_dir` (str or Path, optional): when present and containing expected CSV files, enables download buttons for trades detail, equity curve, and final predictions.
    
    The rendered content includes a performance overview (summary metrics and win/loss stats), equity & drawdown chart, multiple trade-analysis charts, direction/PnL breakdowns, rolling metrics when sufficient trades exist, and download buttons for available CSV artifacts.
    Parameters:
        data (dict): Backtest session data containing at least the keys described above.
    """
    st.markdown("> 🏠 Dashboard > **Backtest Results**")
    st.header("Backtest Results")

    bt = data.get("backtest_results")
    trades = data.get("trades", [])
    metrics = data.get("metrics", {})

    if not bt:
        st.info("No backtest results available.")
        return

    # --- Performance Overview (bordered container) ---
    with st.container(border=True):
        st.subheader("Performance Overview")

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Return", f"{metrics.get('return_pct', 0):.2f}%")
        col2.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
        col3.metric("Max Drawdown", f"{metrics.get('max_drawdown_pct', 0):.2f}%")
        col4.metric("Win Rate", f"{metrics.get('win_rate_pct', 0):.2f}%")
        col5.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")

        # Win/Loss stats
        if trades:
            pnls = [t["pnl"] for t in trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]
            avg_win = sum(wins) / len(wins) if wins else 0
            avg_loss = sum(losses) / len(losses) if losses else 0

            c1, c2, c3 = st.columns(3)
            c1.metric("Winning Trades", f"{len(wins)}", delta=f"Avg ${avg_win:.0f}")
            c2.metric("Losing Trades", f"{len(losses)}", delta=f"Avg ${avg_loss:.0f}")
            rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            c3.metric("Risk/Reward", f"1:{rr:.2f}")

        # Detailed Metrics in expander
        with st.expander("Detailed Metrics", expanded=False):
            left, right = st.columns(2)
            with left:
                st.markdown(f"""
                **Period**: {metrics.get("start", "N/A")} → {metrics.get("end", "N/A")}  
                **Duration**: {metrics.get("duration", "N/A")}  
                **Exposure**: {metrics.get("exposure_time_pct", 0):.2f}%  
                **Total Trades**: {metrics.get("num_trades", 0)}  
                **Annual Return**: {metrics.get("return_ann_pct", 0):.2f}%  
                **CAGR**: {metrics.get("cagr_pct", 0):.2f}%  
                **Volatility (Ann.)**: {metrics.get("volatility_ann_pct", 0):.2f}%  
                """)
            with right:
                st.markdown(f"""
                **Sortino Ratio**: {metrics.get("sortino_ratio", 0):.2f}  
                **Calmar Ratio**: {metrics.get("calmar_ratio", 0):.2f}  
                **SQN**: {metrics.get("sqn", 0):.2f}  
                **Kelly Criterion**: {metrics.get("kelly_criterion", 0):.3f}  
                **Avg Trade**: {metrics.get("avg_trade_pct", 0):.2f}%  
                **Best Trade**: {metrics.get("best_trade_pct", 0):.2f}%  
                **Worst Trade**: {metrics.get("worst_trade_pct", 0):.2f}%  
                """)

    st.divider()

    # --- Equity & Drawdown ---
    st.subheader("Equity Curve & Drawdown")
    chart = build_equity_drawdown_chart(trades, metrics)
    _render_chart(chart, height="600px")

    st.divider()

    # --- Trade Analysis ---
    st.subheader("Trade PnL Distribution")
    chart = build_pnl_histogram_chart(trades, metrics)
    _render_chart(chart, height="500px")

    st.subheader("Trade Duration vs PnL")
    chart = build_duration_pnl_scatter(trades)
    _render_chart(chart, height="500px")

    st.divider()

    # --- Monthly Returns ---
    st.subheader("Monthly Returns")
    chart = build_monthly_returns_heatmap(trades)
    _render_chart(chart, height="400px")

    st.divider()

    # --- Individual Trade Returns ---
    if trades:
        st.subheader("Individual Trade Returns")
        pnls = [t["pnl"] for t in trades]
        x_labels = [str(i) for i in range(len(pnls))]
        win_pnls = [p if p > 0 else 0 for p in pnls]
        loss_pnls = [p if p <= 0 else 0 for p in pnls]

        returns_chart = (
            Bar(init_opts=opts.InitOpts(height="400px"))
            .add_xaxis(x_labels)
            .add_yaxis(
                series_name="Win",
                y_axis=[round(v, 2) for v in win_pnls],
                stack="pnl",
                itemstyle_opts=opts.ItemStyleOpts(color=COLORS["success"]),
                label_opts=opts.LabelOpts(is_show=False),
            )
            .add_yaxis(
                series_name="Loss",
                y_axis=[round(v, 2) for v in loss_pnls],
                stack="pnl",
                itemstyle_opts=opts.ItemStyleOpts(color=COLORS["danger"]),
                label_opts=opts.LabelOpts(is_show=False),
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title="Individual Trade Returns"),
                xaxis_opts=opts.AxisOpts(name="Trade #"),
                yaxis_opts=opts.AxisOpts(name="PnL (USD)"),
                tooltip_opts=opts.TooltipOpts(trigger="axis"),
                legend_opts=opts.LegendOpts(is_show=False),
                datazoom_opts=[opts.DataZoomOpts(type_="inside")],
            )
        )
        _render_chart(returns_chart, height="400px")

        st.divider()

        # --- Direction Analysis ---
        st.subheader("Direction Analysis")
        col_left, col_right = st.columns(2)
        with col_left:
            directions = [t.get("direction", "unknown") for t in trades]
            long_count = directions.count("long")
            short_count = directions.count("short")
            dir_chart = (
                Pie(init_opts=opts.InitOpts(height="400px"))
                .add(
                    series_name="Direction",
                    data_pair=[("Long", long_count), ("Short", short_count)],
                    label_opts=opts.LabelOpts(formatter="{b}: {c} ({d}%)"),
                )
                .set_colors([COLORS["long"], COLORS["short"]])
                .set_global_opts(
                    title_opts=opts.TitleOpts(title="Trade Direction Distribution"),
                    tooltip_opts=opts.TooltipOpts(trigger="item"),
                )
            )
            _render_chart(dir_chart, height="400px")
        with col_right:
            long_pnl = sum(t["pnl"] for t in trades if t.get("direction") == "long")
            short_pnl = sum(t["pnl"] for t in trades if t.get("direction") == "short")
            pnl_dir_chart = (
                Pie(init_opts=opts.InitOpts(height="400px"))
                .add(
                    series_name="PnL",
                    data_pair=[
                        ("Long", round(long_pnl, 2)),
                        ("Short", round(short_pnl, 2)),
                    ],
                    label_opts=opts.LabelOpts(formatter="{b}: ${c}"),
                )
                .set_colors([COLORS["long"], COLORS["short"]])
                .set_global_opts(
                    title_opts=opts.TitleOpts(title="PnL by Direction"),
                    tooltip_opts=opts.TooltipOpts(trigger="item"),
                )
            )
            _render_chart(pnl_dir_chart, height="400px")

    if len(trades) > 30:
        st.divider()
        st.subheader("Rolling Metrics")
        chart = build_rolling_sharpe_chart(trades)
        _render_chart(chart, height="400px")

    # --- Downloads ---
    st.divider()
    st.subheader("Download Data")
    session_dir = data.get("session_dir")
    if session_dir:
        bt_dir = Path(session_dir) / "backtest"
        dl_col1, dl_col2, dl_col3 = st.columns(3)
        with dl_col1:
            csv_path = bt_dir / "trades_detail.csv"
            if csv_path.exists():
                st.download_button(
                    "📄 Trades Detail CSV",
                    data=csv_path.read_text(),
                    file_name="trades_detail.csv",
                    mime="text/csv",
                )
        with dl_col2:
            eq_path = bt_dir / "equity_curve.csv"
            if eq_path.exists():
                st.download_button(
                    "📈 Equity Curve CSV",
                    data=eq_path.read_text(),
                    file_name="equity_curve.csv",
                    mime="text/csv",
                )
        with dl_col3:
            preds_dir = Path(session_dir) / "predictions"
            preds_csv = preds_dir / "final_predictions.csv"
            if preds_csv.exists():
                st.download_button(
                    "🎯 Predictions CSV",
                    data=preds_csv.read_text(),
                    file_name="final_predictions.csv",
                    mime="text/csv",
                )
    else:
        st.info("No session directory available for downloads.")


def _render_training_section(data: dict, session_dir: str) -> None:
    """
    Render the training-history and pipeline logs UI for the selected session in the Streamlit dashboard.
    
    This renders GRU training progress (loss and accuracy curves with summary metrics) when a models/training_history.json file is present, shows LightGBM summary metrics if available, and displays the pipeline.log contents (recent and full) when present. If expected files are missing, informative messages are shown instead.
    
    Parameters:
        data (dict): Loaded session data dictionary (unused directly here but kept for interface consistency).
        session_dir (str): Path to the session directory containing `models/training_history.json` and `logs/pipeline.log`.
    """
    st.markdown("> 🏠 Dashboard > **Training**")
    st.header("Training History")

    session_path = Path(session_dir)

    # --- Training History JSON ---
    history_path = session_path / "models" / "training_history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

        gru_history = history.get("gru", [])
        lgbm_info = history.get("lightgbm", {})

        if gru_history:
            st.subheader("GRU Training Progress")
            epochs = [e["epoch"] for e in gru_history]
            train_loss = [e["train_loss"] for e in gru_history]
            val_loss = [e["val_loss"] for e in gru_history]
            train_acc = [e["train_acc"] for e in gru_history]
            val_acc = [e["val_acc"] for e in gru_history]

            # Metric summary cards
            best_epoch = max(gru_history, key=lambda e: e["val_acc"])
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Best Val Accuracy", f"{best_epoch['val_acc']:.2%}")
            c2.metric("Best Epoch", f"{best_epoch['epoch']}")
            c3.metric("Final Train Loss", f"{gru_history[-1]['train_loss']:.4f}")
            c4.metric("Final Val Loss", f"{gru_history[-1]['val_loss']:.4f}")

            # Loss curve as ECharts
            loss_chart = (
                Line(init_opts=opts.InitOpts(height="400px"))
                .add_xaxis([str(e) for e in epochs])
                .add_yaxis(
                    series_name="Train Loss",
                    y_axis=[round(v, 4) for v in train_loss],
                    linestyle_opts=opts.LineStyleOpts(width=2, color=COLORS["primary"]),
                    label_opts=opts.LabelOpts(is_show=False),
                )
                .add_yaxis(
                    series_name="Val Loss",
                    y_axis=[round(v, 4) for v in val_loss],
                    linestyle_opts=opts.LineStyleOpts(width=2, color=COLORS["danger"]),
                    label_opts=opts.LabelOpts(is_show=False),
                )
                .set_global_opts(
                    title_opts=opts.TitleOpts(title="GRU Loss Curves"),
                    xaxis_opts=opts.AxisOpts(name="Epoch"),
                    yaxis_opts=opts.AxisOpts(name="Loss"),
                    tooltip_opts=opts.TooltipOpts(trigger="axis"),
                    legend_opts=opts.LegendOpts(),
                )
            )
            _render_chart(loss_chart, height="400px")

            # Accuracy curve
            acc_chart = (
                Line(init_opts=opts.InitOpts(height="400px"))
                .add_xaxis([str(e) for e in epochs])
                .add_yaxis(
                    series_name="Train Accuracy",
                    y_axis=[round(v, 4) for v in train_acc],
                    linestyle_opts=opts.LineStyleOpts(width=2, color=COLORS["primary"]),
                    label_opts=opts.LabelOpts(is_show=False),
                )
                .add_yaxis(
                    series_name="Val Accuracy",
                    y_axis=[round(v, 4) for v in val_acc],
                    linestyle_opts=opts.LineStyleOpts(width=2, color=COLORS["success"]),
                    label_opts=opts.LabelOpts(is_show=False),
                )
                .set_global_opts(
                    title_opts=opts.TitleOpts(title="GRU Accuracy Curves"),
                    xaxis_opts=opts.AxisOpts(name="Epoch"),
                    yaxis_opts=opts.AxisOpts(name="Accuracy"),
                    tooltip_opts=opts.TooltipOpts(trigger="axis"),
                    legend_opts=opts.LegendOpts(),
                )
            )
            _render_chart(acc_chart, height="400px")
        else:
            st.info("No GRU training history available.")

        if lgbm_info:
            st.subheader("LightGBM Configuration")
            c1, c2, c3 = st.columns(3)
            c1.metric("Best Iteration", f"{lgbm_info.get('best_iteration', 'N/A')}")
            c2.metric("Features", f"{lgbm_info.get('n_features', 'N/A')}")
            c3.metric("Classes", f"{lgbm_info.get('n_classes', 'N/A')}")
    else:
        st.info("No training history file found for this session.")

    st.divider()

    # --- Pipeline Log ---
    log_path = session_path / "logs" / "pipeline.log"
    if log_path.exists():
        st.subheader("Pipeline Log")

        # Show last 150 lines by default
        with open(log_path) as f:
            all_lines = f.readlines()

        with st.expander("Recent Log (last 150 lines)", expanded=True):
            st.code("".join(all_lines[-150:]), language="log")

        with st.expander("Full Pipeline Log", expanded=False):
            st.code("".join(all_lines), language="log")
    else:
        st.info("No pipeline log found for this session.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_session_meta(name: str) -> dict[str, str]:
    """
    Parse a session directory name into its metadata fields.
    
    Parameters:
        name (str): Session directory name, expected as `SYMBOL_TIMEFRAME_YYYYMMDD_HHMMSS`.
    
    Returns:
        dict[str, str]: A mapping with keys `symbol`, `timeframe`, `date`, and `time`. `date` is formatted as `YYYY-MM-DD` and `time` as `HH:MM:SS`. If the name cannot be parsed, each value is the placeholder `"?"`.
    """
    parts = name.split("_")
    if len(parts) >= 4:
        return {
            "symbol": parts[0],
            "timeframe": parts[1],
            "date": f"{parts[2][:4]}-{parts[2][4:6]}-{parts[2][6:8]}",
            "time": f"{parts[3][:2]}:{parts[3][2:4]}:{parts[3][4:6]}",
        }
    return {"symbol": "?", "timeframe": "?", "date": "?", "time": "?"}


def main() -> None:
    """
    Start and render the Streamlit dashboard for thesis experiment results, providing session selection, configuration display, and navigation between main inspection sections.
    
    Sets up the page layout and styling, discovers and loads a selected session from the local results directory, and dispatches rendering to the appropriate section renderer (Data Exploration, Model Performance, Training, or Backtest).
    """
    st.set_page_config(
        page_title="Thesis Dashboard — XAU/USD",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
    <style>
        /* AMOLED glass effect for metric cards */
        .stMetric {
            background: linear-gradient(135deg,
                rgba(255,255,255,0.05) 0%,
                rgba(255,255,255,0.02) 100%);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 14px 16px;
            box-shadow: 0 4px 24px rgba(0,0,0,0.3),
                        inset 0 1px 0 rgba(255,255,255,0.06);
        }
        .stMetric label {
            font-size: 0.8rem;
            color: rgba(255,255,255,0.6);
            letter-spacing: 0.02em;
        }
        .stMetric div[data-testid="stMetricValue"] {
            font-size: 1.5rem;
            font-weight: 700;
            color: #e2e8f0;
        }
        .stMetric div[data-testid="stMetricDelta"] {
            font-size: 0.85rem;
        }
        /* Subtle glow on hover */
        .stMetric:hover {
            border-color: rgba(255,255,255,0.15);
            box-shadow: 0 4px 30px rgba(0,0,0,0.4),
                        inset 0 1px 0 rgba(255,255,255,0.1),
                        0 0 20px rgba(37,99,235,0.05);
            transition: all 0.2s ease;
        }
        /* Compact sidebar spacing */
        .stSidebar .stExpander details summary {
            font-weight: 600;
            font-size: 0.9rem;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # ── Sidebar Header ──
    st.sidebar.markdown("### 📈 Thesis Dashboard")
    st.sidebar.caption("Hybrid GRU + LightGBM — XAU/USD")

    # ── Session Selector (auto-refresh via fragment) ──
    with st.sidebar.expander("📁 Session", expanded=True):
        selected = _session_selector_fragment()

    if selected is None:
        st.error("No session results found. Run `pixi run workflow` first.")
        return

    # ── Section Navigation ──
    section = st.sidebar.radio(
        "Navigation",
        ["📊 Data", "🧠 Model", "🏃 Training", "💰 Backtest"],
        label_visibility="collapsed",
    )
    section_map = {
        "📊 Data": "Data Exploration",
        "🧠 Model": "Model Performance",
        "🏃 Training": "Training",
        "💰 Backtest": "Backtest Results",
    }

    # Load data
    session_path = str(Path("results") / selected)
    loaded = _load_config(session_path)
    config = loaded["config"]
    data = loaded["data"]
    metrics = data.get("metrics", {})

    # ── Configuration (collapsed expander) ──
    with st.sidebar.expander("⚙️ Configuration", expanded=False):
        st.markdown(
            f"**GRU**: hidden={config.gru.hidden_size}, layers={config.gru.num_layers}, "
            f"seq={config.gru.sequence_length}, epochs={config.gru.epochs}"
        )
        st.markdown(
            f"**LightGBM**: leaves={config.model.num_leaves}, "
            f"depth={config.model.max_depth}, lr={config.model.learning_rate}"
        )
        st.markdown(
            f"**Backtest**: leverage={config.backtest.leverage}:1, "
            f"lots={config.backtest.lots_per_trade}, "
            f"conf≥{config.backtest.confidence_threshold}"
        )
        st.markdown(
            f"**Split**: train={config.splitting.train_start[:10]}→"
            f"{config.splitting.train_end[:10]}"
        )

    # ── Quick Stats (collapsed expander) ──
    if metrics:
        with st.sidebar.expander("📊 Quick Stats", expanded=False):
            c1, c2 = st.columns(2)
            c1.metric("Return", f"{metrics.get('return_pct', 0):.2f}%")
            c2.metric("Win Rate", f"{metrics.get('win_rate_pct', 0):.2f}%")
            c1.metric("Trades", f"{metrics.get('num_trades', 0)}")
            c2.metric("Sharpe", f"{metrics.get('sharpe_ratio', 0):.2f}")

    # Render selected section
    section_name = section_map[section]
    if section_name == "Data Exploration":
        _render_data_section(data, config)
    elif section_name == "Model Performance":
        _render_model_section(data)
    elif section_name == "Training":
        _render_training_section(data, session_path)
    else:
        _render_backtest_section(data)


if __name__ == "__main__":
    main()
