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
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import streamlit as st
from pyecharts import options as opts
from pyecharts.charts import Bar, Line, Pie

# Ensure src/ is on path for imports
_src = str(Path(__file__).resolve().parent.parent.parent)
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

    def parse_session_timestamp(path: Path) -> datetime | None:
        """Extract and parse timestamp from session directory name.

        Session directories follow the pattern: {SYMBOL}_{TIMEFRAME}_{YYYYMMDD}_{HHMMSS}
        e.g., XAUUSD_H1_20260416_143052

        Returns:
            datetime if parsing succeeds, None otherwise.
        """
        import re

        m = re.search(r"(\d{8})_(\d{6})$", path.name)
        if not m:
            return None
        try:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        except ValueError:
            return None

    sessions = sorted(
        [p for p in results.iterdir() if p.is_dir() and (p / "config").exists()],
        key=lambda p: parse_session_timestamp(p) or datetime.min,
        reverse=True,
    )
    return sessions


@st.cache_resource(ttl=60)
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
    # Prefer session snapshot config if available (from a past run), otherwise use default
    snapshot = Path(session_dir) / "config" / "config_snapshot.toml"
    if snapshot.exists():
        config = load_config(snapshot)
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
    if st.button("🔄 Refresh", width="stretch", key="_refresh_btn"):
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


# -----------------------------------------------------------------------------
# Metric Zone Definitions (based on industry benchmarks)
# -----------------------------------------------------------------------------


def _get_metric_zone(metric_name: str, value: float) -> tuple[str, str, str]:
    """
    Return (color_name, zone_label, recommendation) for a given metric.

    Zone colors:
        - excellent (green): Target zone, most strategies should aim here
        - good (light green): Acceptable zone, solid performance
        - moderate (yellow): Marginal, needs attention
        - poor (orange): Below average, review needed
        - dangerous (red): Critical issues, high risk

    Parameters:
        metric_name: The metric key (e.g., 'sharpe_ratio', 'max_drawdown_pct')
        value: The metric value

    Returns:
        Tuple of (color, zone_label, recommendation_text)
    """
    import math

    # Handle NaN/None
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ("moderate", "N/A", "No data available")

    # Sharpe Ratio (higher is better)
    # Source: BoringEdge, LuxAlgo, Quantified Strategies
    if metric_name == "sharpe_ratio":
        if value < 0:
            return ("dangerous", "Negative", "Below risk-free rate — review strategy")
        if value < 0.5:
            return ("dangerous", "Poor", "Below 0.5 — high risk-adjusted cost")
        if value < 1.0:
            return (
                "moderate",
                "Acceptable",
                "0.5-1.0 — acceptable for conservative strategies",
            )
        if value < 1.5:
            return ("good", "Good", "1.0-1.5 — solid risk-adjusted returns")
        if value < 2.0:
            return ("excellent", "Excellent", "1.5-2.0 — hedge fund target range")
        return ("excellent", "Exceptional", ">2.0 — verify no overfitting")

    # Sortino Ratio (same scale as Sharpe)
    if metric_name == "sortino_ratio":
        if value < 0:
            return ("dangerous", "Negative", "Negative — below risk-free rate")
        if value < 0.5:
            return ("dangerous", "Poor", "Below 0.5 — excessive downside risk")
        if value < 1.0:
            return ("moderate", "Acceptable", "0.5-1.0 — acceptable for conservative")
        if value < 1.5:
            return ("good", "Good", "1.0-1.5 — solid downside-adjusted returns")
        if value < 2.0:
            return ("excellent", "Excellent", "1.5-2.0 — very good risk-adjusted")
        return ("excellent", "Exceptional", ">2.0 — verify no overfitting")

    # Max Drawdown (less negative is better)
    # Source: BoringEdge, LuxAlgo (drawdown is stored as negative)
    if metric_name == "max_drawdown_pct":
        if value > -10:
            return ("excellent", "Excellent", "<10% — exceptional capital preservation")
        if value > -20:
            return ("good", "Good", "10-20% — conservative drawdown")
        if value > -30:
            return ("moderate", "Moderate", "20-30% — typical for trend strategies")
        if value > -40:
            return ("poor", "Significant", "30-40% — high, assess suitability")
        if value > -60:
            return ("dangerous", "High Risk", "40-60% — aggressive drawdown")
        return (
            "dangerous",
            "Critical",
            ">60% — similar to buy & hold, question strategy viability",
        )

    # Profit Factor (higher is better)
    # Source: LuxAlgo, Quantified Strategies
    if metric_name == "profit_factor":
        if value < 1.0:
            return ("dangerous", "Losing", "<1.0 — strategy loses money")
        if value < 1.25:
            return ("poor", "Marginal", "1.0-1.25 — barely covers costs")
        if value < 1.5:
            return ("moderate", "Acceptable", "1.25-1.5 — covers costs with margin")
        if value < 2.0:
            return ("good", "Good", "1.5-2.0 — strong profitability")
        if value < 3.0:
            return ("excellent", "Excellent", "2.0-3.0 — very efficient")
        return ("excellent", "Exceptional", ">3.0 — verify no overfitting")

    # Win Rate (higher is better, but context matters)
    # Source: BoringEdge, LuxAlgo
    if metric_name == "win_rate_pct":
        if value < 25:
            return ("poor", "Low", "<25% — very low, requires large R:R")
        if value < 35:
            return ("moderate", "Low", "25-35% — typical for trend-following")
        if value < 45:
            return ("good", "Good", "35-45% — solid for trend strategies")
        if value < 55:
            return ("excellent", "Excellent", "45-55% — strong win rate")
        return (
            "excellent",
            "Very High",
            ">55% — verify no overfitting or data leakage",
        )

    # CAGR / Annual Return (higher is better)
    # Source: BoringEdge
    if metric_name in ("cagr_pct", "return_ann_pct", "return_pct"):
        if value < 5:
            return ("poor", "Underperforming", "<5% — underperforms inflation")
        if value < 10:
            return ("moderate", "Low", "5-10% — barely above index funds")
        if value < 20:
            return ("good", "Acceptable", "10-20% — solid hedge fund territory")
        if value < 40:
            return ("excellent", "Good", "20-40% — exceptional returns")
        if value < 60:
            return ("excellent", "Very High", "40-60% — verify not overfitted")
        return ("dangerous", "Suspicious", ">60% — likely overfitting or short period")

    # Calmar Ratio (CAGR / Max DD — higher is better)
    # Source: BoringEdge, LuxAlgo
    if metric_name == "calmar_ratio":
        if value < 0:
            return ("dangerous", "Negative", "Negative — losses exceed returns")
        if value < 0.5:
            return ("poor", "Weak", "<0.5 — risk outweighs reward")
        if value < 1.0:
            return ("moderate", "Acceptable", "0.5-1.0 — minimum acceptable threshold")
        if value < 1.5:
            return ("good", "Good", "1.0-1.5 — healthy risk/reward balance")
        if value < 2.0:
            return (
                "excellent",
                "Excellent",
                "1.5-2.0 — very strong risk-adjusted returns",
            )
        if value < 3.0:
            return ("excellent", "Outstanding", "2.0-3.0 — exceptional")
        return ("excellent", "Exceptional", ">3.0 — verify no overfitting")

    # SQN (System Quality Number — higher is better)
    # Source: Van Tharp
    if metric_name == "sqn":
        if value < 1.0:
            return ("poor", "Poor", "<1.0 — system has no edge")
        if value < 2.0:
            return ("moderate", "Average", "1.0-2.0 — average system")
        if value < 2.5:
            return ("good", "Good", "2.0-2.5 — good system")
        if value < 3.0:
            return ("excellent", "Excellent", "2.5-3.0 — excellent system")
        return ("dangerous", "Superb?", ">3.0 — verify no overfitting")

    # Risk/Reward Ratio (higher is better)
    if metric_name == "risk_reward_ratio":
        if value < 0.5:
            return ("poor", "Low", "<0.5 — losses exceed wins")
        if value < 1.0:
            return ("moderate", "Below Avg", "0.5-1.0 — marginal")
        if value < 1.5:
            return ("good", "Good", "1.0-1.5 — acceptable")
        if value < 2.0:
            return ("excellent", "Excellent", "1.5-2.0 — strong")
        return ("excellent", "Exceptional", ">2.0 — very efficient")

    # Exposure Time (lower can be better for capital efficiency)
    if metric_name == "exposure_time_pct":
        if value < 20:
            return ("excellent", "Efficient", "<20% — high capital efficiency")
        if value < 40:
            return ("good", "Good", "20-40% — good capital efficiency")
        if value < 60:
            return ("moderate", "Moderate", "40-60% — typical exposure")
        return ("poor", "High", ">60% — significant market exposure")

    # Kelly Criterion (optimal bet size — lower is safer)
    # 0.2-0.3 is typical for trading
    if metric_name == "kelly_criterion":
        if value <= 0:
            return ("dangerous", "Invalid", "0 or negative — no edge")
        if value < 0.2:
            return ("good", "Conservative", "<20% — conservative position sizing")
        if value < 0.3:
            return ("excellent", "Optimal", "20-30% — textbook optimal")
        if value < 0.5:
            return ("moderate", "Aggressive", "30-50% — aggressive, high variance")
        return ("dangerous", "Dangerous", ">50% — extreme risk, unsustainable")

    # Default: no zone
    return ("moderate", "N/A", "No benchmark available")


_ZONE_COLORS = {
    "excellent": "#22c55e",  # green
    "good": "#84cc16",  # lime
    "moderate": "#eab308",  # yellow
    "poor": "#f97316",  # orange
    "dangerous": "#ef4444",  # red
}


def _render_zoned_metric(
    col: object,
    label: str,
    value: float,
    metric_key: str,
    format_str: str = "{:.2f}",
    unit: str = "",
) -> None:
    """
    Render a metric with a color-coded zone indicator and recommendation.

    Parameters:
        col: Streamlit column object
        label: Display label for the metric
        value: Numeric value
        metric_key: Key for zone lookup (e.g., 'sharpe_ratio')
        format_str: Format string for display (e.g., '{:.2f}')
        unit: Optional unit suffix (e.g., '%', ':1')
    """
    color, zone_label, recommendation = _get_metric_zone(metric_key, value)

    # Render zone badge
    hex_color = _ZONE_COLORS.get(color, "#6b7280")

    col.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {hex_color}22 0%, {hex_color}11 100%);
            border-left: 3px solid {hex_color};
            border-radius: 8px;
            padding: 12px 14px;
            margin: 4px 0;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-sizing: border-box;
        ">
            <div>
                <div style="font-size: 0.7rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">{label}</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #e2e8f0; line-height: 1.2;">
                    {format_str.format(value)}{unit}
                </div>
            </div>
            <div style="margin-top: 8px;">
                <span style="
                    background: {hex_color}33;
                    color: {hex_color};
                    padding: 2px 10px;
                    border-radius: 12px;
                    font-size: 0.65rem;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 0.03em;
                ">{zone_label}</span>
                <div style="font-size: 0.65rem; color: #6b7280; margin-top: 4px; line-height: 1.3;">{recommendation}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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


def _render_model_section(data: dict, session_dir: str = "") -> None:
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
        required_cols = {"true_label", "pred_label"}
        if not required_cols.issubset(set(preds.columns)):
            st.warning(
                f"Predictions missing columns: {required_cols - set(preds.columns)}"
            )
            return
        # Compute all metrics
        y_true = preds["true_label"].to_numpy()
        y_pred = preds["pred_label"].to_numpy()
        total = len(y_true)

        # Exact-match accuracy
        exact_acc = float((y_true == y_pred).mean())

        # Directional accuracy (only non-Hold predictions)
        non_hold_mask = (y_true != 0) & (y_pred != 0)
        if non_hold_mask.sum() > 0:
            dir_correct = y_true[non_hold_mask] == y_pred[non_hold_mask]
            dir_acc = float(dir_correct.mean())
            dir_baseline = 0.5  # Random guess for non-Hold
        else:
            dir_acc = 0.0
            dir_baseline = 0.5

        # Per-class metrics
        per_class = {}
        for cls, name in [(-1, "Short"), (0, "Hold"), (1, "Long")]:
            true_mask = y_true == cls
            pred_mask = y_pred == cls
            recall = (
                float((y_pred[true_mask] == cls).mean()) if true_mask.sum() > 0 else 0.0
            )
            precision = (
                float((y_true[pred_mask] == cls).mean()) if pred_mask.sum() > 0 else 0.0
            )
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            per_class[name] = {
                "true_count": int(true_mask.sum()),
                "pred_count": int(pred_mask.sum()),
                "recall": recall,
                "precision": precision,
                "f1": f1,
            }

        # === Primary Metrics Row ===
        st.subheader("Accuracy Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Directional Accuracy",
            f"{dir_acc:.1%}",
            delta=f"+{(dir_acc - dir_baseline) * 100:.1f}pp vs random",
        )
        col2.metric("Exact-Match Accuracy", f"{exact_acc:.1%}")
        col3.metric("Directional Baseline", f"{dir_baseline:.1%}")
        col4.metric("Test Samples", f"{total:,}")

        # === Per-Class Breakdown ===
        st.subheader("Per-Class Performance")
        cls_col1, cls_col2, cls_col3 = st.columns(3)
        for idx, (name, metrics) in enumerate(per_class.items()):
            col = [cls_col1, cls_col2, cls_col3][idx]
            with col:
                st.markdown(f"**{name}**")
                st.caption(
                    f"True: {metrics['true_count']:,} | Predicted: {metrics['pred_count']:,}"
                )
                st.progress(metrics["recall"], text=f"Recall: {metrics['recall']:.1%}")
                st.progress(
                    metrics["precision"], text=f"Precision: {metrics['precision']:.1%}"
                )
                st.progress(metrics["f1"], text=f"F1: {metrics['f1']:.2f}")

        st.divider()

        # === Charts ===
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Confusion Matrix")
            chart = build_confusion_matrix_chart(y_true, y_pred)
            _render_chart(chart, height="500px")
        with col2:
            st.subheader("Confidence Distribution")
            chart = build_confidence_distribution_chart(preds)
            _render_chart(chart, height="500px")

        # === Prediction Distribution ===
        st.subheader("Prediction Distribution")
        pred_counts = {
            "Short": int((y_pred == -1).sum()),
            "Hold": int((y_pred == 0).sum()),
            "Long": int((y_pred == 1).sum()),
        }
        true_counts = {
            "Short": int((y_true == -1).sum()),
            "Hold": int((y_true == 0).sum()),
            "Long": int((y_true == 1).sum()),
        }
        dist_col1, dist_col2 = st.columns(2)
        with dist_col1:
            st.markdown("**Actual Distribution**")
            for name, count in true_counts.items():
                pct = count / total * 100
                st.markdown(f"{name}: {count:,} ({pct:.1f}%)")
        with dist_col2:
            st.markdown("**Predicted Distribution**")
            for name, count in pred_counts.items():
                pct = count / total * 100
                st.markdown(f"{name}: {count:,} ({pct:.1f}%)")
    else:
        st.info("No predictions data available.")

    if fi:
        st.subheader("LightGBM Feature Importance")
        chart = build_feature_importance_chart(fi)
        _render_chart(chart, height="600px")
    else:
        st.info("No feature importance data available.")

    # SHAP summary image from reports
    if session_dir:
        shap_png = Path(session_dir) / "reports" / "shap_summary.png"
        if shap_png.exists():
            st.subheader("SHAP Summary")
            st.image(str(shap_png), width="stretch")


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
        st.caption(
            "Zone indicators based on industry benchmarks for XAU/USD CFD trading"
        )

        # Pre-compute win/loss stats for row3 and cards below
        pnls = [t["pnl"] for t in trades] if trades else []
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        rr = abs(sum(wins) / len(wins) if wins else 0) / (
            abs(sum(losses) / len(losses)) if losses else 1
        )

        # ── Row 1: Return & Risk ──────────────────────────────────────────
        row1_cols = st.columns([1, 1, 1, 1, 1], gap="small")
        _render_zoned_metric(
            row1_cols[0],
            "Total Return",
            metrics.get("return_pct", 0),
            "return_pct",
            "{:.2f}",
            "%",
        )
        _render_zoned_metric(
            row1_cols[1],
            "CAGR",
            metrics.get("cagr_pct", 0),
            "cagr_pct",
            "{:.2f}",
            "%",
        )
        _render_zoned_metric(
            row1_cols[2],
            "Max Drawdown",
            metrics.get("max_drawdown_pct", 0),
            "max_drawdown_pct",
            "{:.1f}",
            "%",
        )
        _render_zoned_metric(
            row1_cols[3],
            "Win Rate",
            metrics.get("win_rate_pct", 0),
            "win_rate_pct",
            "{:.1f}",
            "%",
        )
        _render_zoned_metric(
            row1_cols[4],
            "Profit Factor",
            metrics.get("profit_factor", 0),
            "profit_factor",
            "{:.2f}",
        )

        # ── Row 2: Risk-Adjusted & System ────────────────────────────────
        row2_cols = st.columns([1, 1, 1, 1, 1], gap="small")
        _render_zoned_metric(
            row2_cols[0],
            "Sharpe Ratio",
            metrics.get("sharpe_ratio", 0),
            "sharpe_ratio",
            "{:.2f}",
        )
        _render_zoned_metric(
            row2_cols[1],
            "Sortino Ratio",
            metrics.get("sortino_ratio", 0),
            "sortino_ratio",
            "{:.2f}",
        )
        _render_zoned_metric(
            row2_cols[2],
            "Calmar Ratio",
            metrics.get("calmar_ratio", 0),
            "calmar_ratio",
            "{:.2f}",
        )
        _render_zoned_metric(
            row2_cols[3], "SQN", metrics.get("sqn", 0), "sqn", "{:.2f}"
        )
        _render_zoned_metric(
            row2_cols[4],
            "Exposure",
            metrics.get("exposure_time_pct", 0),
            "exposure_time_pct",
            "{:.1f}",
            "%",
        )

        # ── Row 3: Trade Statistics & Risk ───────────────────────────────
        row3_cols = st.columns([1, 1, 1, 1, 1], gap="small")
        _render_zoned_metric(
            row3_cols[0],
            "Kelly Criterion",
            metrics.get("kelly_criterion", 0),
            "kelly_criterion",
            "{:.3f}",
        )
        _render_zoned_metric(
            row3_cols[1], "Risk/Reward", rr, "risk_reward_ratio", "1:{:.2f}"
        )
        _render_zoned_metric(
            row3_cols[2],
            "Avg Trade",
            metrics.get("avg_trade_pct", 0),
            "avg_trade_pct",
            "{:.2f}",
            "%",
        )
        _render_zoned_metric(
            row3_cols[3],
            "Best Trade",
            metrics.get("best_trade_pct", 0),
            "best_trade_pct",
            "{:.2f}",
            "%",
        )
        _render_zoned_metric(
            row3_cols[4],
            "Worst Trade",
            metrics.get("worst_trade_pct", 0),
            "worst_trade_pct",
            "{:.2f}",
            "%",
        )

        st.divider()

        # Win/Loss stats
        if trades:
            avg_win = sum(wins) / len(wins) if wins else 0
            avg_loss = sum(losses) / len(losses) if losses else 0

            c1, c2 = st.columns(2)
            c1.metric("Winning Trades", f"{len(wins)}", delta=f"Avg ${avg_win:.0f}")
            c2.metric(
                "Losing Trades", f"{len(losses)}", delta=f"Avg ${abs(avg_loss):.0f}"
            )

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
                Bar(init_opts=opts.InitOpts(height="400px"))
                .add_xaxis(["Long", "Short"])
                .add_yaxis(
                    "PnL",
                    [round(long_pnl, 2), round(short_pnl, 2)],
                )
                .set_colors([COLORS["long"], COLORS["short"]])
                .set_global_opts(
                    title_opts=opts.TitleOpts(title="PnL by Direction"),
                    tooltip_opts=opts.TooltipOpts(
                        trigger="axis",
                        formatter="{b}: ${c}",
                    ),
                    xaxis_opts=opts.AxisOpts(type_="category"),
                    yaxis_opts=opts.AxisOpts(
                        axisline_opts=opts.AxisLineOpts(
                            linestyle_opts=opts.LineStyleOpts(is_show=True, opacity=0.5)
                        ),
                    ),
                )
                .set_series_opts(
                    label_opts=opts.LabelOpts(
                        formatter="{b}: ${c}",
                        is_show=True,
                    ),
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
                Line(init_opts=opts.InitOpts(height="550px"))
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
                    yaxis_opts=opts.AxisOpts(name="Loss", is_scale=True),
                    tooltip_opts=opts.TooltipOpts(trigger="axis"),
                    legend_opts=opts.LegendOpts(pos_right="right"),
                    datazoom_opts=[opts.DataZoomOpts(type_="inside", xaxis_index=0)],
                )
            )
            _render_chart(loss_chart, height="550px")

            # Accuracy curve
            acc_chart = (
                Line(init_opts=opts.InitOpts(height="550px"))
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
                    yaxis_opts=opts.AxisOpts(
                        name="Accuracy",
                        is_scale=True,
                        max_=1.0 if max(val_acc) > 0.9 else None,
                    ),
                    tooltip_opts=opts.TooltipOpts(trigger="axis"),
                    legend_opts=opts.LegendOpts(pos_right="right"),
                    datazoom_opts=[opts.DataZoomOpts(type_="inside", xaxis_index=0)],
                )
            )
            _render_chart(acc_chart, height="550px")
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
# Reports Section
# ---------------------------------------------------------------------------


def _render_reports_section(session_dir: str) -> None:
    """
    Render the full Reports section showing all generated static charts and the thesis report.

    Displays:
    - Thesis report (markdown)
    - Equity curve chart
    - SHAP summary
    - Backtest charts (equity drawdown, monthly returns, PnL histogram, duration vs PnL)
    - Model charts (confusion matrix, confidence distribution, feature importance)
    - Data charts (candlestick, feature correlation, label distribution, feature distributions)

    Parameters:
        session_dir (str): Path to the session directory.
    """
    st.markdown("> 🏠 Dashboard > **Reports**")

    session_path = Path(session_dir)
    reports_dir = session_path / "reports"

    # --- Thesis Report ---
    report_md_path = reports_dir / "thesis_report.md"
    if report_md_path.exists():
        content = report_md_path.read_text()
        # Remove section 10 (Visual Evidence & Analytics) from display
        section_10_marker = "## 10. Visual Evidence & Analytics"
        if section_10_marker in content:
            content = content.split(section_10_marker)[0]
        st.markdown(content, unsafe_allow_html=True)
    else:
        st.info("No thesis report available.")

    st.divider()

    # --- Equity Curve ---
    equity_png = reports_dir / "equity_curve.png"
    if equity_png.exists():
        st.subheader("Equity Curve")
        st.image(str(equity_png), width="stretch")

    # --- SHAP Summary ---
    shap_png = reports_dir / "shap_summary.png"
    if shap_png.exists():
        st.subheader("SHAP Feature Importance")
        st.image(str(shap_png), width="stretch")

    st.divider()

    # --- Backtest Charts ---
    bt_charts_dir = reports_dir / "charts" / "backtest"
    if bt_charts_dir.exists():
        st.subheader("Backtest Charts")
        cols = st.columns(2)
        chart_files = [
            ("equity_drawdown.png", "Equity & Drawdown"),
            ("monthly_returns.png", "Monthly Returns"),
            ("pnl_histogram.png", "PnL Distribution"),
            ("duration_vs_pnl.png", "Duration vs PnL"),
        ]
        for idx, (fname, title) in enumerate(chart_files):
            with cols[idx % 2]:
                fpath = bt_charts_dir / fname
                if fpath.exists():
                    st.markdown(f"**{title}**")
                    st.image(str(fpath), width="stretch")

    st.divider()

    # --- Model Charts ---
    model_charts_dir = reports_dir / "charts" / "model"
    if model_charts_dir.exists():
        st.subheader("Model Charts")
        cols = st.columns(2)
        chart_files = [
            ("confusion_matrix.png", "Confusion Matrix"),
            ("confidence_distribution.png", "Confidence Distribution"),
            ("feature_importance.png", "LightGBM Feature Importance"),
        ]
        for idx, (fname, title) in enumerate(chart_files):
            with cols[idx % 2]:
                fpath = model_charts_dir / fname
                if fpath.exists():
                    st.markdown(f"**{title}**")
                    st.image(str(fpath), width="stretch")

    st.divider()

    # --- Data Charts ---
    data_charts_dir = reports_dir / "charts" / "data"
    if data_charts_dir.exists():
        st.subheader("Data Charts")
        cols = st.columns(2)
        chart_files = [
            ("candlestick.png", "Candlestick Chart"),
            ("feature_correlation.png", "Feature Correlation"),
            ("label_distribution.png", "Label Distribution"),
            ("feature_distributions.png", "Feature Distributions"),
        ]
        for idx, (fname, title) in enumerate(chart_files):
            with cols[idx % 2]:
                fpath = data_charts_dir / fname
                if fpath.exists():
                    st.markdown(f"**{title}**")
                    st.image(str(fpath), width="stretch")

    # --- HTML Backtest Chart ---
    bt_html = session_path / "backtest" / "backtest_chart.html"
    if bt_html.exists():
        st.divider()
        st.subheader("Interactive Backtest Chart")
        with open(bt_html) as f:
            html_content = f.read()
        st.iframe(html_content, height=1000)


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
        [
            "📊 Data",
            "🧠 Model",
            "🏃 Training",
            "💰 Backtest",
            "📝 Reports",
        ],
        label_visibility="collapsed",
    )
    section_map = {
        "📊 Data": "Data Exploration",
        "🧠 Model": "Model Performance",
        "🏃 Training": "Training",
        "💰 Backtest": "Backtest Results",
        "📝 Reports": "Reports",
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
        _render_model_section(data, session_path)
    elif section_name == "Training":
        _render_training_section(data, session_path)
    elif section_name == "Reports":
        _render_reports_section(session_path)
    else:
        _render_backtest_section(data)


if __name__ == "__main__":
    main()
