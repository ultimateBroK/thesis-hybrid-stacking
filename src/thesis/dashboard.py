"""Interactive Streamlit dashboard for thesis session visualization.

Launch with ``pixi run streamlit``. The dashboard combines session discovery,
metric cards, zone classification, and five sections: Data, Model, Training,
Backtest, and Reports.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import html
import json
import logging
from pathlib import Path
import re
import sys

import polars as pl
from pyecharts import options as opts
from pyecharts.charts import Line
import streamlit as st

from thesis._shared.session_paths import load_config_for_session

# Zone helpers (from thesis._shared.zones – pure Python, no Streamlit)
from thesis._shared.zones import (
    _ZONE_COLORS,
    _get_metric_zone,
    _is_extreme_value,
)

# Session management & chart builders
from thesis.charts import (
    COLORS,
    EXCLUDED_FEATURE_COLS,
    build_candlestick_chart,
    build_confidence_distribution_chart,
    build_confusion_matrix_chart,
    build_correlation_heatmap,
    build_duration_pnl_scatter,
    build_equity_drawdown_chart,
    build_feature_distribution_chart,
    build_feature_importance_chart,
    build_label_distribution_chart,
    build_monthly_returns_heatmap,
    build_pnl_histogram_chart,
    build_prediction_distribution_chart,
    build_rolling_sharpe_chart,
    load_session_data,
)

logger = logging.getLogger("thesis.app_streamlit")

# Ensure src/ is on sys.path for sibling imports
_src = str(Path(__file__).resolve().parent.parent)
if _src not in sys.path:
    sys.path.insert(0, _src)


# Metric card helpers

# CSS style strings extracted to avoid embedding long inline attributes.
_CSS_METRIC_LABEL = (
    "font-size: 0.7rem; color: inherit; opacity: 0.7; text-transform: uppercase;"
    " letter-spacing: 0.05em; margin-bottom: 4px;"
)
_CSS_METRIC_VALUE = (
    "font-size: 1.5rem; font-weight: 700; color: inherit; line-height: 1.2;"
)
_CSS_METRIC_REC = (
    "font-size: 0.65rem; color: inherit; opacity: 0.6;"
    " margin-top: 4px; line-height: 1.3;"
)


def _render_zoned_metric(
    col: object,
    label: str,
    value: float,
    metric_key: str,
    format_str: str = "{:.2f}",
    unit: str = "",
) -> None:
    """Render a metric card with colour-coded zone indicator."""
    is_extreme, _ = _is_extreme_value(metric_key, value)
    color, zone_label, recommendation = _get_metric_zone(metric_key, value)

    hex_color = _ZONE_COLORS.get(color, "#6b7280")
    display_suffix = " ⚠️" if is_extreme else ""
    safe_label = html.escape(label)
    safe_value = html.escape(format_str.format(value))
    safe_unit = html.escape(unit)
    safe_zone = html.escape(zone_label)
    safe_rec = html.escape(recommendation)

    col.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {hex_color}22 0%, {hex_color}11 100%);
            border-left: 3px solid {hex_color};
            border-radius: 8px;
            padding: 12px 14px;
            margin: 4px 0;
            min-height: 110px;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-sizing: border-box;
        ">
            <div>
                <div style="{_CSS_METRIC_LABEL}">{safe_label}</div>
                <div style="{_CSS_METRIC_VALUE}">
                    {safe_value}{safe_unit}{display_suffix}
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
                ">{safe_zone}</span>
                <div style="{_CSS_METRIC_REC}">{safe_rec}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_metric_card(
    col: object,
    label: str,
    value: str,
    caption: str | None,
    color: str,
) -> None:
    """Render a styled metric card with gradient background and accent border."""
    safe_label = html.escape(label)
    safe_value = html.escape(value)
    caption_html = (
        f'<div style="{_CSS_METRIC_REC}">{html.escape(caption)}</div>'
        if caption
        else ""
    )
    col.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {color}22 0%, {color}11 100%);
            border-left: 3px solid {color};
            border-radius: 8px;
            padding: 12px 14px;
            margin: 4px 0;
            min-height: 90px;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-sizing: border-box;
        ">
            <div>
                <div style="{_CSS_METRIC_LABEL}">{safe_label}</div>
                <div style="{_CSS_METRIC_VALUE}">{safe_value}</div>
            </div>
            {caption_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# Session discovery & loading


def _find_sessions() -> list[Path]:
    """Discover available session directories under ``results/``."""
    results = Path("results")
    if not results.exists():
        return []

    def _parse_session_timestamp(path: Path) -> datetime | None:
        """Parse a session directory name into a datetime."""
        m = re.search(r"(\d{8})_(\d{6})$", path.name)
        if not m:
            return None
        try:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        except ValueError:
            return None

    sessions = sorted(
        [p for p in results.iterdir() if p.is_dir() and (p / "config").exists()],
        key=lambda p: _parse_session_timestamp(p) or datetime.min,
        reverse=True,
    )
    return sessions


def _parse_session_meta(name: str) -> dict[str, str]:
    """Parse a session directory name into metadata fields."""
    parts = name.split("_")
    if len(parts) >= 4:
        return {
            "symbol": parts[0],
            "timeframe": parts[1],
            "date": f"{parts[2][:4]}-{parts[2][4:6]}-{parts[2][6:8]}",
            "time": f"{parts[3][:2]}:{parts[3][2:4]}:{parts[3][4:6]}",
        }
    return {"symbol": "?", "timeframe": "?", "date": "?", "time": "?"}


@st.cache_resource(ttl=60)
def _load_config(session_dir: str) -> dict:
    """Load configuration and session data for *session_dir*."""
    config = load_config_for_session(session_dir)
    data = load_session_data(config)
    return {"config": config, "data": data}


@st.fragment(run_every=30)
def _session_selector_fragment() -> str | None:
    """Render a sidebar session selector and return the chosen session name."""
    sessions = _find_sessions()
    if not sessions:
        return None

    session_names = [s.name for s in sessions]

    known = st.session_state.get("known_sessions", set())
    current_set = set(session_names)
    new_sessions = current_set - known
    if new_sessions and known:
        for ns in sorted(new_sessions):
            meta = _parse_session_meta(ns)
            st.toast(f"🆕 New session: {meta['date']} {meta['time']}", icon="📈")
    st.session_state.known_sessions = current_set

    session_labels = []
    for name in session_names:
        meta = _parse_session_meta(name)
        session_labels.append(
            f"{meta['date']} {meta['time']} ({meta['symbol']} {meta['timeframe']})"
        )

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

    if st.button("🔄 Refresh", width="stretch", key="_refresh_btn"):
        st.rerun()

    st.caption("Run `pixi run workflow` to generate new sessions")
    return selected


# Shared chart renderer


def _render_chart(chart: object, height: str = "500px") -> None:
    """Render a pyecharts chart into the Streamlit app."""
    try:
        from streamlit_echarts import st_pyecharts

        st_pyecharts(chart, height=height)
    except ImportError as e:
        st.warning(f"Chart render failed: {e}")


def _date_only(value: str) -> str:
    """Return the date part of a config timestamp string."""
    return str(value).split()[0]


def _trim_generated_visual_sections(content: str) -> str:
    """Hide report sections duplicated by dashboard-native charts."""
    marker_pattern = re.compile(r"^##\s+\d*\.?\s*Visual Evidence", re.MULTILINE)
    marker = marker_pattern.search(content)
    return content[: marker.start()].rstrip() if marker else content


def _render_config_summary(config: object) -> None:
    """Render compact current experiment settings in the sidebar."""
    train_span = (
        f"{_date_only(config.splitting.train_start)}→"
        f"{_date_only(config.splitting.train_end)}"
    )
    val_span = (
        f"{_date_only(config.splitting.val_start)}→"
        f"{_date_only(config.splitting.val_end)}"
    )
    test_span = (
        f"{_date_only(config.splitting.test_start)}→"
        f"{_date_only(config.splitting.test_end)}"
    )
    st.markdown(
        f"**Data**: {config.data.symbol} {config.data.timeframe}  "
        f"{_date_only(config.data.start_date)}→{_date_only(config.data.end_date)}"
    )
    st.markdown(f"**Split**: train {train_span}  \nval {val_span}  \ntest {test_span}")
    st.markdown(
        f"**Walk-forward**: {config.validation.method}, "
        f"train={config.validation.train_window_bars:,}, "
        f"test={config.validation.test_window_bars:,}, "
        f"purge={config.validation.purge_bars} bars"
    )
    st.markdown(
        f"**GRU**: multiclass, inputs={config.gru.input_size}, "
        f"hidden={config.gru.hidden_size}, seq={config.gru.sequence_length}, "
        f"epochs={config.gru.epochs}"
    )
    st.markdown(
        f"**LightGBM**: {config.model.architecture}, "
        f"objective={config.model.objective}, leaves={config.model.num_leaves}, "
        f"lr={config.model.learning_rate}"
    )
    st.markdown(
        f"**Labels**: horizon={config.labels.horizon_bars}, "
        f"TP={config.labels.atr_tp_multiplier}×ATR, "
        f"SL={config.labels.atr_sl_multiplier}×ATR"
    )
    st.markdown(
        f"**Backtest**: capital=${config.backtest.initial_capital:,.0f}, "
        f"spread={config.backtest.spread_ticks:g} ticks, "
        f"conf≥{config.backtest.confidence_threshold:.2f}"
    )


def _render_trade_direction_summary(trades: list[dict]) -> None:
    """Render compact direction counts and PnL without low-value charts."""
    if not trades:
        return

    long_trades = [t for t in trades if t.get("direction") == "long"]
    short_trades = [t for t in trades if t.get("direction") == "short"]
    long_pnl = sum(float(t.get("pnl", 0)) for t in long_trades)
    short_pnl = sum(float(t.get("pnl", 0)) for t in short_trades)

    with st.expander("Direction summary", expanded=False):
        cols = st.columns(4, gap="small")
        _render_metric_card(
            cols[0], "Long Trades", f"{len(long_trades):,}", None, COLORS["long"]
        )
        _render_metric_card(
            cols[1], "Short Trades", f"{len(short_trades):,}", None, COLORS["short"]
        )
        _render_metric_card(
            cols[2], "Long PnL", f"${long_pnl:,.0f}", None, COLORS["long"]
        )
        _render_metric_card(
            cols[3], "Short PnL", f"${short_pnl:,.0f}", None, COLORS["short"]
        )


# Section: Data Exploration


def _render_data_section(data: dict, config: object) -> None:
    """Render the Data Exploration section with charts and controls."""
    st.markdown("> 🏠 Dashboard > **Data Exploration**")
    st.header("Data Exploration")

    ohlcv = data.get("ohlcv")
    if ohlcv is not None:
        st.caption(
            f"{len(ohlcv):,} bars | "
            f"{ohlcv['timestamp'].cast(pl.Utf8).min()}"
            f" → {ohlcv['timestamp'].cast(pl.Utf8).max()}"
        )
    features = data.get("features")
    labels = data.get("labels")

    if ohlcv is not None and len(ohlcv) > 0:
        st.subheader("Candlestick Chart")

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
                    f"Showing {info['displayed_bars']:,} of"
                    f" {info['total_bars']:,} bars"
                    " (downsampled). Use DataZoom to navigate."
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
        st.caption(
            f"{len(feature_cols)} model-facing columns; raw ATR is label-helper only."
        )
        if feature_cols:
            selected_feature = st.selectbox(
                "Feature", feature_cols, key="_feature_distribution_select"
            )
            chart = build_feature_distribution_chart(features, selected_feature)
            _render_chart(chart, height="450px")
    else:
        st.info("No features data available.")


# Section: Model Performance


def _render_model_section(data: dict, session_dir: str = "") -> None:
    """Render model performance metrics and model-analysis charts."""
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

        y_true = preds["true_label"].to_numpy()
        y_pred = preds["pred_label"].to_numpy()
        total = len(y_true)

        exact_acc = float((y_true == y_pred).mean())

        non_hold_mask = (y_true != 0) & (y_pred != 0)
        if non_hold_mask.sum() > 0:
            dir_correct = y_true[non_hold_mask] == y_pred[non_hold_mask]
            dir_acc = float(dir_correct.mean())
            dir_baseline = 0.5
        else:
            dir_acc = 0.0
            dir_baseline = 0.5

        per_class: dict[str, dict[str, float | int]] = {}
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

        with st.container(border=True):
            st.subheader("Accuracy Metrics")
            st.caption("Model prediction accuracy against test set labels")

            acc_cols = st.columns(4, gap="small")
            _render_metric_card(
                acc_cols[0],
                "Directional Accuracy",
                f"{dir_acc:.1%}",
                f"+{(dir_acc - dir_baseline) * 100:.1f}pp vs random",
                "#3b82f6",
            )
            _render_metric_card(
                acc_cols[1],
                "Exact-Match Accuracy",
                f"{exact_acc:.1%}",
                None,
                "#8b5cf6",
            )
            _render_metric_card(
                acc_cols[2],
                "Directional Baseline",
                f"{dir_baseline:.1%}",
                "Random guess baseline",
                "#6b7280",
            )
            _render_metric_card(
                acc_cols[3],
                "Test Samples",
                f"{total:,}",
                None,
                "#10b981",
            )

        st.subheader("Per-Class Performance")
        cls_col1, cls_col2, cls_col3 = st.columns(3)
        for idx, (name, cls_metrics) in enumerate(per_class.items()):
            col = [cls_col1, cls_col2, cls_col3][idx]
            with col:
                st.markdown(f"**{name}**")
                st.caption(
                    f"True: {cls_metrics['true_count']:,}"
                    f" | Predicted: {cls_metrics['pred_count']:,}"
                )
                st.progress(
                    cls_metrics["recall"], text=f"Recall: {cls_metrics['recall']:.1%}"
                )
                st.progress(
                    cls_metrics["precision"],
                    text=f"Precision: {cls_metrics['precision']:.1%}",
                )
                st.progress(cls_metrics["f1"], text=f"F1: {cls_metrics['f1']:.2f}")

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Confusion Matrix")
            chart = build_confusion_matrix_chart(y_true, y_pred)
            _render_chart(chart, height="500px")
        with col2:
            st.subheader("Confidence Distribution")
            chart = build_confidence_distribution_chart(preds)
            _render_chart(chart, height="500px")

        st.subheader("Prediction Distribution")
        chart = build_prediction_distribution_chart(y_true, y_pred)
        _render_chart(chart, height="400px")
    else:
        st.info("No predictions data available.")

    if fi:
        st.subheader("Feature Importance (Hybrid)")
        chart = build_feature_importance_chart(fi)
        _render_chart(chart, height="600px")
    else:
        st.info("No feature importance data available.")


# Section: Training


def _render_training_section(data: dict, session_dir: str) -> None:
    """Render GRU/LGBM training history and pipeline logs."""
    st.markdown("> 🏠 Dashboard > **Training**")
    st.header("Training History")

    session_path = Path(session_dir)

    history_path = session_path / "models" / "training_history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

        gru_history = history.get("gru", [])
        lgbm_info = history.get("lightgbm", {})

        if gru_history:
            with st.container(border=True):
                st.subheader("GRU Training Progress")
                st.caption("GRU neural network training curves and best epoch metrics")

                epochs = [e["epoch"] for e in gru_history]
                train_loss = [e["train_loss"] for e in gru_history]
                val_loss = [e["val_loss"] for e in gru_history]
                train_acc = [e["train_acc"] for e in gru_history]
                val_acc = [e["val_acc"] for e in gru_history]

                best_epoch = max(gru_history, key=lambda e: e["val_acc"])
                gru_cols = st.columns(4, gap="small")
                _render_metric_card(
                    gru_cols[0],
                    "Best Val Accuracy",
                    f"{best_epoch['val_acc']:.2%}",
                    f"Epoch {best_epoch['epoch']}",
                    "#22c55e",
                )
                _render_metric_card(
                    gru_cols[1],
                    "Best Epoch",
                    f"{best_epoch['epoch']}",
                    f"Val acc: {best_epoch['val_acc']:.2%}",
                    "#3b82f6",
                )
                _render_metric_card(
                    gru_cols[2],
                    "Final Train Loss",
                    f"{gru_history[-1]['train_loss']:.4f}",
                    f"Started: {gru_history[0]['train_loss']:.4f}",
                    "#f59e0b",
                )
                _render_metric_card(
                    gru_cols[3],
                    "Final Val Loss",
                    f"{gru_history[-1]['val_loss']:.4f}",
                    f"Best: {best_epoch['val_loss']:.4f}",
                    "#ef4444",
                )

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
                    datazoom_opts=[
                        opts.DataZoomOpts(
                            is_show=False,
                            type_="slider",
                            xaxis_index=0,
                            range_start=0,
                            range_end=100,
                        ),
                        opts.DataZoomOpts(
                            type_="inside",
                            xaxis_index=0,
                            range_start=0,
                            range_end=100,
                        ),
                    ],
                )
            )
            _render_chart(loss_chart, height="550px")

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
                    datazoom_opts=[
                        opts.DataZoomOpts(
                            is_show=False,
                            type_="slider",
                            xaxis_index=0,
                            range_start=0,
                            range_end=100,
                        ),
                        opts.DataZoomOpts(
                            type_="inside",
                            xaxis_index=0,
                            range_start=0,
                            range_end=100,
                        ),
                    ],
                )
            )
            _render_chart(acc_chart, height="550px")
        else:
            st.info("No GRU training history available.")

        if lgbm_info:
            with st.container(border=True):
                st.subheader("LightGBM Configuration")
                st.caption("Gradient boosting model training parameters and results")

                lgbm_cols = st.columns(3, gap="small")
                _render_metric_card(
                    lgbm_cols[0],
                    "Best Iteration",
                    f"{lgbm_info.get('best_iteration', 'N/A')}",
                    "Optimal boosting round",
                    "#22c55e",
                )
                _render_metric_card(
                    lgbm_cols[1],
                    "Features",
                    f"{lgbm_info.get('n_features', 'N/A')}",
                    "Input feature count",
                    "#3b82f6",
                )
                _render_metric_card(
                    lgbm_cols[2],
                    "Classes",
                    f"{lgbm_info.get('n_classes', 'N/A')}",
                    "Target labels",
                    "#8b5cf6",
                )
    else:
        st.info("No training history file found for this session.")

    st.divider()

    log_path = session_path / "logs" / "pipeline.log"
    if log_path.exists():
        st.subheader("Pipeline Log")

        with open(log_path) as f:
            all_lines = f.readlines()

        with st.expander("Recent Log (last 150 lines)", expanded=True):
            st.code("".join(all_lines[-150:]), language="log")

        with st.expander("Full Pipeline Log", expanded=False):
            st.code("".join(all_lines), language="log")
    else:
        st.info("No pipeline log found for this session.")


# Section: Backtest Results


def _render_backtest_section(data: dict, config: object) -> None:
    """Render backtest metrics, charts, analysis panels, and CSV downloads."""
    st.markdown("> 🏠 Dashboard > **Backtest Results**")
    st.header("Backtest Results")

    bt = data.get("backtest_results")
    trades = data.get("trades", [])
    metrics = data.get("metrics", {})

    if not bt:
        st.info("No backtest results available.")
        return

    with st.container(border=True):
        st.subheader("Performance Overview")
        st.caption(
            "Zone indicators based on industry benchmarks for XAU/USD CFD trading"
        )

        st.markdown("**📊 Core Financial Metrics**")
        st.caption(
            "Kept intentionally small: return, risk, edge,"
            " consistency, and sample size."
        )
        kpi_cols = st.columns(3, gap="small")
        _render_zoned_metric(
            kpi_cols[0],
            "Total Return",
            metrics.get("return_pct", 0),
            "return_pct",
            "{:.2f}",
            "%",
        )
        _render_zoned_metric(
            kpi_cols[1],
            "Max Drawdown",
            metrics.get("max_drawdown_pct", 0),
            "max_drawdown_pct",
            "{:.1f}",
            "%",
        )
        _render_zoned_metric(
            kpi_cols[2],
            "Profit Factor",
            metrics.get("profit_factor", 0),
            "profit_factor",
            "{:.2f}",
        )

        kpi_cols = st.columns(3, gap="small")
        _render_zoned_metric(
            kpi_cols[0],
            "Sharpe Ratio",
            metrics.get("sharpe_ratio", 0),
            "sharpe_ratio",
            "{:.2f}",
        )
        _render_zoned_metric(
            kpi_cols[1],
            "Win Rate",
            metrics.get("win_rate_pct", 0),
            "win_rate_pct",
            "{:.1f}",
            "%",
        )
        _render_zoned_metric(
            kpi_cols[2], "Trades", metrics.get("num_trades", 0), "num_trades", "{:.0f}"
        )

        st.caption(
            f"Period: {metrics.get('start', 'N/A')[:10]} → "
            f"{metrics.get('end', 'N/A')[:10]} | "
            f"Initial balance: ${config.backtest.initial_capital:,.0f} | "
            f"Final equity: ${metrics.get('equity_final', 0):,.0f}"
        )

        st.caption(
            "🟢 Excellent  🟡 Good  🟠 Moderate"
            "  🔴 Poor/Dangerous  ⚪ N/A (context-dependent)"
        )

    st.divider()

    st.subheader("Equity Curve & Drawdown")
    chart = build_equity_drawdown_chart(
        trades, metrics, initial_capital=config.backtest.initial_capital
    )
    _render_chart(chart, height="600px")

    st.divider()

    pnl_col, duration_col = st.columns(2)
    with pnl_col:
        st.subheader("Trade PnL Distribution")
        chart = build_pnl_histogram_chart(trades, metrics)
        _render_chart(chart, height="500px")
    with duration_col:
        st.subheader("Trade Duration vs PnL")
        chart = build_duration_pnl_scatter(trades)
        _render_chart(chart, height="500px")

    st.divider()

    monthly_col, rolling_col = st.columns(2)
    with monthly_col:
        st.subheader("Monthly Returns")
        chart = build_monthly_returns_heatmap(
            trades, initial_capital=config.backtest.initial_capital
        )
        _render_chart(chart, height="400px")
    with rolling_col:
        if len(trades) > 30:
            st.subheader("Rolling Metrics")
            chart = build_rolling_sharpe_chart(trades)
            _render_chart(chart, height="400px")
        else:
            st.info("Need more than 30 trades for rolling metrics.")

    _render_trade_direction_summary(trades)

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


# Section: Reports


def _render_reports_section(session_dir: str) -> None:
    """Render the reports page with thesis markdown and artifact visuals."""
    st.markdown("> 🏠 Dashboard > **Reports**")

    session_path = Path(session_dir)
    reports_dir = session_path / "reports"

    # --- Thesis Report ---
    report_md_path = reports_dir / "thesis_report.md"
    if report_md_path.exists():
        content = _trim_generated_visual_sections(report_md_path.read_text())
        st.markdown(content)
    else:
        st.info("No thesis report available.")

    st.divider()

    # --- Equity Curve ---
    equity_png = reports_dir / "equity_curve.png"
    if equity_png.exists():
        st.subheader("Equity Curve")
        st.image(str(equity_png), width="stretch")

    st.divider()

    # --- Walk-Forward History ---
    wf_path = reports_dir / "walk_forward_history.json"
    if wf_path.exists():
        with open(wf_path) as f:
            wf_data = json.load(f)
        with st.container(border=True):
            st.subheader("Walk-Forward History")
            st.caption("Sliding-window validation summary")
            summary_cols = st.columns(3, gap="small")
            _render_metric_card(
                summary_cols[0],
                "Windows",
                str(wf_data.get("num_windows", "?")),
                "Total walk-forward windows",
                "#3b82f6",
            )
            _render_metric_card(
                summary_cols[1],
                "OOF Predictions",
                f"{wf_data.get('total_oof_predictions', 0):,}",
                "Out-of-fold prediction count",
                "#10b981",
            )
            _render_metric_card(
                summary_cols[2],
                "Architecture",
                str(wf_data.get("architecture", "hybrid")),
                "Model architecture used",
                "#8b5cf6",
            )

            details = wf_data.get("window_details", [])
            if details:
                with st.expander("Window Details", expanded=False):
                    st.dataframe(
                        [
                            {
                                "Window": d["window"],
                                "Train Start": d["train_start_idx"],
                                "Train End": d["train_end_idx"],
                                "Test Start": d["test_start_idx"],
                                "Test End": d["test_end_idx"],
                            }
                            for d in details
                        ],
                        width="stretch",
                        hide_index=True,
                    )

    # --- Feature Importance ---
    fi_path = reports_dir / "feature_importance.json"
    if fi_path.exists():
        with open(fi_path) as f:
            fi_data = json.load(f)
        if fi_data:
            st.divider()
            st.subheader("Feature Importance (Hybrid)")
            chart = build_feature_importance_chart(fi_data)
            _render_chart(chart, height="600px")


# Main entry point


def main() -> None:
    """Render the Streamlit dashboard with session selection and navigation.

    Sets up page layout and styling, discovers and loads a selected session
    from the local results directory, and dispatches rendering to the
    appropriate section renderer.
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

    # ── Session Selector ──
    with st.sidebar.expander("📁 Session", expanded=True):
        selected = _session_selector_fragment()

    if selected is None:
        st.error("No session results found. Run `pixi run workflow` first.")
        return

    # ── Navigation ──
    sections = ["📊 Data", "🧠 Model", "🏃 Training", "💰 Backtest", "📝 Reports"]
    section_map = {
        "📊 Data": "Data Exploration",
        "🧠 Model": "Model Performance",
        "🏃 Training": "Training",
        "💰 Backtest": "Backtest Results",
        "📝 Reports": "Reports",
    }

    current_section = st.session_state.get("nav_section", "📊 Data")

    left_spacer, nav_center, right_spacer = st.columns([0.2, 0.6, 0.2])
    with nav_center:
        nav_cols = st.columns([0.2, 0.2, 0.2, 0.2, 0.2])
        for i, sec in enumerate(sections):
            with nav_cols[i]:
                btn_type = "primary" if sec == current_section else "secondary"
                if st.button(sec, key=f"nav_{sec}", type=btn_type, width="stretch"):
                    st.session_state.nav_section = sec
                    st.rerun()

    section = st.session_state.get("nav_section", "📊 Data")

    # ── Load data ──
    session_path = str(Path("results") / selected)
    loaded = _load_config(session_path)
    config = loaded["config"]
    data = loaded["data"]
    metrics = data.get("metrics", {})

    # ── Configuration sidebar ──
    with st.sidebar.expander("⚙️ Configuration", expanded=False):
        _render_config_summary(config)

    # ── Quick Stats sidebar ──
    if metrics:
        with st.sidebar.expander("📊 Quick Stats", expanded=False):
            c1, c2 = st.columns(2)
            c1.metric("Return", f"{metrics.get('return_pct', 0):.2f}%")
            c2.metric("Win Rate", f"{metrics.get('win_rate_pct', 0):.2f}%")
            c1.metric("Trades", f"{metrics.get('num_trades', 0)}")
            c2.metric("Sharpe", f"{metrics.get('sharpe_ratio', 0):.2f}")

    # ── Render selected section ──
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
        _render_backtest_section(data, config)


if __name__ == "__main__":
    main()
