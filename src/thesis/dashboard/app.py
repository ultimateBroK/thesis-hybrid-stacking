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
from pathlib import Path

import numpy as np
import polars as pl
import streamlit as st
from pyecharts import options as opts
from pyecharts.charts import Bar, Pie

# Ensure src/ is on path for imports
_src = str(Path(__file__).resolve().parent.parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from streamlit_echarts import st_echarts  # noqa: E402

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
    """Find all session directories under results/."""
    results = Path("results")
    if not results.exists():
        return []
    sessions = sorted(
        [p for p in results.iterdir() if p.is_dir() and (p / "config").exists()],
        key=lambda p: p.name,
        reverse=True,
    )
    return sessions


@st.cache_data(ttl=3600)
def _load_config(session_dir: str) -> dict:
    """Load config and session data for a given session directory."""
    config = load_config()
    config.paths.session_dir = session_dir
    data = load_session_data(config)
    return {"config": config, "data": data}


def _render_chart(chart: object, height: str = "500px") -> None:
    """Render a pyecharts chart in Streamlit via st_echarts."""
    try:
        opts_str = chart.dump_options()
        opts_dict = json.loads(opts_str) if isinstance(opts_str, str) else opts_str
        st_echarts(options=opts_dict, height=height)
    except Exception as e:
        st.warning(f"Chart render failed: {e}")


# ---------------------------------------------------------------------------
# Section Renderers
# ---------------------------------------------------------------------------


def _render_data_section(data: dict, config: object) -> None:
    """Render data exploration charts."""
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
        chart = build_candlestick_chart(ohlcv, config)
        _render_chart(chart, height="700px")
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
    """Render model performance charts."""
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
    """Render backtest analysis charts."""
    st.header("Backtest Results")

    bt = data.get("backtest_results")
    trades = data.get("trades", [])
    metrics = data.get("metrics", {})

    if not bt:
        st.info("No backtest results available.")
        return

    # --- Key Metrics Row ---
    st.subheader("Performance Summary")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Return", f"{metrics.get('return_pct', 0):.1f}%")
    col2.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
    col3.metric("Max Drawdown", f"{metrics.get('max_drawdown_pct', 0):.1f}%")
    col4.metric("Win Rate", f"{metrics.get('win_rate_pct', 0):.1f}%")
    col5.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")

    # --- Detailed Metrics in expander ---
    with st.expander("Detailed Metrics", expanded=False):
        left, right = st.columns(2)
        with left:
            st.markdown(f"""
            **Period**: {metrics.get("start", "N/A")} → {metrics.get("end", "N/A")}  
            **Duration**: {metrics.get("duration", "N/A")}  
            **Exposure**: {metrics.get("exposure_time_pct", 0):.1f}%  
            **Total Trades**: {metrics.get("num_trades", 0)}  
            **Annual Return**: {metrics.get("return_ann_pct", 0):.1f}%  
            **CAGR**: {metrics.get("cagr_pct", 0):.1f}%  
            **Volatility (Ann.)**: {metrics.get("volatility_ann_pct", 0):.1f}%  
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

    # --- Win/Loss stats ---
    if trades:
        pnls = [t["pnl"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Winning Trades", f"{len(wins)}", delta=f"Avg ${avg_win:.0f}")
        col2.metric("Losing Trades", f"{len(losses)}", delta=f"Avg ${avg_loss:.0f}")
        rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        col3.metric("Reward/Risk", f"{rr:.2f}:1")

    # --- Charts ---
    st.subheader("Equity Curve & Drawdown")
    chart = build_equity_drawdown_chart(trades, metrics)
    _render_chart(chart, height="600px")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Trade PnL Distribution")
        chart = build_pnl_histogram_chart(trades, metrics)
        _render_chart(chart, height="500px")
    with col2:
        st.subheader("Duration vs PnL")
        chart = build_duration_pnl_scatter(trades)
        _render_chart(chart, height="500px")

    st.subheader("Monthly Returns")
    chart = build_monthly_returns_heatmap(trades)
    _render_chart(chart, height="400px")

    # --- Trade Returns Bar Chart ---
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

        # --- Direction Distribution Pie ---
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
            # PnL by direction
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
        st.subheader("Rolling Sharpe Ratio")
        chart = build_rolling_sharpe_chart(trades)
        _render_chart(chart, height="400px")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the Streamlit dashboard."""
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
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Sidebar
    st.sidebar.title("📊 Thesis Dashboard")
    st.sidebar.caption("Hybrid GRU + LightGBM — XAU/USD H1")

    # Session selector
    sessions = _find_sessions()
    if not sessions:
        st.error("No session results found. Run `pixi run workflow` first.")
        return

    session_names = [s.name for s in sessions]
    selected = st.sidebar.selectbox(
        "Session",
        options=session_names,
        index=0,
    )

    section = st.sidebar.radio(
        "Section",
        ["Data Exploration", "Model Performance", "Backtest Results"],
        label_visibility="collapsed",
    )

    # Load data
    session_path = str(next(s for s in sessions if s.name == selected))
    loaded = _load_config(session_path)
    config = loaded["config"]
    data = loaded["data"]

    # Session info in sidebar
    st.sidebar.divider()
    st.sidebar.markdown(f"**Session**: `{selected}`")
    st.sidebar.markdown(
        f"**Symbol**: {config.data.symbol} | **TF**: {config.data.timeframe}"
    )
    if data.get("ohlcv") is not None:
        ohlcv = data["ohlcv"]
        st.sidebar.markdown(f"**Data**: {len(ohlcv):,} bars")
    if data.get("trades"):
        st.sidebar.markdown(f"**Trades**: {len(data['trades'])}")
    bt = data.get("backtest_results")
    if bt and "metrics" in bt:
        m = bt["metrics"]
        st.sidebar.markdown(f"**Period**: {m.get('start', '?')} → {m.get('end', '?')}")

    # Render selected section
    if section == "Data Exploration":
        _render_data_section(data, config)
    elif section == "Model Performance":
        _render_model_section(data)
    else:
        _render_backtest_section(data)


if __name__ == "__main__":
    main()
