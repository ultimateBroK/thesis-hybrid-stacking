"""Backtest Results section for the Streamlit dashboard."""

from pathlib import Path

import streamlit as st
from pyecharts import options as opts
from pyecharts.charts import Bar, Pie
from streamlit_echarts import st_pyecharts

from thesis.charts import (
    COLORS,
    build_duration_pnl_scatter,
    build_equity_drawdown_chart,
    build_monthly_returns_heatmap,
    build_pnl_histogram_chart,
    build_rolling_sharpe_chart,
)
from thesis.dashboard.cards import _render_zoned_metric


def _render_chart(chart: object, height: str = "500px") -> None:
    try:
        st_pyecharts(chart, height=height)
    except Exception as e:
        st.warning(f"Chart render failed: {e}")


def _render_backtest_section(data: dict) -> None:
    """Render backtest metrics, charts, analysis panels, and CSV downloads.

    Args:
        data: Session data mapping containing backtest results, trades, metrics,
            and optional session directory path for file downloads.
    """
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

        pnls = [t["pnl"] for t in trades] if trades else []
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        avg_loss_abs = abs(sum(losses) / len(losses)) if losses else 0
        rr = abs(sum(wins) / len(wins)) / avg_loss_abs if avg_loss_abs > 0 else 0.0

        st.markdown("**📊 Key Performance Indicators**")
        kpi_cols = st.columns(5, gap="small")
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
            "Sharpe Ratio",
            metrics.get("sharpe_ratio", 0),
            "sharpe_ratio",
            "{:.2f}",
        )
        _render_zoned_metric(
            kpi_cols[2],
            "Max Drawdown",
            metrics.get("max_drawdown_pct", 0),
            "max_drawdown_pct",
            "{:.1f}",
            "%",
        )
        _render_zoned_metric(
            kpi_cols[3],
            "Win Rate",
            metrics.get("win_rate_pct", 0),
            "win_rate_pct",
            "{:.1f}",
            "%",
        )
        _render_zoned_metric(
            kpi_cols[4], "Trades", metrics.get("num_trades", 0), "num_trades", "{:.0f}"
        )

        st.markdown("---")

        st.markdown("**⚖️ Risk-Adjusted Returns**")
        risk_cols = st.columns([1, 1, 1, 1, 1], gap="small")
        _render_zoned_metric(
            risk_cols[0],
            "Sortino Ratio",
            metrics.get("sortino_ratio", 0),
            "sortino_ratio",
            "{:.2f}",
        )
        _render_zoned_metric(
            risk_cols[1],
            "Calmar Ratio",
            metrics.get("calmar_ratio", 0),
            "calmar_ratio",
            "{:.2f}",
        )
        _render_zoned_metric(
            risk_cols[2], "SQN", metrics.get("sqn", 0), "sqn", "{:.2f}"
        )
        _render_zoned_metric(
            risk_cols[3],
            "Volatility",
            metrics.get("volatility_ann_pct", 0),
            "volatility_ann_pct",
            "{:.2f}",
            "%",
        )
        _render_zoned_metric(
            risk_cols[4],
            "Recovery Factor",
            metrics.get("recovery_factor", 0),
            "recovery_factor",
            "{:.2f}",
        )

        st.markdown("**💰 Profitability Metrics**")
        profit_cols = st.columns([1, 1, 1, 1, 1], gap="small")
        _render_zoned_metric(
            profit_cols[0],
            "CAGR",
            metrics.get("cagr_pct", 0),
            "cagr_pct",
            "{:.2f}",
            "%",
        )
        _render_zoned_metric(
            profit_cols[1],
            "Annual Return",
            metrics.get("return_ann_pct", 0),
            "return_ann_pct",
            "{:.2f}",
            "%",
        )
        _render_zoned_metric(
            profit_cols[2],
            "Profit Factor",
            metrics.get("profit_factor", 0),
            "profit_factor",
            "{:.2f}",
        )
        _render_zoned_metric(
            profit_cols[3],
            "Avg Trade",
            metrics.get("avg_trade_pct", 0),
            "avg_trade_pct",
            "{:.2f}",
            "%",
        )
        _render_zoned_metric(
            profit_cols[4],
            "Kelly Criterion",
            metrics.get("kelly_criterion", 0) * 100,
            "kelly_criterion",
            "{:.1f}",
            "%",
        )

        st.markdown("**📈 Trade Analysis**")
        trade_cols = st.columns([1, 1, 1, 1, 1], gap="small")
        _render_zoned_metric(
            trade_cols[0], "Avg Win", metrics.get("avg_win", 0), "avg_win", "${:.0f}"
        )
        _render_zoned_metric(
            trade_cols[1], "Avg Loss", metrics.get("avg_loss", 0), "avg_loss", "${:.0f}"
        )
        _render_zoned_metric(
            trade_cols[2], "Risk/Reward", rr, "risk_reward_ratio", "1:{:.2f}"
        )
        _render_zoned_metric(
            trade_cols[3],
            "Best Trade",
            metrics.get("best_trade_pct", 0),
            "best_trade_pct",
            "{:.2f}",
            "%",
        )
        _render_zoned_metric(
            trade_cols[4],
            "Worst Trade",
            metrics.get("worst_trade_pct", 0),
            "worst_trade_pct",
            "{:.2f}",
            "%",
        )

        st.markdown("**💼 Account Summary**")
        account_cols = st.columns(5, gap="small")
        _render_zoned_metric(
            account_cols[0],
            "Equity Final",
            metrics.get("equity_final", 0),
            "equity_final",
            "${:.0f}",
        )
        _render_zoned_metric(
            account_cols[1],
            "Equity Peak",
            metrics.get("equity_peak", 0),
            "equity_peak",
            "${:.0f}",
        )
        _render_zoned_metric(
            account_cols[2],
            "Commissions",
            metrics.get("commissions", 0),
            "commissions",
            "${:.0f}",
        )
        _render_zoned_metric(
            account_cols[3],
            "Exposure",
            metrics.get("exposure_time_pct", 0),
            "exposure_time_pct",
            "{:.1f}",
            "%",
        )
        account_cols[4].markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #4b556322 0%, #4b556311 100%);
                border-left: 3px solid #4b5563;
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
                    <div style="font-size: 0.7rem; color: inherit; opacity: 0.7; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">Period</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: inherit; line-height: 1.2;">
                        {metrics.get("start", "N/A")[:10]}<br/>→ {metrics.get("end", "N/A")[:10]}
                    </div>
                </div>
                <div style="margin-top: 8px;">
                    <span style="
                        background: #4b556333;
                        color: #9ca3af;
                        padding: 2px 10px;
                        border-radius: 12px;
                        font-size: 0.65rem;
                        font-weight: 700;
                        text-transform: uppercase;
                        letter-spacing: 0.03em;
                    ">Duration</span>
                    <div style="font-size: 0.65rem; color: inherit; opacity: 0.6; margin-top: 4px; line-height: 1.3;">Trading period</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            "🟢 Excellent  🟡 Good  🟠 Moderate  🔴 Poor/Dangerous  ⚪ N/A (context-dependent)"
        )

    st.divider()

    st.subheader("Equity Curve & Drawdown")
    chart = build_equity_drawdown_chart(trades, metrics)
    _render_chart(chart, height="600px")

    st.divider()

    st.subheader("Trade PnL Distribution")
    chart = build_pnl_histogram_chart(trades, metrics)
    _render_chart(chart, height="500px")

    st.subheader("Trade Duration vs PnL")
    chart = build_duration_pnl_scatter(trades)
    _render_chart(chart, height="500px")

    st.divider()

    st.subheader("Monthly Returns")
    chart = build_monthly_returns_heatmap(trades)
    _render_chart(chart, height="400px")

    st.divider()

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
                datazoom_opts=[
                    opts.DataZoomOpts(
                        is_show=False, type_="slider", range_start=0, range_end=100
                    ),
                    opts.DataZoomOpts(type_="inside", range_start=0, range_end=100),
                ],
            )
        )
        _render_chart(returns_chart, height="400px")

        st.divider()

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
                .add_yaxis("PnL", [round(long_pnl, 2), round(short_pnl, 2)])
                .set_colors([COLORS["long"], COLORS["short"]])
                .set_global_opts(
                    title_opts=opts.TitleOpts(title="PnL by Direction"),
                    tooltip_opts=opts.TooltipOpts(
                        trigger="axis", formatter="{b}: ${c}"
                    ),
                    xaxis_opts=opts.AxisOpts(type_="category"),
                    yaxis_opts=opts.AxisOpts(
                        axisline_opts=opts.AxisLineOpts(
                            linestyle_opts=opts.LineStyleOpts(is_show=True, opacity=0.5)
                        ),
                    ),
                )
                .set_series_opts(
                    label_opts=opts.LabelOpts(formatter="{b}: ${c}", is_show=True)
                )
            )
            _render_chart(pnl_dir_chart, height="400px")

    if len(trades) > 30:
        st.divider()
        st.subheader("Rolling Metrics")
        chart = build_rolling_sharpe_chart(trades)
        _render_chart(chart, height="400px")

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
