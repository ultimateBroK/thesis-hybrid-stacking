"""Backtest Results: zone metrics, equity curve, PnL charts, downloads."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from thesis.charts import (
    build_duration_pnl_scatter,
    build_equity_drawdown_chart,
    build_monthly_returns_heatmap,
    build_pnl_histogram_chart,
    build_rolling_sharpe_chart,
)
from thesis.dashboard.cards import render_zoned_metric
from thesis.dashboard.shared import render_chart, render_trade_direction_summary


def render_backtest_section(data: dict, config: object, session_dir: str) -> None:
    """Zone-coded KPIs, equity/drawdown, PnL histograms, monthly heatmap, CSV downloads."""
    st.markdown("> 🏠 Dashboard > **Backtest Results**")
    st.header("Backtest Results")

    bt = data.get("backtest_results")
    trades = data.get("trades", [])
    metrics = data.get("metrics", {})

    if not bt:
        st.info("No backtest results available.")
        return

    # ── Zone-coded KPIs ──
    with st.container(border=True):
        st.subheader("Performance Overview")
        st.caption("Zone indicators: industry benchmarks for XAU/USD CFD trading")

        st.markdown("**📊 Core Financial Metrics**")
        st.caption("Return · Risk · Edge · Consistency · Sample size")

        kpi_cols = st.columns(3, gap="small")
        render_zoned_metric(
            kpi_cols[0], "Total Return",
            metrics.get("return_pct", 0), "return_pct", "{:.2f}", "%",
        )
        render_zoned_metric(
            kpi_cols[1], "Max Drawdown",
            metrics.get("max_drawdown_pct", 0), "max_drawdown_pct", "{:.1f}", "%",
        )
        render_zoned_metric(
            kpi_cols[2], "Profit Factor",
            metrics.get("profit_factor", 0), "profit_factor", "{:.2f}",
        )

        kpi_cols = st.columns(3, gap="small")
        render_zoned_metric(
            kpi_cols[0], "Sharpe Ratio",
            metrics.get("sharpe_ratio", 0), "sharpe_ratio", "{:.2f}",
        )
        render_zoned_metric(
            kpi_cols[1], "Win Rate",
            metrics.get("win_rate_pct", 0), "win_rate_pct", "{:.1f}", "%",
        )
        render_zoned_metric(
            kpi_cols[2], "Trades",
            metrics.get("num_trades", 0), "num_trades", "{:.0f}",
        )

        st.caption(
            f"Period: {metrics.get('start', 'N/A')[:10]} → {metrics.get('end', 'N/A')[:10]}"
        )
        st.caption(f"Initial: ${config.backtest.initial_capital:,.0f}")
        st.caption(f"Final equity: ${metrics.get('equity_final', 0):,.0f}")
        st.caption("🟢 Excellent  🟡 Good  🟠 Moderate  🔴 Poor/Dangerous  ⚪ N/A")

    st.divider()

    # ── Equity Curve + Drawdown ──
    st.subheader("Equity Curve & Drawdown")
    render_chart(
        build_equity_drawdown_chart(
            trades, metrics, initial_capital=config.backtest.initial_capital
        ),
        height="600px",
    )

    st.divider()

    # ── PnL Histogram + Duration Scatter ──
    pnl_col, dur_col = st.columns(2)
    with pnl_col:
        st.subheader("Trade PnL Distribution")
        render_chart(build_pnl_histogram_chart(trades, metrics), height="500px")
    with dur_col:
        st.subheader("Trade Duration vs PnL")
        render_chart(build_duration_pnl_scatter(trades), height="500px")

    st.divider()

    # ── Monthly Returns + Rolling Sharpe ──
    monthly_col, rolling_col = st.columns(2)
    with monthly_col:
        st.subheader("Monthly Returns")
        render_chart(
            build_monthly_returns_heatmap(
                trades, initial_capital=config.backtest.initial_capital
            ),
            height="400px",
        )
    with rolling_col:
        if len(trades) > 30:
            st.subheader("Rolling Metrics")
            render_chart(build_rolling_sharpe_chart(trades), height="400px")
        else:
            st.info("Need > 30 trades for rolling metrics.")

    render_trade_direction_summary(trades)

    st.divider()

    # ── Downloads ──
    st.subheader("Download Data")
    session_dir = data.get("session_dir")
    if not session_dir:
        st.info("No session directory available for downloads.")
        return

    bt_dir = Path(session_dir) / "backtest"
    d1, d2, d3 = st.columns(3)

    csv_path = bt_dir / "trades_detail.csv"
    if csv_path.exists():
        with d1:
            st.download_button(
                "📄 Trades Detail CSV",
                data=csv_path.read_text(),
                file_name="trades_detail.csv",
                mime="text/csv",
            )

    eq_path = bt_dir / "equity_curve.csv"
    if eq_path.exists():
        with d2:
            st.download_button(
                "📈 Equity Curve CSV",
                data=eq_path.read_text(),
                file_name="equity_curve.csv",
                mime="text/csv",
            )

    preds_csv = Path(session_dir) / "predictions" / "final_predictions.csv"
    if preds_csv.exists():
        with d3:
            st.download_button(
                "🎯 Predictions CSV",
                data=preds_csv.read_text(),
                file_name="final_predictions.csv",
                mime="text/csv",
            )