"""Application Demo: compact backtest illustration."""

from __future__ import annotations

import streamlit as st

from thesis.charts import build_equity_drawdown_chart
from thesis.dashboard.cards import render_zoned_metric
from thesis.dashboard.shared import render_chart


def render_backtest_section(data: dict, config: object, session_dir: str) -> None:
    """Render a minimal backtest demo, not a trading-performance dashboard."""
    st.markdown("> Home > **Demo**")
    st.header("Application Demo")
    st.info(
        "Backtest only illustrates how predictions can drive a trading simulation. "
        "Do not use it as primary evidence that the ML model is good or bad."
    )

    bt = data.get("backtest_results")
    trades = data.get("trades", [])
    metrics = data.get("metrics", {})

    if not bt:
        st.info("No backtest results available.")
        return

    with st.container(border=True):
        st.subheader("Backtest Demo Summary")
        kpi_cols = st.columns(4, gap="small")
        render_zoned_metric(
            kpi_cols[0],
            "Return",
            metrics.get("return_pct", 0),
            "return_pct",
            "{:.2f}",
            "%",
        )
        render_zoned_metric(
            kpi_cols[1],
            "Max Drawdown",
            metrics.get("max_drawdown_pct", 0),
            "max_drawdown_pct",
            "{:.1f}",
            "%",
        )
        render_zoned_metric(
            kpi_cols[2],
            "Trades",
            metrics.get("num_trades", 0),
            "num_trades",
            "{:.0f}",
        )
        render_zoned_metric(
            kpi_cols[3],
            "Profit Factor",
            metrics.get("profit_factor", 0),
            "profit_factor",
            "{:.2f}",
        )

        st.caption(
            f"Period: {metrics.get('start', 'N/A')[:10]} "
            f"-> {metrics.get('end', 'N/A')[:10]}"
        )
        st.caption(f"Initial capital: ${config.backtest.initial_capital:,.0f}")

    st.subheader("Equity Curve + Drawdown")
    render_chart(
        build_equity_drawdown_chart(
            trades,
            metrics,
            initial_capital=config.backtest.initial_capital,
        ),
        height="600px",
    )


__all__ = ["render_backtest_section"]
