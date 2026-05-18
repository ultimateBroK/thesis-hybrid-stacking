"""Backtest demo chart."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Grid, Line

from thesis.charts.shared import COLORS


def build_equity_drawdown_chart(
    trades: list[dict],
    metrics: dict,
    initial_capital: float = 10_000.0,
) -> Grid:
    """Equity curve + drawdown subplot for Application Demo."""
    if not trades or initial_capital <= 0:
        return Grid()

    pnls = [t["pnl"] for t in trades]
    equity = [initial_capital]
    for pnl in pnls:
        equity.append(equity[-1] + pnl)

    equity_arr = np.array(equity)
    peak = np.maximum.accumulate(equity_arr)
    drawdown_pct = (equity_arr - peak) / peak * 100

    try:
        times = [pd.to_datetime(trades[0]["entry_time"]).strftime("%Y-%m-%d %H:%M")]
        times.extend(
            pd.to_datetime(t["exit_time"]).strftime("%Y-%m-%d %H:%M") for t in trades
        )
        x_labels = times
    except (ValueError, TypeError, KeyError):
        x_labels = [str(i) for i in range(len(equity))]

    total_trades = metrics.get("num_trades", len(trades))
    total_return = metrics.get("return_pct", 0)
    title = f"Equity Curve — {total_trades} trades, {total_return:.2f}% return"

    equity_line = (
        Line()
        .add_xaxis(x_labels)
        .add_yaxis(
            series_name="Equity",
            y_axis=[round(v, 2) for v in equity],
            is_smooth=False,
            linestyle_opts=opts.LineStyleOpts(width=1.5, color=COLORS["primary"]),
            areastyle_opts=opts.AreaStyleOpts(opacity=0.1),
            label_opts=opts.LabelOpts(is_show=False),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title=title),
            yaxis_opts=opts.AxisOpts(name="Equity (USD)", is_scale=True),
            xaxis_opts=opts.AxisOpts(is_show=False),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            legend_opts=opts.LegendOpts(is_show=False),
            datazoom_opts=[
                opts.DataZoomOpts(
                    is_show=False,
                    type_="slider",
                    xaxis_index=[0, 1],
                    range_start=0,
                    range_end=100,
                ),
                opts.DataZoomOpts(
                    type_="inside",
                    xaxis_index=[0, 1],
                    range_start=0,
                    range_end=100,
                ),
            ],
        )
    )

    dd_line = (
        Line()
        .add_xaxis(x_labels)
        .add_yaxis(
            series_name="Drawdown",
            y_axis=[round(v, 2) for v in drawdown_pct],
            is_smooth=False,
            linestyle_opts=opts.LineStyleOpts(width=0.8, color=COLORS["danger"]),
            areastyle_opts=opts.AreaStyleOpts(opacity=0.4, color=COLORS["danger"]),
            label_opts=opts.LabelOpts(is_show=False),
        )
        .set_global_opts(
            yaxis_opts=opts.AxisOpts(name="Drawdown (%)"),
            xaxis_opts=opts.AxisOpts(name="Trade #"),
            legend_opts=opts.LegendOpts(is_show=False),
        )
    )

    return (
        Grid(init_opts=opts.InitOpts(height="600px"))
        .add(equity_line, grid_opts=opts.GridOpts(pos_top="5%", pos_bottom="35%"))
        .add(dd_line, grid_opts=opts.GridOpts(pos_top="73%", pos_bottom="16%"))
    )


__all__ = ["build_equity_drawdown_chart"]
