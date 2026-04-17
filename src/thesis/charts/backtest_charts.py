"""Backtest interactive ECharts: equity/drawdown, PnL, monthly returns, rolling Sharpe, scatter."""

from datetime import datetime

import numpy as np
import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Bar, Grid, HeatMap, Line, Scatter

from .data import COLORS


def build_equity_drawdown_chart(
    trades: list[dict],
    metrics: dict,
    initial_capital: float = 10_000.0,
) -> Grid:
    """
    Create an equity curve chart with a drawdown subplot arranged in a two-row Grid.

    The Grid places the equity line in the top region (~75%) and the drawdown area in the bottom region (~25%). Expects each trade dict in `trades` to contain a numeric `pnl`. `metrics` is used to populate the chart title (e.g., total trades and return percent).

    Parameters:
        trades (list[dict]): List of trade records; each must include a numeric 'pnl' key.
        metrics (dict): Backtest metrics used in the chart title (e.g., "total_trades", "return_pct").
        initial_capital (float): Starting capital used to build the equity curve.

    Returns:
        pyecharts.charts.Grid: Grid containing the equity line chart (top) and drawdown chart (bottom).
    """
    if not trades or initial_capital <= 0:
        return Grid()

    pnls = [t["pnl"] for t in trades]
    equity = [initial_capital]
    for p in pnls:
        equity.append(equity[-1] + p)

    equity_arr = np.array(equity)
    peak = np.maximum.accumulate(equity_arr)
    drawdown_pct = (equity_arr - peak) / peak * 100

    try:
        times = [pd.to_datetime(trades[0]["entry_time"]).strftime("%Y-%m-%d %H:%M")]
        for t in trades:
            times.append(pd.to_datetime(t["exit_time"]).strftime("%Y-%m-%d %H:%M"))
        x_labels = times
    except Exception:
        x_labels = [str(i) for i in range(len(equity))]

    total_trades = metrics.get("total_trades", len(trades))
    total_return = metrics.get("return_pct", 0)

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
            title_opts=opts.TitleOpts(
                title=f"Equity Curve — {total_trades} trades, {total_return:.2f}% return"
            ),
            yaxis_opts=opts.AxisOpts(name="Equity (USD)", is_scale=True),
            xaxis_opts=opts.AxisOpts(is_show=False),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            legend_opts=opts.LegendOpts(is_show=False),
            datazoom_opts=[
                opts.DataZoomOpts(
                    type_="slider", xaxis_index=[0, 1], pos_bottom="4%", height=28
                ),
                opts.DataZoomOpts(type_="inside", xaxis_index=[0, 1]),
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

    grid = (
        Grid(init_opts=opts.InitOpts(height="600px"))
        .add(equity_line, grid_opts=opts.GridOpts(pos_top="5%", pos_bottom="35%"))
        .add(dd_line, grid_opts=opts.GridOpts(pos_top="73%", pos_bottom="16%"))
    )
    return grid


def build_pnl_histogram_chart(
    trades: list[dict],
    metrics: dict,
) -> Bar:
    """
    Create a histogram of trade PnL that splits wins and losses into two bar series.

    The function separates trade PnLs into wins (pnl > 0) and losses (pnl <= 0), bins the values across the observed PnL range, and produces a pyecharts Bar chart with one series for wins (green) and one for losses (red). The chart title includes average win and average loss taken from `metrics` (or computed from the data).

    Parameters:
        trades (list[dict]): Sequence of trade dictionaries; each must contain a numeric `'pnl'` key.
        metrics (dict): Backtest metrics used to populate title values (e.g., `avg_win`, `avg_loss`).

    Returns:
        Bar: A pyecharts Bar chart where each x-axis bin shows counts for wins and losses.
    """
    if not trades:
        return Bar()

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    all_pnls = np.array(pnls)
    if all_pnls.min() == all_pnls.max():
        # Constant PnL: create a single centred bin so the histogram still renders
        center = all_pnls.min()
        bins = np.array([center - 0.5, center + 0.5])
    else:
        bins = np.linspace(all_pnls.min(), all_pnls.max(), 51)
    win_counts, _ = np.histogram(wins, bins=bins)
    loss_counts, _ = np.histogram(losses, bins=bins)
    bin_labels = [f"{bins[i]:.0f}" for i in range(len(bins) - 1)]

    avg_win = metrics.get("avg_win", np.mean(wins) if wins else 0)
    avg_loss = metrics.get("avg_loss", np.mean(losses) if losses else 0)

    chart = (
        Bar(init_opts=opts.InitOpts(height="500px"))
        .add_xaxis(bin_labels)
        .add_yaxis(
            series_name=f"Wins ({len(wins)})",
            y_axis=win_counts.tolist(),
            itemstyle_opts=opts.ItemStyleOpts(color=COLORS["success"]),
            label_opts=opts.LabelOpts(is_show=False),
        )
        .add_yaxis(
            series_name=f"Losses ({len(losses)})",
            y_axis=loss_counts.tolist(),
            itemstyle_opts=opts.ItemStyleOpts(color=COLORS["danger"]),
            label_opts=opts.LabelOpts(is_show=False),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title=f"Trade PnL — Avg Win: ${avg_win:.0f}, Avg Loss: ${avg_loss:.0f}"
            ),
            xaxis_opts=opts.AxisOpts(name="PnL (USD)"),
            yaxis_opts=opts.AxisOpts(name="Count"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            legend_opts=opts.LegendOpts(),
        )
    )
    return chart


def _compute_monthly_returns(
    trades: list[dict],
    initial_capital: float = 10_000.0,
) -> dict[tuple[int, int], float]:
    """
    Compute percentage returns for each month based on sequential trade PnL applied to an equity curve.

    Processes trades in order, updating running equity starting from `initial_capital`. For each trade with a parsable `exit_time`, the function records the equity at the first trade seen in that (year, month) as the month's start equity and the equity after the last trade in that month as the month's end equity; the monthly percentage return is (end - start) / start * 100. Trades with unparseable `exit_time` values are skipped. Months where the start equity is less than or equal to zero are omitted.

    Parameters:
        trades (list[dict]): Sequence of trade records; each trade is expected to contain numeric `pnl` and an `exit_time` parseable by ISO format.
        initial_capital (float): Starting equity before applying any trades.

    Returns:
        dict[tuple[int, int], float]: Mapping from `(year, month)` tuples to the monthly percentage return (percent).
    """
    equity = initial_capital
    equity_by_month: dict[tuple[int, int], tuple[float, float]] = {}

    for t in trades:
        try:
            exit_time = datetime.fromisoformat(
                str(t["exit_time"]).replace("Z", "+00:00")
            )
            key = (exit_time.year, exit_time.month)
            start_eq = equity
            equity += t["pnl"]
            end_eq = equity
            if key not in equity_by_month:
                equity_by_month[key] = (start_eq, end_eq)
            else:
                old_start, _old_end = equity_by_month[key]
                equity_by_month[key] = (old_start, end_eq)
        except (ValueError, TypeError):
            continue

    monthly_returns = {}
    for key, (start, end) in equity_by_month.items():
        if start > 0:
            monthly_returns[key] = (end - start) / start * 100

    return monthly_returns


def build_monthly_returns_heatmap(
    trades: list[dict], initial_capital: float = 10_000.0
) -> HeatMap:
    """
    Create a month-by-year heatmap of percentage returns.

    Each cell shows the monthly return (%) for a specific year and month, rounded to two decimals; years appear on the y-axis and months (Jan–Dec) on the x-axis.

    Parameters:
        trades (list[dict]): Trades containing at least the keys 'pnl' and 'exit_time' (ISO timestamp or parseable datetime).

    Returns:
        HeatMap: A pyecharts HeatMap where each data point value is the monthly return percentage (rounded to 2 decimals).
    """
    monthly = _compute_monthly_returns(trades, initial_capital)
    if not monthly:
        return HeatMap()

    years = sorted(set(k[0] for k in monthly))
    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    data = []
    for (yr, mo), ret in monthly.items():
        yi = years.index(yr)
        data.append([mo - 1, yi, round(ret, 2)])

    chart = (
        HeatMap(init_opts=opts.InitOpts(height="400px"))
        .add_xaxis(month_names)
        .add_yaxis(
            series_name="Return",
            yaxis_data=[str(y) for y in years],
            value=data,
            label_opts=opts.LabelOpts(is_show=False),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Monthly Returns Heatmap"),
            visualmap_opts=opts.VisualMapOpts(
                min_=-5,
                max_=10,
                is_calculable=True,
                orient="vertical",
                pos_right="0%",
                pos_top="center",
                range_color=["#DC2626", "#FDE68A", "#059669"],
            ),
            tooltip_opts=opts.TooltipOpts(trigger="item"),
        )
    )
    return chart


def build_rolling_sharpe_chart(
    trades: list[dict],
    window: int = 30,
) -> Line:
    """
    Create a line chart of the rolling annualized Sharpe ratio computed over trade PnL.

    Parameters:
        trades (list[dict]): Ordered list of trade dictionaries containing a numeric 'pnl' key; order determines the rolling sequence.
        window (int): Number of consecutive trades used for each rolling Sharpe calculation.

    Returns:
        pyecharts Line: A Line chart plotting the rolling annualized Sharpe (values rounded to 2 decimals). Returns an empty `Line()` if the number of trades is less than or equal to `window`.
    """
    if len(trades) <= window:
        return Line()

    pnls = np.array([t["pnl"] for t in trades])
    rolling_mean = np.convolve(pnls, np.ones(window) / window, mode="valid")
    rolling_std = np.array(
        [pnls[i : i + window].std() for i in range(len(pnls) - window + 1)]
    )

    try:
        entry = pd.to_datetime(trades[0]["entry_time"])
        exit_ = pd.to_datetime(trades[-1]["exit_time"])
        days = max((exit_ - entry).days, 1)
        trades_per_year = len(trades) / (days / 365.25)
    except Exception:
        trades_per_year = 100  # Fallback

    annualization_factor = np.sqrt(trades_per_year)

    with np.errstate(divide="ignore", invalid="ignore"):
        rolling_sharpe = rolling_mean / rolling_std * annualization_factor
    rolling_sharpe = np.where(rolling_std == 0, np.nan, rolling_sharpe)

    x_labels = [str(i + window) for i in range(len(rolling_sharpe))]

    chart = (
        Line(init_opts=opts.InitOpts(height="400px"))
        .add_xaxis(x_labels)
        .add_yaxis(
            series_name="Rolling Sharpe",
            y_axis=[round(v, 2) for v in rolling_sharpe],
            is_smooth=False,
            linestyle_opts=opts.LineStyleOpts(width=1, color=COLORS["secondary"]),
            label_opts=opts.LabelOpts(is_show=False),
            markline_opts=opts.MarkLineOpts(
                data=[
                    opts.MarkLineItem(
                        y=0,
                        linestyle_opts=opts.LineStyleOpts(color="#333", width=0.5),
                    ),
                    opts.MarkLineItem(
                        y=2,
                        linestyle_opts=opts.LineStyleOpts(
                            color=COLORS["success"], width=1, type_="dashed"
                        ),
                    ),
                ]
            ),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title=f"Rolling Sharpe Ratio (window={window} trades)"
            ),
            xaxis_opts=opts.AxisOpts(name="Trade #"),
            yaxis_opts=opts.AxisOpts(name="Annualized Sharpe"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            datazoom_opts=[opts.DataZoomOpts(type_="inside")],
        )
    )
    return chart


def build_duration_pnl_scatter(trades: list[dict]) -> Scatter:
    """Build trade duration vs PnL scatter plot.

    Points colored by win (green) / loss (red).
    Uses numeric x-axis so points are plotted at exact duration values.

    Args:
        trades: List of trade dicts with 'entry_time', 'exit_time', 'pnl' keys.

    Returns:
        pyecharts Scatter chart.
    """
    win_data: list[list[float]] = []
    loss_data: list[list[float]] = []

    for t in trades:
        try:
            entry = datetime.fromisoformat(str(t["entry_time"]).replace("Z", "+00:00"))
            exit_ = datetime.fromisoformat(str(t["exit_time"]).replace("Z", "+00:00"))
            dur_hours = (exit_ - entry).total_seconds() / 3600
            dur = round(dur_hours, 2)
            pnl = round(t["pnl"], 2)
            if t["pnl"] > 0:
                win_data.append([dur, pnl])
            else:
                loss_data.append([dur, pnl])
        except (ValueError, TypeError):
            continue

    if not win_data and not loss_data:
        return Scatter()

    chart = (
        Scatter(init_opts=opts.InitOpts(height="500px"))
        .add_xaxis([])
        .add_yaxis(
            series_name="Wins",
            y_axis=win_data,
            symbol_size=8,
            label_opts=opts.LabelOpts(is_show=False),
            itemstyle_opts=opts.ItemStyleOpts(color=COLORS["success"]),
        )
        .add_yaxis(
            series_name="Losses",
            y_axis=loss_data,
            symbol_size=8,
            label_opts=opts.LabelOpts(is_show=False),
            itemstyle_opts=opts.ItemStyleOpts(color=COLORS["danger"]),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Trade Duration vs PnL"),
            xaxis_opts=opts.AxisOpts(
                type_="value",
                name="Duration (hours)",
            ),
            yaxis_opts=opts.AxisOpts(name="PnL (USD)"),
            legend_opts=opts.LegendOpts(),
        )
    )
    return chart
