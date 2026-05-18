"""Data exploration charts (pyecharts)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from pyecharts import options as opts
from pyecharts.charts import Bar, Grid, HeatMap, Kline

from thesis.visualization.chart_data import (
    make_kline_series,
    make_volume_series,
    parse_chart_timestamps,
    prepare_candlestick_data,
)
from thesis.visualization.style import COLORS, EXCLUDED_FEATURE_COLS

if TYPE_CHECKING:
    from thesis.shared.config import Config


def _get_feature_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDED_FEATURE_COLS]


def _kline_global_opts(config: Config | None) -> opts.GlobalOpts:
    return dict(
        title_opts=opts.TitleOpts(
            title=f"{config.data.symbol} Candlestick ({config.data.timeframe})"
            if config
            else "Candlestick Chart"
        ),
        legend_opts=opts.LegendOpts(is_show=False, pos_bottom=10, pos_left="center"),
        yaxis_opts=opts.AxisOpts(
            is_scale=True,
            splitarea_opts=opts.SplitAreaOpts(is_show=False),
            splitline_opts=opts.SplitLineOpts(is_show=False),
        ),
        xaxis_opts=opts.AxisOpts(is_show=False),
        tooltip_opts=opts.TooltipOpts(
            trigger="axis",
            axis_pointer_type="cross",
            background_color="rgba(245, 245, 245, 0.8)",
            border_width=1,
            border_color="#ccc",
            textstyle_opts=opts.TextStyleOpts(color="#000"),
        ),
        datazoom_opts=[
            opts.DataZoomOpts(
                is_show=False,
                type_="inside",
                xaxis_index=[0, 1],
                range_start=50,
                range_end=100,
            ),
            opts.DataZoomOpts(
                is_show=True,
                xaxis_index=[0, 1],
                type_="slider",
                pos_top="85%",
                range_start=50,
                range_end=100,
            ),
        ],
        axispointer_opts=opts.AxisPointerOpts(
            is_show=True,
            link=[{"xAxisIndex": "all"}],
            label=opts.LabelOpts(background_color="#777"),
        ),
        brush_opts=opts.BrushOpts(
            x_axis_index="all",
            brush_link="all",
            out_of_brush={"colorAlpha": 0.1},
            brush_type="lineX",
        ),
    )


def build_price_kline(
    kline_data: list[list[float]],
    timestamps: list[str],
    config: Config | None,
) -> Kline:
    return (
        Kline()
        .add_xaxis(xaxis_data=timestamps)
        .add_yaxis(
            series_name=f"{config.data.symbol}" if config else "XAUUSD",
            y_axis=kline_data,
            itemstyle_opts=opts.ItemStyleOpts(
                color=COLORS["long"],
                color0=COLORS["short"],
                border_color=COLORS["long"],
                border_color0=COLORS["short"],
            ),
        )
        .set_global_opts(**_kline_global_opts(config))
    )


def build_volume_bar(
    volume_data: list[list[float]],
    timestamps: list[str],
) -> Bar:
    return (
        Bar()
        .add_xaxis(xaxis_data=timestamps)
        .add_yaxis(
            series_name="Volume",
            y_axis=volume_data,
            xaxis_index=1,
            yaxis_index=1,
            label_opts=opts.LabelOpts(is_show=False),
        )
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(
                type_="category",
                is_scale=True,
                grid_index=1,
                boundary_gap=False,
                axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                axistick_opts=opts.AxisTickOpts(is_show=False),
                splitline_opts=opts.SplitLineOpts(is_show=False),
                axislabel_opts=opts.LabelOpts(is_show=False),
                split_number=20,
                min_="dataMin",
                max_="dataMax",
            ),
            yaxis_opts=opts.AxisOpts(
                grid_index=1,
                is_scale=True,
                split_number=2,
                axislabel_opts=opts.LabelOpts(is_show=False),
                axisline_opts=opts.AxisLineOpts(is_show=False),
                axistick_opts=opts.AxisTickOpts(is_show=False),
                splitline_opts=opts.SplitLineOpts(is_show=False),
            ),
            legend_opts=opts.LegendOpts(is_show=False),
        )
    )


def compose_price_volume_grid(price_chart: Kline, volume_chart: Bar) -> Grid:
    return (
        Grid(
            init_opts=opts.InitOpts(
                width="1000px",
                height="800px",
                animation_opts=opts.AnimationOpts(animation=False),
            )
        )
        .add(
            price_chart,
            grid_opts=opts.GridOpts(pos_left="10%", pos_right="8%", height="50%"),
        )
        .add(
            volume_chart,
            grid_opts=opts.GridOpts(
                pos_left="10%", pos_right="8%", pos_top="63%", height="16%"
            ),
        )
    )


def build_candlestick_chart(
    df: pl.DataFrame,
    config: Config | None = None,
    max_bars: int = 3000,
) -> tuple[Grid, dict]:
    """OHLCV candlestick + volume chart."""
    sampled, info = prepare_candlestick_data(df, max_bars)
    timestamps = parse_chart_timestamps(sampled["timestamp"])
    kline_data = make_kline_series(sampled)
    volume_data = make_volume_series(sampled)

    price_chart = build_price_kline(kline_data, timestamps, config)
    volume_chart = (
        build_volume_bar(volume_data, timestamps) if volume_data is not None else Bar()
    )
    return compose_price_volume_grid(price_chart, volume_chart), info


def build_correlation_heatmap(df: pl.DataFrame) -> HeatMap:
    """Correlation heatmap for feature columns."""
    feature_cols = _get_feature_cols(df)
    if len(feature_cols) < 2:
        feature_cols = [
            c
            for c in df.columns
            if df[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
        ]

    numeric_df = df.select(feature_cols)
    corr = numeric_df.corr().to_numpy()
    n = len(feature_cols)

    data = []
    for i in range(n):
        for j in range(n):
            data.append([j, i, round(float(corr[i, j]), 3)])

    return (
        HeatMap(init_opts=opts.InitOpts(height="600px"))
        .add_xaxis(feature_cols)
        .add_yaxis(
            series_name="Correlation",
            yaxis_data=feature_cols,
            value=data,
            label_opts=opts.LabelOpts(is_show=False),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Feature Correlation Matrix"),
            visualmap_opts=opts.VisualMapOpts(
                min_=-1,
                max_=1,
                is_calculable=True,
                orient="vertical",
                pos_right="0%",
                pos_top="center",
                range_color=["#DC2626", "#FFFFFF", "#2563EB"],
            ),
            xaxis_opts=opts.AxisOpts(
                axislabel_opts=opts.LabelOpts(rotate=45, font_size=8),
            ),
            yaxis_opts=opts.AxisOpts(
                axislabel_opts=opts.LabelOpts(font_size=8),
            ),
            tooltip_opts=opts.TooltipOpts(trigger="item"),
        )
    )


def build_label_distribution_chart(df: pl.DataFrame) -> Bar:
    """Bar chart for triple-barrier label distribution."""
    labels = df["label"].to_numpy()
    counts = {k: int((labels == k).sum()) for k in [-1, 0, 1]}
    total = max(sum(counts.values()), 1)

    names = ["Short (-1)", "Hold (0)", "Long (1)"]
    values = [counts.get(k, 0) for k in [-1, 0, 1]]
    percentages = [round(v / total * 100, 2) for v in values]

    return (
        Bar(init_opts=opts.InitOpts(height="500px"))
        .add_xaxis(names)
        .add_yaxis(
            series_name="Count",
            y_axis=values,
            label_opts=opts.LabelOpts(is_show=True, position="top"),
            itemstyle_opts=opts.ItemStyleOpts(color=COLORS["primary"]),
        )
        .add_yaxis(
            series_name="Percent",
            y_axis=percentages,
            label_opts=opts.LabelOpts(is_show=True, position="top", formatter="{c}%"),
            itemstyle_opts=opts.ItemStyleOpts(color=COLORS["secondary"]),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Triple-Barrier Label Distribution"),
            xaxis_opts=opts.AxisOpts(name="Label"),
            yaxis_opts=opts.AxisOpts(name="Count / Percent"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            legend_opts=opts.LegendOpts(),
        )
    )


def build_feature_distribution_chart(df: pl.DataFrame, feature_col: str) -> Bar:
    """Histogram for one feature column."""
    if feature_col not in df.columns:
        return Bar()

    vals = df[feature_col].drop_nulls().to_numpy()
    if vals.size == 0:
        return Bar()

    counts, bin_edges = np.histogram(vals, bins=50)
    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(counts))]

    return (
        Bar(init_opts=opts.InitOpts(height="420px"))
        .add_xaxis([f"{v:.2f}" for v in bin_centers])
        .add_yaxis(
            series_name=feature_col,
            y_axis=counts.tolist(),
            label_opts=opts.LabelOpts(is_show=False),
            itemstyle_opts=opts.ItemStyleOpts(color=COLORS["primary"]),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title=f"Distribution: {feature_col}"),
            xaxis_opts=opts.AxisOpts(name=feature_col),
            yaxis_opts=opts.AxisOpts(name="Count"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            datazoom_opts=[opts.DataZoomOpts(type_="inside")],
        )
    )


__all__ = [
    "_get_feature_cols",
    "build_candlestick_chart",
    "build_correlation_heatmap",
    "build_feature_distribution_chart",
    "build_label_distribution_chart",
]
