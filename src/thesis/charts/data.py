"""Data exploration charts."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from pyecharts import options as opts
from pyecharts.charts import Bar, HeatMap, Kline, Pie, Tab

from thesis.charts.shared import COLORS, EXCLUDED_FEATURE_COLS

if TYPE_CHECKING:
    from thesis.shared.config import Config

logger = logging.getLogger("thesis.charts")


def _get_feature_cols(df: pl.DataFrame) -> list[str]:
    """Feature cols — exclude metadata."""
    return [c for c in df.columns if c not in EXCLUDED_FEATURE_COLS]


def _downsample_ohlcv(df: pl.DataFrame, max_bars: int) -> pl.DataFrame:
    """Downsample OHLCV to max_bars rows."""
    stride = max(1, len(df) // max_bars)
    group_col = pl.int_range(0, len(df)) // stride
    agg_exprs = [
        pl.col("timestamp").first(),
        pl.col("open").first(),
        pl.col("high").max(),
        pl.col("low").min(),
        pl.col("close").last(),
    ]
    if "volume" in df.columns:
        agg_exprs.append(pl.col("volume").sum())
    return (
        df.with_columns(group_col.alias("_group"))
        .group_by("_group", maintain_order=True)
        .agg(*agg_exprs)
        .drop("_group")
    )


def build_candlestick_chart(
    df: pl.DataFrame,
    config: Config | None = None,
    max_bars: int = 3000,
) -> tuple[Kline, dict]:
    """OHLCV candlestick + volume chart."""
    total_bars = len(df)
    if total_bars > max_bars:
        df = _downsample_ohlcv(df, max_bars)
        downsampled = True
        logger.info(
            "Candlestick: downsampled %d -> %d bars",
            total_bars,
            len(df),
        )
    else:
        downsampled = False

    n = len(df)
    logger.info("Candlestick: rendering %d bars", n)

    ts_col = df["timestamp"]
    if ts_col.dtype == pl.Utf8:
        ts_col = ts_col.str.to_datetime()
    if ts_col.dtype.is_temporal():
        has_intraday = (ts_col.dt.hour().sum() + ts_col.dt.minute().sum()) > 0
        fmt = "%Y-%m-%d %H:%M" if has_intraday else "%Y-%m-%d"
        timestamps = ts_col.dt.strftime(fmt).to_list()
    else:
        timestamps = ts_col.cast(pl.Utf8).to_list()

    opens = df["open"].to_numpy().astype(float)
    closes = df["close"].to_numpy().astype(float)
    lows = df["low"].to_numpy().astype(float)
    highs = df["high"].to_numpy().astype(float)
    kline_data = [
        [float(o), float(c), float(lo), float(hi)]
        for o, c, lo, hi in zip(opens, closes, lows, highs, strict=True)
    ]

    volumes = df["volume"].to_numpy().astype(float) if "volume" in df.columns else None

    kline = (
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
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title=f"{config.data.symbol} Candlestick ({config.data.timeframe})"
                if config
                else "Candlestick Chart"
            ),
            legend_opts=opts.LegendOpts(
                is_show=False, pos_bottom=10, pos_left="center"
            ),
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
    )

    if volumes is not None:
        volume_data = [
            [i, float(volumes[i]), 1 if closes[i] >= opens[i] else -1]
            for i in range(len(volumes))
        ]
        bar = (
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
    else:
        bar = Bar()

    from pyecharts.charts import Grid

    grid = (
        Grid(
            init_opts=opts.InitOpts(
                width="1000px",
                height="800px",
                animation_opts=opts.AnimationOpts(animation=False),
            )
        )
        .add(
            kline,
            grid_opts=opts.GridOpts(pos_left="10%", pos_right="8%", height="50%"),
        )
        .add(
            bar,
            grid_opts=opts.GridOpts(
                pos_left="10%", pos_right="8%", pos_top="63%", height="16%"
            ),
        )
    )

    info = {
        "total_bars": total_bars,
        "displayed_bars": n,
        "downsampled": downsampled,
    }
    return grid, info


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

    chart = (
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
    return chart


def build_label_distribution_chart(df: pl.DataFrame) -> Pie:
    """Pie chart for triple-barrier label distribution."""
    labels = df["label"].to_numpy()
    counts = {k: int((labels == k).sum()) for k in [-1, 0, 1]}

    data_pairs = [
        ("Short (-1)", counts.get(-1, 0)),
        ("Hold (0)", counts.get(0, 0)),
        ("Long (1)", counts.get(1, 0)),
    ]

    chart = (
        Pie(init_opts=opts.InitOpts(height="500px"))
        .add(
            series_name="Labels",
            data_pair=data_pairs,
            radius="75%",
            label_opts=opts.LabelOpts(
                formatter="{name|{b}}\n{count|{c}} {per|{d}%}",
                position="outside",
                rich={
                    "name": {
                        "fontSize": 14,
                        "fontWeight": "bold",
                        "padding": [0, 0, 4, 0],
                    },
                    "count": {
                        "fontSize": 12,
                        "color": "#666",
                    },
                    "per": {
                        "fontSize": 12,
                        "fontWeight": "bold",
                        "color": "#333",
                    },
                },
            ),
        )
        .set_colors([COLORS["short"], COLORS["flat"], COLORS["long"]])
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Triple-Barrier Label Distribution"),
            legend_opts=opts.LegendOpts(pos_left="left", orient="vertical"),
            tooltip_opts=opts.TooltipOpts(trigger="item", formatter="{b}: {c} ({d}%)"),
        )
    )
    return chart


def build_feature_distributions_chart(df: pl.DataFrame) -> Tab:
    """Tabbed histograms for feature distributions."""
    feature_cols = _get_feature_cols(df)
    tab = Tab()

    for col in feature_cols:
        vals = df[col].drop_nulls().to_numpy()
        if vals.size == 0:
            continue

        counts, bin_edges = np.histogram(vals, bins=50)
        bin_centers = [
            (bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(counts))
        ]
        x_labels = [f"{v:.2f}" for v in bin_centers]

        bar = (
            Bar(init_opts=opts.InitOpts(height="400px"))
            .add_xaxis(x_labels)
            .add_yaxis(
                series_name=col,
                y_axis=counts.tolist(),
                label_opts=opts.LabelOpts(is_show=False),
                itemstyle_opts=opts.ItemStyleOpts(color=COLORS["primary"]),
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title=f"Distribution: {col}"),
                xaxis_opts=opts.AxisOpts(name=col),
                yaxis_opts=opts.AxisOpts(name="Count"),
                tooltip_opts=opts.TooltipOpts(trigger="axis"),
                datazoom_opts=[opts.DataZoomOpts(type_="inside")],
            )
        )
        tab.add(bar, col)

    return tab


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
    "build_candlestick_chart",
    "build_correlation_heatmap",
    "build_label_distribution_chart",
    "build_feature_distribution_chart",
    "build_feature_distributions_chart",
    "_get_feature_cols",
]
