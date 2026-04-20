"""Data exploration interactive ECharts: candlestick, correlation, labels, features."""

import logging

import numpy as np
import polars as pl
from pyecharts import options as opts
from pyecharts.charts import Bar, Grid, HeatMap, Kline, Pie, Tab

from thesis.config import Config

from .data import COLORS, _get_feature_cols

logger = logging.getLogger("thesis.charts")


def _downsample_ohlcv(df: pl.DataFrame, max_bars: int) -> pl.DataFrame:
    """
    Reduce an OHLCV DataFrame to at most `max_bars` rows by aggregating contiguous rows into fixed-size groups.

    Parameters:
        df (pl.DataFrame): Input OHLCV time series with columns `timestamp`, `open`, `high`, `low`, `close`, and optionally `volume`.
        max_bars (int): Maximum number of bars to retain after downsampling.

    Returns:
        pl.DataFrame: Aggregated OHLCV DataFrame with at most `max_bars` rows. For each group:
            - `timestamp`: first timestamp in the group
            - `open`: first open price in the group
            - `high`: maximum high price in the group
            - `low`: minimum low price in the group
            - `close`: last close price in the group
            - `volume` (if present): sum of volumes in the group
    """
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
    config: Config,
    max_bars: int = 3000,
) -> tuple[Grid, dict]:
    """
    Build an interactive OHLCV candlestick chart with stacked volume and time zoom controls.

    Expects `df` to contain columns: `timestamp`, `open`, `high`, `low`, `close`; `volume` is optional.
    The chart is laid out as price (top) and volume (bottom) with a visible slider and inside data zoom.
    Downsamples `df` when its row count exceeds `max_bars` to limit rendered bars.

    Parameters:
        df (pl.DataFrame): OHLCV data. `timestamp` may be temporal or UTF-8 strings.
        config (Config): Application configuration used for chart title (expects `config.data.symbol` and `config.data.timeframe`).
        max_bars (int): Maximum number of bars to render before downsampling.

    Returns:
        tuple[Grid, dict]: A tuple containing the pyecharts `Grid` chart and an info dict with keys:
            - `total_bars`: original number of rows in `df`
            - `displayed_bars`: number of bars actually rendered (after downsampling)
            - `downsampled`: `true` if downsampling was applied, `false` otherwise
    """
    total_bars = len(df)
    if total_bars > max_bars:
        df = _downsample_ohlcv(df, max_bars)
        downsampled = True
        logger.info(
            "Candlestick: downsampled %d -> %d bars (stride=%d)",
            total_bars,
            len(df),
            max(1, total_bars // max_bars),
        )
    else:
        downsampled = False

    n = len(df)
    logger.info("Candlestick: rendering %d bars", n)

    # Format timestamps — detect whether data has intraday time

    ts_col = df["timestamp"]
    if ts_col.dtype == pl.Utf8:
        ts_col = ts_col.str.to_datetime()
    if ts_col.dtype.is_temporal():
        has_intraday = (ts_col.dt.hour().sum() + ts_col.dt.minute().sum()) > 0
        fmt = "%Y-%m-%d %H:%M" if has_intraday else "%Y-%m-%d"
        timestamps = ts_col.dt.strftime(fmt).to_list()
    else:
        timestamps = ts_col.cast(pl.Utf8).to_list()

    # ECharts candlestick format: [open, close, low, high]
    opens = df["open"].to_numpy().astype(float)
    closes = df["close"].to_numpy().astype(float)
    lows = df["low"].to_numpy().astype(float)
    highs = df["high"].to_numpy().astype(float)
    kline_data = [
        [float(o), float(c), float(lo), float(hi)]
        for o, c, lo, hi in zip(opens, closes, lows, highs, strict=True)
    ]

    # Volume with color
    volumes = df["volume"].to_numpy().astype(float) if "volume" in df.columns else None

    kline = (
        Kline()
        .add_xaxis(timestamps)
        .add_yaxis(
            series_name="Price",
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
            ),
            legend_opts=opts.LegendOpts(pos_top="1%", pos_right="10%"),
            yaxis_opts=opts.AxisOpts(
                is_scale=True, splitarea_opts=opts.SplitAreaOpts(is_show=False)
            ),
            xaxis_opts=opts.AxisOpts(is_show=False),
            tooltip_opts=opts.TooltipOpts(
                trigger="axis",
                axis_pointer_type="cross",
            ),
            datazoom_opts=[
                opts.DataZoomOpts(
                    is_show=True,
                    type_="slider",
                    xaxis_index=[0, 1],
                    pos_bottom="4%",
                    height=28,
                ),
                opts.DataZoomOpts(
                    type_="inside",
                    xaxis_index=[0, 1],
                ),
            ],
        )
    )

    # Volume bar chart - split into up/down volumes using stack
    if volumes is not None:
        up_volumes = [
            float(v) if closes[i] >= opens[i] else 0 for i, v in enumerate(volumes)
        ]
        down_volumes = [
            float(v) if closes[i] < opens[i] else 0 for i, v in enumerate(volumes)
        ]
        bar = (
            Bar()
            .add_xaxis(timestamps)
            .add_yaxis(
                series_name="Up Volume",
                y_axis=up_volumes,
                stack="volume",
                itemstyle_opts=opts.ItemStyleOpts(color=COLORS["long"]),
                label_opts=opts.LabelOpts(is_show=False),
            )
            .add_yaxis(
                series_name="Down Volume",
                y_axis=down_volumes,
                stack="volume",
                itemstyle_opts=opts.ItemStyleOpts(color=COLORS["short"]),
                label_opts=opts.LabelOpts(is_show=False),
            )
            .set_global_opts(
                yaxis_opts=opts.AxisOpts(
                    splitarea_opts=opts.SplitAreaOpts(is_show=False),
                ),
                xaxis_opts=opts.AxisOpts(
                    is_show=True,
                    axislabel_opts=opts.LabelOpts(is_show=True, font_size=10, margin=8),
                ),
                legend_opts=opts.LegendOpts(is_show=False),
            )
        )
    else:
        bar = Bar()

    grid = (
        Grid(init_opts=opts.InitOpts(height="700px"))
        .add(
            kline,
            grid_opts=opts.GridOpts(pos_top="5%", pos_bottom="25%"),
        )
        .add(
            bar,
            grid_opts=opts.GridOpts(pos_top="72%", pos_bottom="12%"),
        )
    )

    info = {
        "total_bars": total_bars,
        "displayed_bars": n,
        "downsampled": downsampled,
    }
    return grid, info


def build_correlation_heatmap(df: pl.DataFrame) -> HeatMap:
    """
    Builds a correlation heatmap for numeric feature columns.

    Selects numeric feature columns (falls back to numeric dtypes if fewer than two detected features), computes pairwise Pearson correlations, and renders them as a pyecharts HeatMap using a blue-white-red diverging colormap mapped to the range -1..1.

    Parameters:
        df (pl.DataFrame): DataFrame containing timestamp and feature columns.

    Returns:
        HeatMap: pyecharts HeatMap chart of the correlation matrix with values rounded to three decimals.
    """
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

    # Flatten to [x, y, value] for pyecharts HeatMap
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
    """Build triple-barrier label distribution pie chart.

    Args:
        df: Labels DataFrame with 'label' column.

    Returns:
        pyecharts Pie chart.
    """
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
    """
    Build a tabbed chart of per-feature 50-bin histograms.

    Each tab contains a bar chart showing the histogram counts for one feature; features with no non-null values are skipped. Bin labels use bin centers formatted to two decimal places.

    Parameters:
        df (pl.DataFrame): Polars DataFrame containing feature columns to plot.

    Returns:
        Tab: A pyecharts Tab where each tab is a Bar chart of a feature's histogram.
    """
    feature_cols = _get_feature_cols(df)
    tab = Tab()

    for col in feature_cols:
        vals = df[col].drop_nulls().to_numpy()
        if len(vals) == 0:
            continue

        counts, bin_edges = np.histogram(vals, bins=50)
        # Use bin centers as labels
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
