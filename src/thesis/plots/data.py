"""Static matplotlib/seaborn data exploration charts."""

import logging
from pathlib import Path

import numpy as np
import polars as pl

from thesis.config import Config
from thesis.constants import CHART_COLORS as _COLORS
from thesis.constants import EXCLUDE_COLS

logger = logging.getLogger("thesis.visualize")


def _output_dir(config: Config, subdir: str) -> Path:
    """Create and return the output directory for static charts.

    Args:
        config: Application configuration containing session paths.
        subdir: Chart category subdirectory name.

    Returns:
        Absolute output path for the requested chart category.
    """
    base = (
        Path(config.paths.session_dir) if config.paths.session_dir else Path("results")
    )
    d = base / "reports" / "charts" / subdir
    d.mkdir(parents=True, exist_ok=True)
    return d.resolve()


def _plot_candlestick(df: pl.DataFrame, config: Config, out: Path) -> None:
    """Plot OHLC candlestick chart with volume bars for data confirmation.

    Subsamples to ~500 bars for readability. Saves as candlestick.png.

    Args:
        df: OHLCV DataFrame with timestamp, open, high, low, close, volume.
        config: Application configuration.
        out: Output directory for the chart.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle

    # Subsample for readability (max ~500 candles)
    n = len(df)
    if n > 500:
        step = n // 500
        df = df.gather_every(step)
        logger.info(
            "Candlestick: subsampled %d -> %d bars (every %d)", n, len(df), step
        )

    timestamps = df["timestamp"].to_numpy()
    opens = df["open"].to_numpy().astype(float)
    highs = df["high"].to_numpy().astype(float)
    lows = df["low"].to_numpy().astype(float)
    closes = df["close"].to_numpy().astype(float)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 12), height_ratios=[4, 1.5], sharex=True
    )

    # Add spacing between price and volume charts
    fig.subplots_adjust(hspace=0.05)

    # Remove grid for clean candlestick look
    ax1.grid(False)
    ax2.grid(False)
    ax1.set_facecolor("#FAFAFA")
    ax2.set_facecolor("#FAFAFA")

    x = np.arange(len(timestamps))

    # Draw candlestick shadows (wicks) and bodies
    for i in range(len(x)):
        is_bull = closes[i] >= opens[i]
        color = _COLORS["long"] if is_bull else _COLORS["short"]

        # Shadow / wick (high-low line)
        ax1.plot(
            [x[i], x[i]],
            [lows[i], highs[i]],
            color=color,
            linewidth=0.8,
            solid_capstyle="round",
        )

        # Body with shadow
        body_bottom = min(opens[i], closes[i])
        body_height = max(abs(closes[i] - opens[i]), 0.01)

        # Shadow rectangle (offset slightly)
        shadow = Rectangle(
            (x[i] - 0.35 + 0.06, body_bottom - 0.08),
            0.7,
            body_height + 0.16,
            facecolor="#00000015",
            edgecolor="none",
            zorder=0,
        )
        ax1.add_patch(shadow)

        # Main body
        body = FancyBboxPatch(
            (x[i] - 0.35, body_bottom),
            0.7,
            body_height,
            boxstyle="round,pad=0.02",
            facecolor=color if is_bull else color,
            edgecolor=color,
            linewidth=0.5,
            alpha=0.92,
            zorder=1,
        )
        ax1.add_patch(body)

    ax1.set_xlim(-1, len(x))
    ax1.set_ylim(lows.min() * 0.998, highs.max() * 1.002)
    ax1.set_title(f"{config.data.symbol} Candlestick Chart ({config.data.timeframe})")
    ax1.set_ylabel("Price (USD)")

    # Volume bars
    if "volume" in df.columns:
        volumes = df["volume"].to_numpy().astype(float)
        vol_colors = [
            _COLORS["long"] if closes[i] >= opens[i] else _COLORS["short"]
            for i in range(len(x))
        ]
        ax2.bar(x, volumes, color=vol_colors, alpha=0.6, width=0.8)
        ax2.set_ylabel("Volume")
        ax2.set_xlabel("Date")

    # Format x-axis with dates (show ~10 labels)
    n_ticks = min(10, len(timestamps))
    tick_indices = np.linspace(0, len(timestamps) - 1, n_ticks, dtype=int)
    tick_labels = [str(timestamps[i])[:10] for i in tick_indices]  # YYYY-MM-DD

    for ax in (ax1, ax2):
        ax.set_xticks(tick_indices)
        ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=8)

    fig.savefig(out / "candlestick.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart: candlestick.png")


def _generate_data_charts(config: Config) -> None:
    """Generate static data-exploration charts for report assets.

    Args:
        config: Application configuration containing input/output paths.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = _output_dir(config, "data")

    features_path = Path(config.paths.features)
    labels_path = Path(config.paths.labels)
    ohlcv_path = Path(config.paths.ohlcv)

    # --- 1. Candlestick Chart with Volume ---
    if ohlcv_path.exists():
        df = pl.read_parquet(ohlcv_path)
        if len(df) > 0:
            _plot_candlestick(df, config, out)

    # --- 2. Label Distribution ---
    if labels_path.exists():
        df = pl.read_parquet(labels_path)
        if "label" in df.columns:
            labels = df["label"].to_numpy()
            counts = {k: int((labels == k).sum()) for k in [-1, 0, 1]}
            total = sum(counts.values())

            fig, ax = plt.subplots(figsize=(8, 6))
            names = [
                f"Short (-1)\n{counts[-1]} ({counts[-1] / total * 100:.1f}%)",
                f"Flat (0)\n{counts[0]} ({counts[0] / total * 100:.1f}%)",
                f"Long (1)\n{counts[1]} ({counts[1] / total * 100:.1f}%)",
            ]
            colors = [_COLORS["short"], _COLORS["flat"], _COLORS["long"]]
            ax.pie(
                [counts[-1], counts[0], counts[1]],
                labels=names,
                colors=colors,
                autopct="",
                startangle=90,
                textprops={"fontsize": 11},
            )
            ax.set_title("Triple-Barrier Label Distribution")
            fig.savefig(out / "label_distribution.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("Chart: label_distribution.png")

    # --- 3. Feature Correlation Heatmap ---
    if features_path.exists():
        df = pl.read_parquet(features_path)
        feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]

        if len(feature_cols) > 1:
            numeric_df = df.select(feature_cols)
            corr = numeric_df.corr().to_numpy()

            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(len(feature_cols)))
            ax.set_yticks(range(len(feature_cols)))
            ax.set_xticklabels(feature_cols, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(feature_cols, fontsize=8)
            ax.tick_params(length=0)  # remove tick marks
            ax.grid(False)
            fig.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title("Feature Correlation Matrix")
            # Annotate cells with correlation values
            for i in range(len(feature_cols)):
                for j in range(len(feature_cols)):
                    val = corr[i, j]
                    color = "white" if abs(val) > 0.5 else "black"
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color=color,
                    )
            fig.savefig(
                out / "feature_correlation.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)
            logger.info("Chart: feature_correlation.png")

    # --- 4. Feature Distributions ---
    if features_path.exists():
        df = pl.read_parquet(features_path)
        feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]

        if feature_cols:
            n = len(feature_cols)
            ncols = min(3, n)
            nrows = (n + ncols - 1) // ncols

            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows))
            if nrows == 1:
                axes = [axes] if ncols == 1 else axes
            else:
                axes = axes.flatten()

            for i, col in enumerate(feature_cols):
                if i < len(axes):
                    vals = df[col].drop_nulls().to_numpy()
                    axes[i].hist(
                        vals,
                        bins=50,
                        color=_COLORS["primary"],
                        alpha=0.7,
                        edgecolor="white",
                    )
                    axes[i].set_title(col, fontsize=10)
                    axes[i].tick_params(labelsize=8)

            # Hide unused axes
            for j in range(len(feature_cols), len(axes)):
                axes[j].set_visible(False)

            fig.suptitle("Feature Distributions", fontsize=13, y=1.01)
            fig.tight_layout()
            fig.savefig(out / "feature_distributions.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("Chart: feature_distributions.png")
