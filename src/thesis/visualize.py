"""Comprehensive thesis visualization — data exploration, model performance, backtest.

Generates all charts needed for the thesis report in one pass.

Charts generated:
    Data Exploration (charts/data/):
        1. Price series with labels overlay
        2. Feature correlation heatmap
        3. Label distribution pie chart
        4. Feature distribution histograms

    Model Performance (charts/model/):
        5. GRU training loss curves
        6. Confusion matrix (test set)
        7. Prediction confidence distribution
        8. Feature importance bar chart (top 20)
        9. SHAP summary plot (if available)

    Backtest (charts/backtest/):
        10. Equity curve with drawdown
        11. Trade PnL histogram
        12. Monthly returns heatmap
        13. Rolling Sharpe ratio

Usage:
    from thesis.visualize import generate_all_charts
    generate_all_charts(config)
"""

import json
import logging
from pathlib import Path

import numpy as np
import polars as pl

from thesis.config import Config

logger = logging.getLogger("thesis.visualize")

# Professional color palette
_COLORS = {
    "primary": "#2563EB",
    "secondary": "#7C3AED",
    "success": "#059669",
    "danger": "#DC2626",
    "warning": "#D97706",
    "gray": "#6B7280",
    "long": "#059669",
    "short": "#DC2626",
    "flat": "#6B7280",
}

_SESSION_SESSION_DIR: str = ""


def _output_dir(config: Config, subdir: str) -> Path:
    """Get output directory for chart category."""
    base = (
        Path(config.paths.session_dir) if config.paths.session_dir else Path("results")
    )
    d = base / "reports" / "charts" / subdir
    d.mkdir(parents=True, exist_ok=True)
    return d


def generate_all_charts(config: Config) -> None:
    """Generate all thesis visualization charts.

    Args:
        config: Loaded application configuration.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        }
    )

    logger.info("Generating all thesis charts...")

    _generate_data_charts(config)
    _generate_model_charts(config)
    # Backtest charts are now handled by backtesting.py Bokeh HTML output.
    # See: {session_dir}/backtest/backtest_chart.html

    logger.info("All charts generated.")


# ---------------------------------------------------------------------------
# Data Exploration Charts
# ---------------------------------------------------------------------------


def _generate_data_charts(config: Config) -> None:
    """Generate data exploration charts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = _output_dir(config, "data")

    features_path = Path(config.paths.features)
    labels_path = Path(config.paths.labels)
    ohlcv_path = Path(config.paths.ohlcv)

    # --- 1. Price Series with Volume ---
    if ohlcv_path.exists():
        df = pl.read_parquet(ohlcv_path)
        if len(df) > 0:
            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(14, 8), height_ratios=[3, 1], sharex=True
            )

            timestamps = df["timestamp"].to_list()
            ax1.plot(
                timestamps,
                df["close"].to_numpy(),
                color=_COLORS["primary"],
                linewidth=0.8,
            )
            ax1.set_title(f"{config.data.symbol} Close Price ({config.data.timeframe})")
            ax1.set_ylabel("Price (USD)")
            ax1.grid(True, alpha=0.3)

            if "volume" in df.columns:
                ax2.bar(
                    timestamps,
                    df["volume"].to_numpy(),
                    color=_COLORS["gray"],
                    alpha=0.5,
                    width=1,
                )
                ax2.set_ylabel("Volume")
                ax2.set_xlabel("Date")

            fig.savefig(out / "price_series.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("Chart: price_series.png")

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
        feature_cols = [
            c
            for c in df.columns
            if c
            not in {
                "timestamp",
                "label",
                "tp_price",
                "sl_price",
                "touched_bar",
                "open_right",
                "high_right",
                "low_right",
                "close_right",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "avg_spread",
                "tick_count",
                "dead_hour",
                "log_returns",
            }
        ]

        if len(feature_cols) > 1:
            # Select only numeric features for correlation
            numeric_df = df.select(feature_cols)
            corr = numeric_df.corr().to_numpy()

            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(len(feature_cols)))
            ax.set_yticks(range(len(feature_cols)))
            ax.set_xticklabels(feature_cols, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(feature_cols, fontsize=8)
            fig.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title("Feature Correlation Matrix")
            fig.savefig(out / "feature_correlation.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("Chart: feature_correlation.png")

    # --- 4. Feature Distributions ---
    if features_path.exists():
        df = pl.read_parquet(features_path)
        feature_cols = [
            c
            for c in df.columns
            if c
            not in {
                "timestamp",
                "label",
                "tp_price",
                "sl_price",
                "touched_bar",
                "open_right",
                "high_right",
                "low_right",
                "close_right",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "avg_spread",
                "tick_count",
                "dead_hour",
                "log_returns",
            }
        ]

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


# ---------------------------------------------------------------------------
# Model Performance Charts
# ---------------------------------------------------------------------------


def _generate_model_charts(config: Config) -> None:
    """Generate model performance charts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    out = _output_dir(config, "model")

    # Load predictions
    preds_path = Path(config.paths.predictions)
    if not preds_path.exists():
        logger.warning("Predictions not found: %s", preds_path)
        return

    preds_df = pl.read_parquet(preds_path)
    y_true = preds_df["true_label"].to_numpy()
    y_pred = preds_df["pred_label"].to_numpy()

    # --- 1. Confusion Matrix ---
    fig, ax = plt.subplots(figsize=(8, 6))
    labels_order = [-1, 0, 1]
    display_labels = ["Short (-1)", "Flat (0)", "Long (1)"]
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=labels_order,
        display_labels=display_labels,
        cmap="Blues",
        ax=ax,
        normalize="true",
    )
    ax.set_title("Normalized Confusion Matrix (Test Set)")
    fig.savefig(out / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart: confusion_matrix.png")

    # --- 2. Prediction Confidence Distribution ---
    if "pred_proba_class_1" in preds_df.columns:
        long_conf = preds_df["pred_proba_class_1"].to_numpy()
        short_conf = preds_df["pred_proba_class_minus1"].to_numpy()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(
            long_conf[y_pred == 1],
            bins=50,
            alpha=0.6,
            color=_COLORS["long"],
            label="Long confidence",
        )
        ax.hist(
            short_conf[y_pred == -1],
            bins=50,
            alpha=0.6,
            color=_COLORS["short"],
            label="Short confidence",
        )
        ax.set_title("Prediction Confidence Distribution")
        ax.set_xlabel("Confidence (max softmax probability)")
        ax.set_ylabel("Count")
        ax.legend()
        fig.savefig(out / "confidence_distribution.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Chart: confidence_distribution.png")

    # --- 3. Feature Importance ---
    if config.paths.session_dir:
        fi_path = Path(config.paths.session_dir) / "reports" / "feature_importance.json"
    else:
        fi_path = Path("results/feature_importance.json")

    if fi_path.exists():
        with open(fi_path) as f:
            fi = json.load(f)

        # Separate GRU and static features
        top_n = 20
        items = list(fi.items())[:top_n]
        names = [n for n, _ in items]
        values = [v for _, v in items]
        colors = [
            _COLORS["secondary"] if n.startswith("gru_") else _COLORS["primary"]
            for n in names
        ]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(names, values, color=colors)
        ax.set_title(f"Feature Importance (Top {top_n})")
        ax.invert_yaxis()

        # Legend for feature types
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor=_COLORS["secondary"], label="GRU hidden state"),
            Patch(facecolor=_COLORS["primary"], label="Static feature"),
        ]
        ax.legend(handles=legend_elements, loc="lower right")

        fig.savefig(out / "feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Chart: feature_importance.png")


# ---------------------------------------------------------------------------
# Backtest Charts
# ---------------------------------------------------------------------------


def _generate_backtest_charts(config: Config) -> None:
    """Generate backtest analysis charts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from datetime import datetime

    out = _output_dir(config, "backtest")

    bt_path = Path(config.paths.backtest_results)
    if not bt_path.exists():
        logger.warning("Backtest results not found: %s", bt_path)
        return

    with open(bt_path) as f:
        bt = json.load(f)

    trades = bt.get("trades", [])
    metrics = bt.get("metrics", {})

    if not trades:
        logger.warning("No trades found in backtest results")
        return

    # --- 1. Equity Curve with Drawdown ---
    pnls = [t["pnl"] for t in trades]
    equity = [config.backtest.initial_capital]
    for p in pnls:
        equity.append(equity[-1] + p)

    equity_arr = np.array(equity)
    peak = np.maximum.accumulate(equity_arr)
    drawdown_pct = (equity_arr - peak) / peak * 100

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), height_ratios=[3, 1], sharex=True
    )

    # Equity curve
    ax1.plot(equity, color=_COLORS["primary"], linewidth=1)
    ax1.fill_between(
        range(len(equity)), peak, equity_arr, alpha=0.2, color=_COLORS["danger"]
    )
    ax1.set_title(
        f"Equity Curve — {metrics.get('total_trades', 0)} trades, "
        f"{metrics.get('total_return_pct', 0):.1f}% return"
    )
    ax1.set_ylabel("Equity (USD)")
    ax1.grid(True, alpha=0.3)

    # Drawdown
    ax2.fill_between(
        range(len(drawdown_pct)), drawdown_pct, 0, color=_COLORS["danger"], alpha=0.5
    )
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Trade #")
    ax2.grid(True, alpha=0.3)

    fig.savefig(out / "equity_drawdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart: equity_drawdown.png")

    # --- 2. Trade PnL Histogram ---
    fig, ax = plt.subplots(figsize=(10, 5))
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    ax.hist(
        wins, bins=50, alpha=0.7, color=_COLORS["success"], label=f"Wins ({len(wins)})"
    )
    ax.hist(
        losses,
        bins=50,
        alpha=0.7,
        color=_COLORS["danger"],
        label=f"Losses ({len(losses)})",
    )
    ax.axvline(x=0, color="black", linewidth=1, linestyle="-")
    ax.set_title(
        f"Trade PnL Distribution — Avg Win: ${metrics.get('avg_win', 0):.0f}, "
        f"Avg Loss: ${metrics.get('avg_loss', 0):.0f}"
    )
    ax.set_xlabel("PnL (USD)")
    ax.set_ylabel("Count")
    ax.legend()
    fig.savefig(out / "pnl_histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart: pnl_histogram.png")

    # --- 3. Monthly Returns Heatmap ---
    try:
        monthly = _compute_monthly_returns(trades)
        if monthly:
            years = sorted(set(k[0] for k in monthly.keys()))
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

            data = np.full((len(years), 12), np.nan)
            for (yr, mo), ret in monthly.items():
                yi = years.index(yr)
                data[yi, mo - 1] = ret

            fig, ax = plt.subplots(figsize=(12, max(3, len(years) * 0.8)))
            im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-5, vmax=10)

            ax.set_xticks(range(12))
            ax.set_xticklabels(month_names)
            ax.set_yticks(range(len(years)))
            ax.set_yticklabels(years)

            # Annotate cells
            for i in range(len(years)):
                for j in range(12):
                    if not np.isnan(data[i, j]):
                        ax.text(
                            j,
                            i,
                            f"{data[i, j]:.1f}%",
                            ha="center",
                            va="center",
                            fontsize=8,
                            fontweight="bold",
                        )

            fig.colorbar(im, ax=ax, label="Return %")
            ax.set_title("Monthly Returns Heatmap")
            fig.savefig(out / "monthly_returns.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("Chart: monthly_returns.png")
    except Exception as e:
        logger.warning("Monthly returns chart failed: %s", e)

    # --- 4. Rolling Sharpe Ratio ---
    if len(pnls) > 30:
        pnls_arr = np.array(pnls)
        window = 30
        rolling_mean = np.convolve(pnls_arr, np.ones(window) / window, mode="valid")
        rolling_std = np.array(
            [pnls_arr[i : i + window].std() for i in range(len(pnls_arr) - window + 1)]
        )
        rolling_sharpe = rolling_mean / (rolling_std + 1e-10) * np.sqrt(252)

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(rolling_sharpe, color=_COLORS["secondary"], linewidth=0.8)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.axhline(
            y=2,
            color=_COLORS["success"],
            linewidth=1,
            linestyle="--",
            alpha=0.5,
            label="Sharpe = 2",
        )
        ax.set_title(f"Rolling Sharpe Ratio (window={window} trades)")
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Annualized Sharpe")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(out / "rolling_sharpe.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Chart: rolling_sharpe.png")

    # --- 5. Trade Duration vs PnL Scatter ---
    durations = []
    pnls_for_scatter = []
    for t in trades:
        try:
            entry = datetime.fromisoformat(str(t["entry_time"]).replace("Z", "+00:00"))
            exit_ = datetime.fromisoformat(str(t["exit_time"]).replace("Z", "+00:00"))
            dur_hours = (exit_ - entry).total_seconds() / 3600
            durations.append(dur_hours)
            pnls_for_scatter.append(t["pnl"])
        except (ValueError, TypeError):
            continue

    if durations:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = [
            _COLORS["success"] if p > 0 else _COLORS["danger"] for p in pnls_for_scatter
        ]
        ax.scatter(durations, pnls_for_scatter, c=colors, alpha=0.5, s=20)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_title("Trade Duration vs PnL")
        ax.set_xlabel("Duration (hours)")
        ax.set_ylabel("PnL (USD)")
        ax.grid(True, alpha=0.3)
        fig.savefig(out / "duration_vs_pnl.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Chart: duration_vs_pnl.png")


def _compute_monthly_returns(trades: list[dict]) -> dict[tuple[int, int], float]:
    """Compute monthly percentage returns from trade list."""
    from datetime import datetime

    monthly_pnl: dict[tuple[int, int], float] = {}

    for t in trades:
        try:
            exit_time = datetime.fromisoformat(
                str(t["exit_time"]).replace("Z", "+00:00")
            )
            key = (exit_time.year, exit_time.month)
            monthly_pnl[key] = monthly_pnl.get(key, 0.0) + t["pnl"]
        except (ValueError, TypeError):
            continue

    # Convert to percentage returns (assume starting equity = 100k)
    # Use cumulative equity for more accurate monthly returns
    equity = 100_000.0
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
                old_start, old_end = equity_by_month[key]
                equity_by_month[key] = (old_start, end_eq)
        except (ValueError, TypeError):
            continue

    monthly_returns = {}
    for key, (start, end) in equity_by_month.items():
        if start > 0:
            monthly_returns[key] = (end - start) / start * 100

    return monthly_returns
