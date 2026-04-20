"""Report generation orchestrator."""

import json
import logging
from pathlib import Path
import pandas as pd

from thesis.config import Config
from thesis.report.builder import _build_markdown

from thesis.plots import _generate_data_charts
from thesis.plots import _generate_model_charts
from thesis.plots import _generate_backtest_charts

logger = logging.getLogger("thesis.report")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _plot_equity_curve(trades: list[dict], config: Config, out_dir: Path) -> None:
    """Generate and save the equity curve chart from trade history."""
    if not trades:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times, equity = _build_equity_series(trades, config.backtest.initial_capital)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(times, equity, linewidth=1)
    ax.set_title("Equity Curve")
    ax.set_ylabel("Equity (USD)")
    ax.set_xlabel("Date")
    fig.autofmt_xdate()
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "equity_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart saved: equity_curve.png")


def _build_equity_series(
    trades: list[dict], initial_capital: float
) -> tuple[list, list]:
    """Build timestamp and equity lists from trade history."""
    times = [pd.to_datetime(trades[0]["entry_time"])]
    equity = [initial_capital]
    for t in trades:
        times.append(pd.to_datetime(t["exit_time"]))
        equity.append(equity[-1] + t["pnl"])
    return times, equity


def _plot_feature_importance(feature_importance: dict, out_dir: Path) -> None:
    """Generate feature importance chart if data is available."""
    if not feature_importance:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = dict(list(feature_importance.items())[:20])
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(list(top.keys()), list(top.values()))
    ax.set_title("Feature Importance (Top 20)")
    ax.invert_yaxis()
    fig.savefig(out_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart saved: feature_importance.png")


def _load_feature_importance(config: Config, out_dir: Path) -> dict:
    """Load feature importance from JSON if available."""
    fi_path = (
        Path(config.paths.session_dir) / "reports" / "feature_importance.json"
        if config.paths.session_dir
        else out_dir.parent / "feature_importance.json"
    )
    if not fi_path.exists():
        return {}
    with open(fi_path) as f:
        return json.load(f)


def _load_ablation_results(config: Config) -> dict:
    """Load ablation results from JSON if available."""
    if not config.paths.session_dir:
        return {}
    abl_path = Path(config.paths.session_dir) / "reports" / "ablation_results.json"
    if not abl_path.exists():
        return {}
    with open(abl_path) as f:
        return json.load(f)


def generate_report(config: Config) -> None:
    """
    Generate thesis report with static charts and markdown.

    Args:
        config: Loaded application configuration.
    """
    import matplotlib

    matplotlib.use("Agg")

    if config.paths.session_dir:
        out_dir = Path(config.paths.session_dir) / "reports"
    else:
        out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load backtest results
    bt_path = Path(config.paths.backtest_results)
    metrics: dict = {}
    trades: list[dict] = []
    if bt_path.exists():
        with open(bt_path) as f:
            bt = json.load(f)
        metrics = bt.get("metrics", {})
        trades = bt.get("trades", [])

    _generate_data_charts(config)
    _generate_model_charts(config)
    _generate_backtest_charts(config)
    _plot_equity_curve(trades, config, out_dir)
    feature_importance = _load_feature_importance(config, out_dir)
    _plot_feature_importance(feature_importance, out_dir)
    ablation = _load_ablation_results(config)

    # Markdown Report
    md = _build_markdown(config, metrics, trades, feature_importance, ablation)
    report_path = Path(config.paths.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(md)
    logger.info("Report saved: %s", report_path)
