"""Report generation orchestrator."""

import json
import logging
from pathlib import Path

from thesis.config import Config
from thesis.report.builder import _build_markdown

logger = logging.getLogger("thesis.report")


def generate_report(config: Config) -> None:
    """Generate thesis report with static charts and markdown.

    Args:
        config: Loaded application configuration.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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

    # --- 1. Equity Curve ---
    if trades:
        pnls = [t["pnl"] for t in trades]
        equity = [config.backtest.initial_capital]
        for p in pnls:
            equity.append(equity[-1] + p)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(equity, linewidth=1)
        ax.set_title("Equity Curve")
        ax.set_ylabel("Equity (USD)")
        ax.set_xlabel("Trade #")
        ax.grid(True, alpha=0.3)
        fig.savefig(out_dir / "equity_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Chart saved: equity_curve.png")

    # --- 2. Feature Importance ---
    if config.paths.session_dir:
        fi_path = Path(config.paths.session_dir) / "reports" / "feature_importance.json"
    else:
        fi_path = Path("results/feature_importance.json")
    feature_importance: dict = {}
    if fi_path.exists():
        with open(fi_path) as f:
            feature_importance = json.load(f)
        top = dict(list(feature_importance.items())[:20])

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(list(top.keys()), list(top.values()))
        ax.set_title("Feature Importance (Top 20)")
        ax.invert_yaxis()
        fig.savefig(out_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Chart saved: feature_importance.png")

    # --- 3. Load ablation results if available ---
    ablation: dict = {}
    if config.paths.session_dir:
        abl_path = Path(config.paths.session_dir) / "reports" / "ablation_results.json"
        if abl_path.exists():
            with open(abl_path) as f:
                ablation = json.load(f)

    # --- 4. Markdown Report ---
    md = _build_markdown(config, metrics, trades, feature_importance, ablation)
    report_path = Path(config.paths.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(md)
    logger.info("Report saved: %s", report_path)
