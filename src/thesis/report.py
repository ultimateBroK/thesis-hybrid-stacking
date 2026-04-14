"""Static report generation — matplotlib charts + markdown.

Generates: equity curve, feature importance bar chart, and a comprehensive
markdown report with model details, backtest metrics, and ablation results.
"""

import json
import logging
from pathlib import Path

from thesis.config import Config

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


def _build_markdown(
    config: Config,
    metrics: dict,
    trades: list[dict],
    feature_importance: dict,
    ablation: dict,
) -> str:
    """Build a comprehensive markdown report."""
    lines = [
        "# Thesis Report: Hybrid GRU + LightGBM for XAU/USD H1",
        "",
        "## Configuration",
        f"- Symbol: {config.data.symbol}",
        f"- Timeframe: {config.data.timeframe}",
        f"- ATR multiplier: {config.labels.atr_multiplier}",
        f"- Horizon: {config.labels.horizon_bars} bars",
        f"- Leverage: 1:{config.backtest.leverage}",
        f"- ATR stop multiplier: {config.backtest.atr_stop_multiplier}",
        "",
        "## Data Splits",
        f"- Train: {config.splitting.train_start} → {config.splitting.train_end}",
        f"- Val: {config.splitting.val_start} → {config.splitting.val_end}",
        f"- Test: {config.splitting.test_start} → {config.splitting.test_end}",
        f"- Purge: {config.splitting.purge_bars} bars",
        f"- Embargo: {config.splitting.embargo_bars} bars",
        "",
        "## Model Architecture",
        "### GRU Feature Extractor",
        f"- Input: log_returns + rsi_14 ({config.gru.input_size} features)",
        f"- Hidden size: {config.gru.hidden_size}",
        f"- Layers: {config.gru.num_layers}",
        f"- Sequence length: {config.gru.sequence_length}",
        f"- Dropout: {config.gru.dropout}",
        f"- Learning rate: {config.gru.learning_rate}",
        f"- Epochs: {config.gru.epochs} (patience={config.gru.patience})",
        "",
        "### LightGBM Classifier",
        f"- Leaves: {config.model.num_leaves}, Depth: {config.model.max_depth}",
        f"- Learning rate: {config.model.learning_rate}",
        f"- Estimators: {config.model.n_estimators}",
        f"- Min child samples: {config.model.min_child_samples}",
        f"- Subsample: {config.model.subsample}, Feature fraction: {config.model.feature_fraction}",
        f"- Regularization: alpha={config.model.reg_alpha}, lambda={config.model.reg_lambda}",
        f"- Optuna: {'enabled' if config.model.use_optuna else 'disabled'}",
        "",
        "### Hybrid Feature Space",
        f"- GRU hidden states: {config.gru.hidden_size} dimensions",
        "- Static features: 11 (RSI, ATR, MACD, sessions, pivots, etc.)",
        f"- Total features: {config.gru.hidden_size + 11}",
        "",
        "## Backtest Results",
    ]

    if metrics:
        # Group metrics logically (using backtesting.py normalized keys)
        trade_metrics = {
            "num_trades": "Total Trades",
        }
        perf_metrics = {
            "return_pct": "Return %",
            "profit_factor": "Profit Factor",
            "equity_final": "Final Equity",
        }
        risk_metrics = {
            "sharpe_ratio": "Sharpe Ratio",
            "sortino_ratio": "Sortino Ratio",
            "calmar_ratio": "Calmar Ratio",
            "max_drawdown_pct": "Max Drawdown %",
        }
        other_metrics = {
            "win_rate_pct": "Win Rate %",
            "avg_trade_pct": "Avg Trade %",
        }

        for group_name, group in [
            ("Trade Statistics", trade_metrics),
            ("Performance", perf_metrics),
            ("Risk Metrics", risk_metrics),
            ("Other", other_metrics),
        ]:
            lines.append(f"### {group_name}")
            for key, label in group.items():
                if key in metrics:
                    val = metrics[key]
                    if isinstance(val, float):
                        lines.append(f"- **{label}**: {val:.4f}")
                    else:
                        lines.append(f"- **{label}**: {val}")
            lines.append("")
    else:
        lines.append("No backtest results available.")
        lines.append("")

    # Top features
    if feature_importance:
        lines.append("## Top 10 Features")
        lines.append("")
        for i, (name, imp) in enumerate(list(feature_importance.items())[:10], 1):
            ftype = "GRU" if name.startswith("gru_") else "Static"
            lines.append(f"{i}. **{name}** ({ftype}): {imp:.1f}")
        lines.append("")

    # Ablation results
    if ablation:
        lines.append("## Ablation Study")
        lines.append("")
        lines.append(
            "| Variant | Features | Trades | Win Rate | Return % | Sharpe | Max DD % |"
        )
        lines.append(
            "|---------|----------|--------|----------|----------|--------|----------|"
        )

        for variant in ["lgbm_only", "gru_only", "combined"]:
            if variant in ablation:
                v = ablation[variant]
                m = v.get("metrics", {})
                fc = v.get("feature_count", "?")
                trades_v = m.get("num_trades", 0)
                wr = m.get("win_rate_pct", 0)
                ret = m.get("return_pct", 0)
                sh = m.get("sharpe_ratio", 0)
                dd = m.get("max_drawdown_pct", 0)
                lines.append(
                    f"| {variant} | {fc} | {trades_v} | {wr:.4f} | {ret:.2f} | {sh:.4f} | {dd:.2f} |"
                )

        if "comparison_note" in ablation:
            lines.append("")
            lines.append(ablation["comparison_note"])
        lines.append("")

    # Charts
    lines.append("## Charts")
    lines.append("")
    if config.paths.session_dir:
        bt_chart = Path(config.paths.session_dir) / "backtest" / "backtest_chart.html"
        if bt_chart.exists():
            lines.append("### Interactive Backtest Chart")
            lines.append("")
            lines.append(
                "See [backtest_chart.html](backtest_chart.html) for the interactive Bokeh visualization."
            )
            lines.append("")
    lines.append("![Equity Curve](equity_curve.png)")
    lines.append("![Feature Importance](feature_importance.png)")
    lines.append("")

    # Visualization charts
    if config.paths.session_dir:
        charts_dir = Path(config.paths.session_dir) / "reports" / "charts"
        if charts_dir.exists():
            for subdir in ["data", "model", "backtest"]:
                sub = charts_dir / subdir
                if sub.exists():
                    lines.append(f"### {subdir.title()} Charts")
                    lines.append("")
                    for img in sorted(sub.glob("*.png")):
                        rel = f"charts/{subdir}/{img.name}"
                        lines.append(f"![{img.stem}]({rel})")
                    lines.append("")

    return "\n".join(lines)
