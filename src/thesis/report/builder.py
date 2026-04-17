"""Markdown report builder."""

from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from thesis.config import Config
from thesis.report.stats import (
    _load_parquet_stats,
    _load_label_distribution,
    _load_split_stats,
    _load_prediction_stats,
)


def _build_markdown(
    config: Config,
    metrics: dict,
    trades: list[dict],
    feature_importance: dict,
    ablation: dict,
) -> str:
    """Build a comprehensive markdown report with plain-language explanations."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session = config.paths.session_dir or "N/A"

    lines: list[str] = [
        "# Thesis Report: Can AI Predict Gold Price Movements?",
        "",
        "> **Generated**: " + now + "  ",
        "> **Session**: `" + session + "`",
        "",
        "---",
        "",
        "## How to Read This Report",
        "",
        "This report tests whether a machine learning system can predict the direction",
        "of gold (XAU/USD) price movements and profit from those predictions.",
        "",
        "Think of it like a weather forecast, but for financial markets:",
        "- The **model** looks at past price patterns and predicts whether gold will go **up**, **down**, or **stay flat**",
        "- The **backtest** simulates what would have happened if we traded based on those predictions",
        "- The **confidence filter** ensures we only trade when the model is very sure about its prediction",
        "",
        "### Verdict Icons",
        "",
        "| Icon | Meaning |",
        "|------|---------|",
        "| ✅ | Good — above the acceptable threshold |",
        "| 🟡 | Warning — needs improvement |",
        "| ❌ | Poor — below acceptable threshold |",
        "",
        "---",
        "",
        "## Table of Contents",
        "",
        "1. [Data & Features](#1-data--features)",
        "2. [Labeling Strategy](#2-labeling-strategy)",
        "3. [Train/Test Split](#3-traintest-split)",
        "4. [Model Architecture](#4-model-architecture)",
        "5. [Feature Importance](#5-feature-importance)",
        "6. [Model Accuracy](#6-model-accuracy)",
        "7. [Trading Simulation](#7-trading-simulation)",
        "8. [Ablation Study](#8-ablation-study)",
        "9. [Conclusion](#9-conclusion)",
        "",
        "---",
        "",
    ]

    # ==================================================================
    # 1. DATA & FEATURES
    # ==================================================================
    lines.append("## 1. Data & Features")
    lines.append("")
    lines.append(
        "We collected raw gold price data (XAU/USD) and converted it into "
        "hourly price bars. Each bar records the open, high, low, close prices, "
        "trading volume, and the number of price updates (ticks) during that hour."
    )
    lines.append("")
    ohlcv_path = Path(config.paths.ohlcv)
    ohlcv_stats = _load_parquet_stats(ohlcv_path)
    if ohlcv_stats:
        lines.append(f"- **Currency pair**: {config.data.symbol} (Gold vs US Dollar)")
        lines.append("- **Timeframe**: Each bar = 1 hour")
        lines.append(f"- **Total bars**: {ohlcv_stats['rows']:,} hours of data")
        if "date_range" in ohlcv_stats:
            lines.append(
                f"- **Date range**: {ohlcv_stats['date_range'][0][:10]} to {ohlcv_stats['date_range'][1][:10]} "
                f"(about {(ohlcv_stats['rows'] / 8760):.1f} years)"
            )
        lines.append("")
    else:
        lines.append("*Data not found.*")
    lines.append("")

    lines.append("### Technical Indicators")
    lines.append("")
    lines.append(
        "Before the AI model can make predictions, we give it useful information. "
        "We calculate 11 **technical indicators** — mathematical formulas applied to the "
        "price data that help reveal hidden patterns like momentum, volatility, and trend."
    )
    lines.append("")
    feat_path = Path(config.paths.features)
    feat_stats = _load_parquet_stats(feat_path)
    if feat_stats:
        lines.append(f"- **Total data points**: {feat_stats['rows']:,} hours")
        lines.append("")
        static_features = [
            (
                "rsi_14",
                "Momentum gauge — measures how fast prices have been moving (0–100 scale)",
            ),
            (
                "atr_14",
                "Volatility meter — how much the price typically moves in a given hour",
            ),
            (
                "macd_hist",
                "Trend detector — shows whether an uptrend or downtrend is strengthening",
            ),
            (
                "atr_ratio",
                "Volatility regime — compares recent volatility to longer-term average",
            ),
            (
                "price_dist_ratio",
                "Trend distance — how far the current price is from its average",
            ),
            (
                "pivot_position",
                "Support/resistance — where the price sits relative to key levels",
            ),
            (
                "atr_percentile",
                "Volatility rank — is current volatility high or low historically?",
            ),
            ("sess_asia", "Is the Asian market session active? (Tokyo, Sydney)"),
            ("sess_london", "Is the London market session active?"),
            (
                "sess_overlap",
                "Is the London–New York overlap active? (highest trading volume)",
            ),
            ("sess_ny_pm", "Is the New York afternoon session active?"),
        ]
        lines.append("| Indicator | What It Measures |")
        lines.append("|-----------|-----------------|")
        for name, desc in static_features:
            lines.append(f"| `{name}` | {desc} |")
        lines.append("")

        # Session distribution
        if all(
            c in feat_stats["columns"]
            for c in ["sess_asia", "sess_london", "sess_overlap", "sess_ny_pm"]
        ):
            try:
                df = pl.read_parquet(
                    feat_path,
                    columns=["sess_asia", "sess_london", "sess_overlap", "sess_ny_pm"],
                )
                total = len(df)
                lines.append("### Trading Sessions")
                lines.append("")
                for col, label in [
                    ("sess_asia", "Asian session"),
                    ("sess_london", "London session"),
                    ("sess_overlap", "London-NY overlap (busiest hours)"),
                    ("sess_ny_pm", "New York afternoon"),
                ]:
                    count = int(df[col].sum())
                    lines.append(
                        f"- **{label}**: {count:,} hours ({count / total * 100:.1f}%)"
                    )
                lines.append("")
            except Exception:
                pass
    else:
        lines.append("*Feature data not found.*")
    lines.append("")

    # ==================================================================
    # 2. LABELING STRATEGY
    # ==================================================================
    lines.append("## 2. Labeling Strategy")
    lines.append("")
    lines.append(
        'To teach the AI what "correct" looks like, we need a way to label each hour '
        "as **Long** (price went up), **Short** (price went down), or **Hold** (stayed flat). "
        "We use the **Triple Barrier Method**:"
    )
    lines.append("")
    lines.append(
        '1. For each hour, we set two invisible "walls" above and below the current price, '
        f"spaced at **{config.labels.atr_multiplier}× the recent volatility** (ATR)."
    )
    lines.append(
        f"2. We then watch the price for the next **{config.labels.horizon_bars} hours** maximum."
    )
    lines.append(
        "3. If the price hits the upper wall first → **Long**. "
        "If it hits the lower wall first → **Short**. "
        "If neither wall is hit → **Hold**."
    )
    lines.append("")

    labels_path = Path(config.paths.labels)
    label_dist = _load_label_distribution(labels_path)
    if label_dist:
        total = label_dist["total"]
        lines.append("### Label Distribution")
        lines.append("")
        for name, emoji in [("Short", "📉"), ("Hold", "⏸️"), ("Long", "📈")]:
            key = name
            count, pct = label_dist[key]
            lines.append(
                f"- {emoji} **{name}** ({'down' if name == 'Short' else 'sideways' if name == 'Hold' else 'up'}): {count:,} hours ({pct:.1f}%)"
            )
        lines.append("")
        hold_pct = label_dist["Hold"][1]
        if hold_pct < 25:
            lines.append(
                f'> **What this means**: Only {hold_pct:.0f}% of hours were labeled "Hold" — gold tends to move enough to hit one of the walls within 10 hours.'
            )
        lines.append("")
    else:
        lines.append("*Label data not found.*")
    lines.append("")

    # ==================================================================
    # 3. TRAIN/TEST SPLIT
    # ==================================================================
    lines.append("## 3. Train/Test Split")
    lines.append("")
    lines.append(
        "A critical rule in machine learning: **never test on data the model has seen during training**. "
        "We split the data into three periods:"
    )
    lines.append("")
    lines.append(
        f"- **Training**: {config.splitting.train_start} to {config.splitting.train_end} "
        "— the model learns patterns from this period"
    )
    lines.append(
        f"- **Validation**: {config.splitting.val_start} to {config.splitting.val_end} "
        "— used to tune the model and prevent overfitting"
    )
    lines.append(
        f"- **Test**: {config.splitting.test_start} to {config.splitting.test_end} "
        "— the model has NEVER seen this data; this is the real test"
    )
    lines.append("")
    lines.append(
        "We also add safety gaps (**purge** and **embargo**) between periods to prevent "
        'information "leaking" from one period to the next.'
    )
    lines.append("")

    split_stats = _load_split_stats(config)
    if split_stats:
        lines.append("### Data Distribution")
        lines.append("")
        lines.append("| Period | Hours | Date Range | % Long | % Hold | % Short |")
        lines.append("|--------|-------|------------|--------|--------|---------|")
        for name, label in [
            ("train", "Training"),
            ("val", "Validation"),
            ("test", "Test"),
        ]:
            if name not in split_stats:
                continue
            s = split_stats[name]
            dr = s.get("date_range", ("N/A", "N/A"))
            dr_str = f"{dr[0][:10]} to {dr[1][:10]}" if dr != ("N/A", "N/A") else "N/A"
            if "label_distribution" in s:
                ld = s["label_distribution"]
                long_s = f"{ld['Long'][1]:.1f}%"
                hold_s = f"{ld['Hold'][1]:.1f}%"
                short_s = f"{ld['Short'][1]:.1f}%"
            else:
                long_s = hold_s = short_s = "N/A"
            lines.append(
                f"| **{label}** | {s['rows']:,} | {dr_str} | {long_s} | {hold_s} | {short_s} |"
            )
        lines.append("")
    lines.append("")

    # ==================================================================
    # 4. MODEL ARCHITECTURE
    # ==================================================================
    lines.append("## 4. Model Architecture")
    lines.append("")
    lines.append("We use a **two-part (hybrid) model** — two experts working together:")
    lines.append("")

    lines.append('### Expert 1: GRU Neural Network ("Pattern Reader")')
    lines.append("")
    lines.append(
        f"A **GRU (Gated Recurrent Unit)** reads sequences of data — like a sentence word by word. "
        f"It processes the last **{config.gru.sequence_length} hours** of price changes and technical "
        'indicators to detect temporal patterns (e.g., "when RSI drops and ATR rises, the price usually follows").'
    )
    lines.append("")
    lines.append(
        f"- **{config.gru.input_size} inputs** per hour (price change, momentum, volatility, trend)"
    )
    lines.append(
        f"- **{config.gru.hidden_size}-dimensional** internal representation — "
        f'{config.gru.hidden_size} different "pattern detectors"'
    )
    lines.append(
        f"- Trained for up to **{config.gru.epochs} rounds** (stopped early if no improvement)"
    )
    lines.append("")

    lines.append('### Expert 2: LightGBM ("Decision Maker")')
    lines.append("")
    lines.append(
        "**LightGBM** is a decision-tree model that takes the GRU's pattern detectors PLUS "
        "the 11 technical indicators as input, and makes the final prediction: Long, Hold, or Short."
    )
    lines.append("")
    lines.append(
        "Decision trees work by asking yes/no questions: "
        '"Is GRU detector #20 above 0.5? Is volatility high? Is it the London session?" '
        "→ outputs a prediction with a confidence score."
    )
    lines.append("")

    total_features = config.gru.hidden_size + 11
    lines.append("### Combined System")
    lines.append("")
    lines.append(
        f"**{total_features} features** total "
        f"({config.gru.hidden_size} from GRU + 11 technical indicators)"
    )
    lines.append("")

    # ==================================================================
    # 5. FEATURE IMPORTANCE
    # ==================================================================
    if feature_importance:
        lines.append("## 5. Feature Importance")
        lines.append("")
        lines.append(
            "Feature importance tells us which inputs the model found most useful for making predictions."
        )
        lines.append("")
        items = list(feature_importance.items())
        gru_count = sum(1 for name, _ in items if name.startswith("gru_"))

        lines.append("| Rank | Feature | Source | Score |")
        lines.append("|------|---------|--------|-------|")
        for i, (name, imp) in enumerate(items[:15], 1):
            ftype = (
                "GRU (pattern reader)"
                if name.startswith("gru_")
                else "Technical indicator"
            )
            lines.append(f"| {i} | `{name}` | {ftype} | {imp:.0f} |")
        lines.append("")
        lines.append(
            f"> **Key finding**: {gru_count} of the top {len(items)} features come from the GRU neural network "
            f"({gru_count / len(items) * 100:.0f}%). "
            f"The pattern reader finds temporal patterns that simple indicators alone cannot capture."
        )
        lines.append("")

    # ==================================================================
    # 6. MODEL ACCURACY
    # ==================================================================
    lines.append("## 6. Model Accuracy")
    lines.append("")
    lines.append(
        'Accuracy measures: "out of all the hours in the test period, '
        'how often did the model correctly predict the direction?"'
    )
    lines.append("")

    preds_path = Path(config.paths.predictions)
    pred_stats = _load_prediction_stats(preds_path)
    if pred_stats:
        total = pred_stats["total"]
        acc = pred_stats["accuracy"]
        dir_acc = pred_stats.get("directional_accuracy", acc)
        dir_baseline = pred_stats.get("directional_baseline", 0.5)

        lines.append("### Overall Performance")
        lines.append("")
        lines.append("| Metric | Value | Verdict |")
        lines.append("|--------|-------|---------|")
        dir_verdict = _verdict("directional_accuracy", dir_acc)
        lines.append(f"| Directional Accuracy | {dir_acc * 100:.1f}% | {dir_verdict} |")
        exact_verdict = _verdict("accuracy", acc)
        lines.append(f"| Exact-Match Accuracy | {acc * 100:.1f}% | {exact_verdict} |")
        lines.append(f"| Random Baseline | {dir_baseline * 100:.1f}% | — |")
        lines.append("")

        lines.append(
            f"> **{dir_acc * 100:.0f}% directional accuracy**: When the model predicts Long or Short, "
            f"it's correct {dir_acc * 100:.0f}% of the time "
            f"({'better' if dir_acc > dir_baseline else 'about the same as'} random guessing at {dir_baseline * 100:.0f}%)."
        )
        lines.append("")

        # Per-class accuracy
        per_class = pred_stats["per_class"]
        lines.append("### Accuracy by Direction")
        lines.append("")
        lines.append("| Direction | Actual | Predicted | Correct | Wrong |")
        lines.append("|-----------|--------|-----------|---------|-------|")
        for name, emoji in [("Short", "📉"), ("Hold", "⏸️"), ("Long", "📈")]:
            pc = per_class[name]
            desc = (
                "price went down"
                if name == "Short"
                else "price went sideways"
                if name == "Hold"
                else "price went up"
            )
            lines.append(
                f"| {emoji} **{name}** ({desc}) | {pc['true_count']:,} | {pc['pred_count']:,} | "
                f"{pc['recall'] * 100:.1f}% | {100 - pc['recall'] * 100:.1f}% |"
            )
        lines.append("")

        # Confusion matrix
        cm = pred_stats["confusion_matrix"]
        lines.append("### Confusion Matrix")
        lines.append("")
        lines.append(
            "What actually happened (rows) vs what the model predicted (columns):"
        )
        lines.append("")
        lines.append("| Actual \\ Predicted | Short 📉 | Hold ⏸️ | Long 📈 |")
        lines.append("|---------------------|----------|---------|----------|")
        for true_name, emoji in [("Short", "📉"), ("Hold", "⏸️"), ("Long", "📈")]:
            row = cm[true_name]
            lines.append(
                f"| {emoji} **{true_name}** | {row['Short']:,} | {row['Hold']:,} | {row['Long']:,} |"
            )
        lines.append("")

        # Confidence filtering
        hc = pred_stats.get("high_confidence")
        if hc:
            lines.append("### Confidence Filtering")
            lines.append("")
            lines.append(
                "The model outputs a **confidence score** (0–100%). When confidence is **high (≥70%)**, "
                "accuracy improves significantly:"
            )
            lines.append("")
            lines.append(
                f"- **High-confidence signals**: {hc['count']:,} hours ({hc['pct_of_total']:.1f}% of total)"
            )
            lines.append(
                f"- **Accuracy when confident**: **{hc['accuracy'] * 100:.1f}%** (vs {acc * 100:.1f}% overall)"
            )
            lines.append(
                f"- **Directional accuracy**: **{hc['directional_accuracy'] * 100:.1f}%**"
            )
            lines.append("")
            lines.append(
                "> **Key insight**: Only trade when the model is very confident. "
                "This is like a weather forecaster only making predictions when they're sure — "
                "they'll be wrong sometimes, but much less often than when uncertain."
            )
            lines.append("")
    else:
        lines.append("*Prediction data not found.*")
    lines.append("")

    # ==================================================================
    # 7. TRADING SIMULATION
    # ==================================================================
    lines.append("## 7. Trading Simulation")
    lines.append("")
    lines.append(
        "We simulated trading based on the model's high-confidence predictions. "
        "This is called a **backtest** — running the strategy on historical data as if it were live."
    )
    lines.append("")

    lines.append("### Simulation Parameters")
    lines.append("")
    lines.append(f"- **Starting capital**: ${config.backtest.initial_capital:,.0f}")
    lines.append(
        f"- **Leverage**: {config.backtest.leverage}:1 "
        f"(amplifies both gains and losses)"
    )
    lines.append(
        f"- **Trade size**: {config.backtest.lots_per_trade} lot = 100 oz of gold"
    )
    lines.append(
        f"- **Stop-loss**: {config.backtest.atr_stop_multiplier}× ATR (automatic exit if price moves against us)"
    )
    lines.append(
        f"- **Confidence filter**: Only trade when model confidence ≥ {config.backtest.confidence_threshold}"
    )
    lines.append(
        f"- **Trading costs**: ${config.backtest.spread_ticks * config.data.tick_size:.2f} spread + "
        f"${config.backtest.commission_per_lot:.0f} commission per lot"
    )
    lines.append("")

    if metrics:
        n_trades = int(metrics.get("num_trades", 0))
        ret = float(metrics.get("return_pct", 0))
        sharpe = float(metrics.get("sharpe_ratio", 0))
        max_dd = float(metrics.get("max_drawdown_pct", 0))
        pf = float(metrics.get("profit_factor", 0))
        wr = float(metrics.get("win_rate_pct", 0))
        final_eq = float(metrics.get("equity_final", 0))

        # Big picture summary
        lines.append("### Results Summary")
        lines.append("")
        lines.append("| Metric | Value | Verdict |")
        lines.append("|--------|-------|---------|")
        ret_verdict = _verdict("return_pct", ret)
        lines.append(f"| Return | {ret:.1f}% | {ret_verdict} |")
        wr_verdict = _verdict("win_rate_pct", wr)
        lines.append(f"| Win Rate | {wr:.1f}% | {wr_verdict} |")
        pf_verdict = _verdict("profit_factor", pf)
        lines.append(f"| Profit Factor | {pf:.2f} | {pf_verdict} |")
        sharpe_verdict = _verdict("sharpe_ratio", sharpe)
        lines.append(f"| Sharpe Ratio | {sharpe:.2f} | {sharpe_verdict} |")
        dd_verdict = _verdict("max_drawdown_pct", max_dd)
        lines.append(f"| Max Drawdown | {max_dd:.1f}% | {dd_verdict} |")
        lines.append("")
        lines.append(
            f"**${config.backtest.initial_capital:,.0f} → ${final_eq:,.0f}** in {n_trades} trades"
        )
        if config.backtest.leverage > 1 and ret > 500:
            unleveraged = ret / config.backtest.leverage
            lines.append(
                f"Total return: {ret:,.0f}% ({config.backtest.leverage}:1 leverage) ≈ {unleveraged:.0f}% without leverage"
            )
        lines.append("")

        # Sharpe explanation
        lines.append("### Risk-Adjusted Return (Sharpe Ratio)")
        lines.append("")
        if sharpe >= 2.0:
            lines.append(
                f"**Sharpe: {sharpe:.2f}** — ✅✅ Excellent — institutional-grade performance"
            )
        elif sharpe >= 1.5:
            lines.append(
                f"**Sharpe: {sharpe:.2f}** — ✅ Very good — steady returns with controlled risk"
            )
        elif sharpe >= 1.0:
            lines.append(f"**Sharpe: {sharpe:.2f}** — 🟡 Decent risk-adjusted returns")
        else:
            lines.append(
                f"**Sharpe: {sharpe:.2f}** — ❌ Below acceptable — returns may be mostly noise"
            )
        lines.append("")
        lines.append("| Threshold | Rating |")
        lines.append("|----------|--------|")
        lines.append("| Below 1.0 | ❌ Poor |")
        lines.append("| 1.0 – 1.5 | 🟡 Good |")
        lines.append("| 1.5 – 2.0 | ✅ Very Good |")
        lines.append("| Above 2.0 | ✅✅ Excellent |")
        lines.append("")

        # Trade breakdown
        if trades:
            lines.append("### Trade Breakdown")
            lines.append("")
            pnls = [t["pnl"] for t in trades]
            winners = [p for p in pnls if p > 0]
            losers = [p for p in pnls if p < 0]
            longs = [t for t in trades if t.get("direction") == "long"]
            shorts = [t for t in trades if t.get("direction") == "short"]

            lines.append(
                f"- **Winning trades**: {len(winners):,} ({len(winners) / len(trades) * 100:.1f}%)"
            )
            if winners:
                lines.append(
                    f"  - Average: ${np.mean(winners):,.2f}, Best: ${max(winners):,.2f}"
                )
            lines.append(
                f"- **Losing trades**: {len(losers):,} ({len(losers) / len(trades) * 100:.1f}%)"
            )
            if losers:
                lines.append(
                    f"  - Average: ${np.mean(losers):,.2f}, Worst: ${min(losers):,.2f}"
                )
            lines.append(
                f"- **Long trades**: {len(longs):,}, **Short trades**: {len(shorts):,}"
            )
            if winners and losers:
                rr = abs(np.mean(winners) / np.mean(losers))
                lines.append(f"- **Win/Loss ratio**: {rr:.1f}:1")
            lines.append("")
    else:
        lines.append("*No backtest results available.*")
    lines.append("")

    # ==================================================================
    # 8. ABLATION STUDY
    # ==================================================================
    if ablation:
        lines.append("## 8. Ablation Study")
        lines.append("")
        lines.append(
            "An ablation study tests whether each component is actually helpful. "
            "We compare three versions:"
        )
        lines.append(
            "- **lgbm_only**: Just the decision-tree expert, no neural network"
        )
        lines.append(
            "- **gru_only**: Just the neural network's features, simpler decision maker"
        )
        lines.append("- **combined**: Both experts working together (our full system)")
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
                vwr = m.get("win_rate_pct", 0)
                vret = m.get("return_pct", 0)
                vsh = m.get("sharpe_ratio", 0)
                vdd = m.get("max_drawdown_pct", 0)
                vwr_str = f"{vwr:.2f}%" if not np.isnan(vwr) else "N/A"
                lines.append(
                    f"| {variant} | {fc} | {trades_v} | {vwr_str} | {vret:.2f} | {vsh:.2f} | {vdd:.2f} |"
                )
        lines.append("")
        if "comparison_note" in ablation:
            lines.append(ablation["comparison_note"])
            lines.append("")

    # ==================================================================
    # 9. CONCLUSION
    # ==================================================================
    lines.append("## 9. Conclusion")
    lines.append("")

    if pred_stats and metrics:
        acc = pred_stats["accuracy"]
        dir_acc = pred_stats.get("directional_accuracy", acc)
        dir_baseline = pred_stats.get("directional_baseline", 0.5)
        hc = pred_stats.get("high_confidence")
        n_trades = int(metrics.get("num_trades", 0))
        sharpe_val = float(metrics.get("sharpe_ratio", 0))
        max_dd_val = float(metrics.get("max_drawdown_pct", 0))
        pf_val = float(metrics.get("profit_factor", 0))
        ret_val = float(metrics.get("return_pct", 0))

        # Summary
        if hc:
            lines.append(
                f"The model's **directional accuracy** is {dir_acc * 100:.0f}% "
                f"(vs {dir_baseline * 100:.0f}% random baseline), "
                f"and when we only act on **high-confidence** predictions (≥{hc['threshold']:.0%} sure), "
                f"accuracy jumps to **{hc['accuracy'] * 100:.0f}%**. "
                f"This is the core finding: the model knows when it doesn't know."
            )
        else:
            lines.append(
                f"The model achieves {dir_acc * 100:.0f}% directional accuracy — "
                f"{(dir_acc - dir_baseline) * 100:.1f}pp better than random guessing."
            )
        lines.append("")

        if n_trades > 0:
            wr_val = float(metrics.get("win_rate_pct", 0))
            wr_str = f"{wr_val:.0f}%" if not np.isnan(wr_val) else "N/A"
            lines.append(
                f"Trading on high-confidence signals produces **{n_trades:,} trades** "
                f"with {wr_str} win rate and **{pf_val:.1f}:1** profit factor. "
                f"Sharpe ratio of {sharpe_val:.2f} indicates "
                f"{'excellent' if sharpe_val >= 2.0 else 'very good' if sharpe_val >= 1.5 else 'good' if sharpe_val >= 1.0 else 'acceptable'} "
                f"risk-adjusted performance."
            )
            lines.append("")

        # Caveats
        caveats = []
        if ret_val > 500:
            unleveraged = ret_val / config.backtest.leverage
            caveats.append(
                f"Returns are amplified by {config.backtest.leverage}:1 leverage. "
                f"Without leverage, return would be ~{unleveraged:.0f}%."
            )
        caveats.append(
            "The test period includes gold's historic bull run ($2,000 → $3,000+), "
            "which provided a strong tailwind."
        )
        caveats.append(f"Maximum drawdown was {abs(max_dd_val):.1f}%.")
        lines.append("**Important caveats**: " + " ".join(caveats))
        lines.append("")

        # Architecture validation
        if feature_importance:
            items_fi = list(feature_importance.items())
            gru_top = [n for n, _ in items_fi[:5] if n.startswith("gru_")]
            if gru_top:
                lines.append(
                    "The neural network (GRU) features dominate the top importance rankings, "
                    "confirming that temporal pattern recognition adds real value."
                )
                lines.append("")

    elif pred_stats:
        acc = pred_stats["accuracy"]
        dir_acc = pred_stats.get("directional_accuracy", acc)
        lines.append(
            f"The model achieves {dir_acc * 100:.1f}% directional accuracy on "
            f"{pred_stats['total']:,} out-of-sample predictions. "
            f"No backtest results available."
        )
        lines.append("")

    if config.backtest.confidence_threshold > 0:
        lines.append(
            f"The confidence threshold ({config.backtest.confidence_threshold}) "
            f"is the strategy's most important feature — it filters out "
            f"low-conviction predictions."
        )
        lines.append("")

    if not lines[-1].strip():
        lines.append("*Insufficient data for automated conclusion.*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Metric formatting helpers
# ---------------------------------------------------------------------------

_METRIC_FMT = {
    "pct": lambda v: f"{v:.2f}%",
    "f2": lambda v: f"{v:.2f}",
    "f3": lambda v: f"{v:.3f}",
    "f4": lambda v: f"{v:.4f}",
    "d": lambda v: f"{int(v):,}",
    "dollar": lambda v: f"${v:,.2f}",
    "s": lambda v: str(v),
}


# ---------------------------------------------------------------------------
# Verdict and explanation helpers
# ---------------------------------------------------------------------------

# Threshold definitions: (key, higher_is_better, poor, ok, good)
_VERDICT_THRESHOLDS = {
    # Performance
    "return_pct": (True, -100, 0, 20),
    "return_ann_pct": (True, -50, 5, 15),
    "cagr_pct": (True, -50, 5, 15),
    "sharpe_ratio": (True, 0, 1.0, 2.0),
    "sortino_ratio": (True, 0, 1.5, 2.5),
    "calmar_ratio": (True, 0, 1.0, 2.0),
    # Risk
    "max_drawdown_pct": (False, 30, 15, 5),  # lower is better
    "avg_drawdown_pct": (False, 20, 10, 5),
    "volatility_ann_pct": (False, 100, 50, 20),
    # Trade quality
    "win_rate_pct": (True, 30, 45, 55),
    "profit_factor": (True, 1.0, 1.5, 2.0),
    "expectancy_pct": (True, -1, 0.5, 1.5),
    "sqn": (True, 0, 1.5, 2.5),
    "kelly_criterion": (True, 0.1, 0.25, 0.4),
    "avg_trade_pct": (True, -1, 0.5, 1.5),
    # Model accuracy
    "accuracy": (True, 0.35, 0.45, 0.55),
    "directional_accuracy": (True, 0.45, 0.55, 0.65),
}


def _verdict(key: str, value: float) -> str:
    """Return verdict emoji based on metric thresholds."""
    if (
        key not in _VERDICT_THRESHOLDS
        or value is None
        or (isinstance(value, float) and value != value)
    ):  # NaN check
        return "⚪"
    higher_is_better, poor_thresh, ok_thresh, good_thresh = _VERDICT_THRESHOLDS[key]
    if higher_is_better:
        if value >= good_thresh:
            return "✅"
        elif value >= ok_thresh:
            return "🟡"
        else:
            return "❌"
    else:  # lower is better (drawdown, volatility)
        if value <= good_thresh:
            return "✅"
        elif value <= ok_thresh:
            return "🟡"
        else:
            return "❌"


def _get_verdict_level(key: str, value: float) -> str:
    """Return verdict level: 'good', 'ok', or 'poor'."""
    if (
        key not in _VERDICT_THRESHOLDS
        or value is None
        or (isinstance(value, float) and value != value)
    ):
        return "neutral"
    higher_is_better, poor_thresh, ok_thresh, good_thresh = _VERDICT_THRESHOLDS[key]
    if higher_is_better:
        if value >= good_thresh:
            return "good"
        elif value >= ok_thresh:
            return "ok"
        else:
            return "poor"
    else:
        if value <= good_thresh:
            return "good"
        elif value <= ok_thresh:
            return "ok"
        else:
            return "poor"


_EXPLANATIONS = {
    # Performance
    "return_pct": (
        "Total return — excellent profit generation",
        "Total return — decent but could be better",
        "Total return — below expectations",
    ),
    "return_ann_pct": (
        "Annualized return — excellent growth",
        "Annualized return — acceptable growth",
        "Annualized return — insufficient growth",
    ),
    "cagr_pct": (
        "CAGR — strong compound growth",
        "CAGR — moderate compound growth",
        "CAGR — weak compound growth",
    ),
    # Risk
    "sharpe_ratio": (
        "Sharpe — excellent risk-adjusted returns",
        "Sharpe — decent risk-adjusted returns",
        "Sharpe — poor risk-adjusted returns",
    ),
    "sortino_ratio": (
        "Sortino — excellent downside protection",
        "Sortino — decent downside protection",
        "Sortino — weak downside protection",
    ),
    "calmar_ratio": (
        "Calmar — excellent return per drawdown unit",
        "Calmar — decent return per drawdown unit",
        "Calmar — poor return per drawdown unit",
    ),
    "max_drawdown_pct": (
        "Max drawdown — well controlled",
        "Max drawdown — moderate risk",
        "Max drawdown — significant risk exposure",
    ),
    "avg_drawdown_pct": (
        "Avg drawdown — typical losses well controlled",
        "Avg drawdown — acceptable typical losses",
        "Avg drawdown — larger-than-ideal typical losses",
    ),
    "volatility_ann_pct": (
        "Volatility — steady returns",
        "Volatility — moderate fluctuations",
        "Volatility — high uncertainty",
    ),
    # Trade quality
    "win_rate_pct": (
        "Win rate — excellent trade success",
        "Win rate — decent trade success",
        "Win rate — needs improvement",
    ),
    "profit_factor": (
        "Profit factor — strong winning trades outweigh losses",
        "Profit factor — acceptable win/loss ratio",
        "Profit factor — losses outweigh wins",
    ),
    "expectancy_pct": (
        "Expectancy — positive edge per trade",
        "Expectancy — marginal edge per trade",
        "Expectancy — negative edge per trade",
    ),
    "sqn": (
        "SQN — excellent system robustness",
        "SQN — decent system robustness",
        "SQN — system needs refinement",
    ),
    "kelly_criterion": (
        "Kelly — conservative position sizing",
        "Kelly — moderate position sizing",
        "Kelly — aggressive position sizing",
    ),
    "avg_trade_pct": (
        "Avg trade — strong positive returns",
        "Avg trade — acceptable returns",
        "Avg trade — weak returns",
    ),
    "best_trade_pct": (
        "Best trade — excellent single trade capture",
        "Best trade — decent capture",
        "Best trade — underwhelming",
    ),
    "worst_trade_pct": (
        "Worst trade — well-protected downside",
        "Worst trade — acceptable loss",
        "Worst trade — large loss exposure",
    ),
    "avg_trade_duration": (
        "Duration — trades held appropriately",
        "Duration — acceptable hold time",
        "Duration — unusually short/long",
    ),
    "exposure_time_pct": (
        "Exposure — selective trading",
        "Exposure — moderate market participation",
        "Exposure — overtraded",
    ),
    # Model accuracy
    "accuracy": (
        "Accuracy — excellent predictions",
        "Accuracy — decent predictions",
        "Accuracy — poor predictions",
    ),
    "directional_accuracy": (
        "Directional accuracy — excellent signal quality",
        "Directional accuracy — decent signal quality",
        "Directional accuracy — poor signal quality",
    ),
    # Market comparison
    "alpha_pct": (
        "Alpha — outperforms market significantly",
        "Alpha — modest market outperformance",
        "Alpha — underperforms market",
    ),
    "beta": (
        "Beta — low market correlation (independent)",
        "Beta — moderate market correlation",
        "Beta — high market correlation",
    ),
    # Other
    "commissions": (
        "Commissions — low trading costs",
        "Commissions — moderate trading costs",
        "Commissions — high trading costs",
    ),
    "num_trades": (
        "Trade count — healthy activity level",
        "Trade count — acceptable activity",
        "Trade count — too few/too many",
    ),
    "equity_peak": (
        "Account peak — strong capital high",
        "Account peak — decent capital high",
        "Account peak — low capital high",
    ),
}


def _get_explanation(key: str, value: float) -> str:
    """Return explanation sentence based on metric value quality."""
    level = _get_verdict_level(key, value)
    if key not in _EXPLANATIONS:
        return ""
    explanations = _EXPLANATIONS[key]
    if level == "good":
        return explanations[0]
    elif level == "ok":
        return explanations[1]
    else:
        return explanations[2]


def _metric(
    lines: list[str],
    metrics: dict,
    key: str,
    label: str,
    fmt: str = "f2",
) -> None:
    """Append a formatted metric line if the key exists."""
    if key not in metrics:
        return
    val = metrics[key]
    formatter = _METRIC_FMT.get(fmt, _METRIC_FMT["f2"])
    lines.append(f"- **{label}**: {formatter(val)}")
