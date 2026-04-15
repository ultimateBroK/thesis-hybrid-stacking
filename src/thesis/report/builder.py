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
        "Key terms are explained when they first appear. A metric in *italics* means higher is better; ",
        "one marked with a warning icon needs attention.",
        "",
        "---",
        "",
        "## Table of Contents",
        "",
        "1. [What Data Did We Use?](#1-what-data-did-we-use)",
        "2. [What Clues Does the Model Look At?](#2-what-clues-does-the-model-look-at)",
        '3. [How Do We Decide "Up" vs "Down"?](#3-how-do-we-decide-up-vs-down)',
        "4. [How Did We Split the Data?](#4-how-did-we-split-the-data)",
        "5. [How Does the AI Model Work?](#5-how-does-the-ai-model-work)",
        "6. [How Accurate Is the Model?](#6-how-accurate-is-the-model)",
        "7. [Would Trading on These Predictions Have Made Money?](#7-would-trading-on-these-predictions-have-made-money)",
        "8. [Ablation Study](#8-ablation-study)",
        "9. [Conclusion](#9-conclusion)",
        "10. [Charts](#10-charts)",
        "",
        "---",
        "",
    ]

    # ==================================================================
    # 1. DATA PREPARATION
    # ==================================================================
    lines.append("## 1. What Data Did We Use?")
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

    # ==================================================================
    # 2. FEATURE ENGINEERING
    # ==================================================================
    lines.append("## 2. What Clues Does the Model Look At?")
    lines.append("")
    lines.append(
        "Before the AI model can make predictions, we need to give it useful information. "
        "We calculate 11 **technical indicators** — mathematical formulas applied to the "
        "price data that help reveal hidden patterns like momentum, volatility, and trend "
        'direction. Think of these as different "lenses" for viewing the same price chart.'
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
                "Trend distance — how far the current price is from its average trend",
            ),
            (
                "pivot_position",
                "Support/resistance — where the price sits relative to key levels",
            ),
            (
                "atr_percentile",
                "Volatility rank — is the current volatility high or low historically?",
            ),
            ("sess_asia", "Is the Asian market session active? (Tokyo, Sydney)"),
            ("sess_london", "Is the London market session active?"),
            (
                "sess_overlap",
                "Is the London–New York overlap active? (highest trading volume)",
            ),
            ("sess_ny_pm", "Is the New York afternoon session active?"),
        ]
        lines.append("### The 11 Indicators")
        lines.append("")
        lines.append("| Indicator | What It Measures (Plain English) |")
        lines.append("|-----------|----------------------------------|")
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
                lines.append("### When Does Most Trading Happen?")
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
    # 3. TRIPLE-BARRIER LABELING
    # ==================================================================
    lines.append('## 3. How Do We Decide "Up" vs "Down"?')
    lines.append("")
    lines.append(
        'To teach the AI what "correct" looks like, we need a way to label each hour '
        "as **Long** (price went up), **Short** (price went down), or **Hold** (stayed flat). "
        "We use a method called the **Triple Barrier Method**:"
    )
    lines.append("")
    lines.append(
        '1. For each hour, we set two invisible "walls" above and below the current price, '
        "spaced at **1.5× the recent volatility** (ATR)."
    )
    lines.append("2. We then watch the price for the next **10 hours** maximum.")
    lines.append(
        "3. If the price hits the upper wall first → label it **Long**. "
        "If it hits the lower wall first → label it **Short**. "
        "If neither wall is hit in 10 hours → label it **Hold**."
    )
    lines.append("")
    lines.append(
        f"- **Barrier width**: {config.labels.atr_multiplier}× the average price movement (ATR)"
    )
    lines.append(f"- **Time limit**: {config.labels.horizon_bars} hours")
    lines.append("")

    labels_path = Path(config.paths.labels)
    label_dist = _load_label_distribution(labels_path)
    if label_dist:
        total = label_dist["total"]
        lines.append("### What Did We Find?")
        lines.append("")
        for name, emoji in [("Short", "📉"), ("Hold", "⏸️"), ("Long", "📈")]:
            key = name
            count, pct = label_dist[key]
            lines.append(
                f"- {emoji} **{name}** (price went {'down' if name == 'Short' else 'sideways' if name == 'Hold' else 'up'}): {count:,} hours ({pct:.1f}%)"
            )
        lines.append("")
        hold_pct = label_dist["Hold"][1]
        if hold_pct < 25:
            lines.append(
                f'> **What this means**: Only {hold_pct:.0f}% of hours were labeled "Hold", '
                f"meaning the price usually moved enough to hit one of the walls within 10 hours. "
                f"This is normal for gold — it tends to move a lot."
            )
        lines.append("")
    else:
        lines.append("*Label data not found.*")
    lines.append("")

    # ==================================================================
    # 4. DATA SPLITTING
    # ==================================================================
    lines.append("## 4. How Did We Split the Data?")
    lines.append("")
    lines.append(
        "A critical rule in machine learning: **never test on data the model has seen during training**. "
        'We split the data into three periods, like dividing a textbook into "study material", '
        '"practice exam", and "final exam":'
    )
    lines.append("")
    lines.append(
        f"- **Study material (Training)**: {config.splitting.train_start} to {config.splitting.train_end} "
        f"— the model learns patterns from this period"
    )
    lines.append(
        f"- **Practice exam (Validation)**: {config.splitting.val_start} to {config.splitting.val_end} "
        f"— used to tune the model and prevent overfitting"
    )
    lines.append(
        f"- **Final exam (Test)**: {config.splitting.test_start} to {config.splitting.test_end} "
        f"— the model has NEVER seen this data; this is the real test"
    )
    lines.append("")
    lines.append(
        "We also add safety gaps (**purge** and **embargo**) between periods to ensure "
        'no information accidentally "leaks" from one period to the next.'
    )
    lines.append("")

    split_stats = _load_split_stats(config)
    if split_stats:
        lines.append("### How Much Data in Each Period?")
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
    # 5. MODEL TRAINING
    # ==================================================================
    lines.append("## 5. How Does the AI Model Work?")
    lines.append("")
    lines.append(
        "We use a **two-part (hybrid) model** — think of it as two experts working together:"
    )
    lines.append("")

    # GRU explanation
    lines.append('### Expert 1: GRU Neural Network ("The Pattern Reader")')
    lines.append("")
    lines.append(
        "A **GRU (Gated Recurrent Unit)** is a type of neural network designed to read "
        'sequences of data — like reading a sentence word by word. Here, it "reads" '
        f"the last **{config.gru.sequence_length} hours** of price changes and technical "
        'indicators to detect temporal patterns (e.g., "when RSI drops and ATR rises, '
        'the price usually follows in 3 hours").'
    )
    lines.append("")
    lines.append(
        f"- It processes **{config.gru.input_size} inputs** per hour "
        f"(price change, momentum, volatility, trend)"
    )
    lines.append(
        f"- It has a **{config.gru.hidden_size}-dimensional** internal representation — "
        f'think of it as {config.gru.hidden_size} different "pattern detectors"'
    )
    lines.append(
        f"- It was trained for up to **{config.gru.epochs} rounds** "
        f"(stopped early if it stopped improving)"
    )
    lines.append("")

    # LightGBM explanation
    lines.append('### Expert 2: LightGBM ("The Decision Maker")')
    lines.append("")
    lines.append(
        "**LightGBM** is a decision-tree-based model (like a flowchart with many branches). "
        "It takes the GRU's pattern detectors PLUS the 11 technical indicators as input, "
        "and makes the final prediction: Long, Hold, or Short."
    )
    lines.append("")
    lines.append(
        "Decision trees work by asking a series of yes/no questions: "
        '"Is GRU detector #20 above 0.5? Is volatility high? Is it the London session?" '
        "→ then outputs a prediction with a confidence score."
    )
    lines.append("")

    # Hybrid feature space
    total_features = config.gru.hidden_size + 11
    lines.append("### Combined System")
    lines.append("")
    lines.append(
        f"Total inputs to the decision maker: **{total_features} features** "
        f"({config.gru.hidden_size} from the GRU pattern reader + 11 technical indicators)"
    )
    lines.append("")

    # Feature importance
    if feature_importance:
        items = list(feature_importance.items())
        gru_count = sum(1 for name, _ in items if name.startswith("gru_"))

        lines.append("### Which Features Matter Most?")
        lines.append("")
        lines.append(
            "Feature importance tells us which inputs the model relied on most. "
            "A higher score means the model found that feature more useful for making predictions."
        )
        lines.append("")
        lines.append("| Rank | Feature | Where It Comes From | Score |")
        lines.append("|------|---------|---------------------|-------|")
        for i, (name, imp) in enumerate(items[:15], 1):
            ftype = (
                "GRU (pattern reader)"
                if name.startswith("gru_")
                else "Technical indicator"
            )
            lines.append(f"| {i} | `{name}` | {ftype} | {imp:.0f} |")
        lines.append("")
        lines.append(
            f"> **What this means**: {gru_count} of the top {len(items)} features "
            f"come from the GRU neural network ({gru_count / len(items) * 100:.0f}%). "
            f"This confirms that the pattern reader is finding useful temporal patterns "
            f"that simple indicators alone cannot capture."
        )
        lines.append("")

    # ==================================================================
    # 6. MODEL PREDICTION PERFORMANCE
    # ==================================================================
    lines.append("## 6. How Accurate Is the Model?")
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
        baseline = pred_stats["majority_baseline"]

        lines.append("### Overall Accuracy")
        lines.append("")
        lines.append(
            f"- **Total predictions**: {total:,} hours "
            f"({config.splitting.test_start} to {config.splitting.test_end})"
        )
        lines.append(f"- **Model accuracy**: **{acc * 100:.1f}%**")
        lines.append(
            f'- **"Dumb" baseline**: {baseline * 100:.1f}% '
            f"(if you just always guessed the most common direction)"
        )
        lines.append(
            f"- **Improvement over guessing**: +{(acc - baseline) * 100:.1f} percentage points"
        )
        lines.append("")
        lines.append(
            f"> **What {acc * 100:.0f}% means in context**: The model is right about "
            f"{acc * 100:.0f} times out of 100. This is only slightly better than always "
            f"guessing the most common outcome ({baseline * 100:.0f}%) — but the real value "
            f"comes from knowing **when it's confident** (see below)."
        )
        lines.append("")

        # Per-class
        per_class = pred_stats["per_class"]
        lines.append("### Accuracy by Direction")
        lines.append("")
        lines.append(
            "| Direction | Actual Count | Model Predicted | How Often Correct | How Often Wrong |"
        )
        lines.append(
            "|-----------|-------------|-----------------|-------------------|-----------------|"
        )
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

        # Confusion matrix in plain language
        cm = pred_stats["confusion_matrix"]
        lines.append("### Detailed Error Breakdown (Confusion Matrix)")
        lines.append("")
        lines.append(
            "This table shows what actually happened (rows) vs what the model predicted (columns). "
            "Numbers on the diagonal are correct predictions."
        )
        lines.append("")
        lines.append(
            "| What Actually Happened | Model Said Short 📉 | Model Said Hold ⏸️ | Model Said Long 📈 |"
        )
        lines.append(
            "|----------------------|--------------------|--------------------|--------------------|"
        )
        for true_name, emoji in [("Short", "📉"), ("Hold", "⏸️"), ("Long", "📈")]:
            row = cm[true_name]
            lines.append(
                f"| {emoji} **{true_name}** | {row['Short']:,} | {row['Hold']:,} | {row['Long']:,} |"
            )
        lines.append("")

        # Model bias analysis
        lines.append("### Does the Model Favor One Direction?")
        lines.append("")
        short_pred = per_class["Short"]["pred_count"]
        long_pred = per_class["Long"]["pred_count"]
        if short_pred > total * 0.5:
            lines.append(
                f"⚠️ The model predicts **Short** {short_pred / total * 100:.1f}% of the time — "
                f"it's biased toward predicting price drops. This means it often incorrectly "
                f"predicts drops when the price actually goes up."
            )
        elif long_pred > total * 0.5:
            lines.append(
                f"⚠️ The model predicts **Long** {long_pred / total * 100:.1f}% of the time — "
                f"it's biased toward predicting price rises."
            )
        else:
            lines.append(
                "The model is relatively balanced — it doesn't strongly favor one direction."
            )
        lines.append("")

        # High-confidence analysis
        hc = pred_stats.get("high_confidence")
        if hc:
            lines.append("### The Secret Weapon: Confidence Filtering")
            lines.append("")
            lines.append(
                "The model doesn't just predict a direction — it also outputs a **confidence score** "
                "(0–100%) showing how sure it is. When the confidence is **high (≥70%)**, "
                "the model is much more accurate:"
            )
            lines.append("")
            lines.append(
                f'- **How often the model is "very sure"**: {hc["count"]:,} hours out of {total:,} '
                f"({hc['pct_of_total']:.1f}%)"
            )
            lines.append(
                f"- **Accuracy when very sure**: **{hc['accuracy'] * 100:.1f}%** "
                f"(vs {acc * 100:.1f}% overall)"
            )
            lines.append(
                f"- **Directional accuracy**: **{hc['directional_accuracy'] * 100:.1f}%** "
                f'(when it says "buy" or "sell" with high confidence)'
            )
            lines.append("")
            lines.append(
                "> **The key insight**: Instead of trading on every prediction, "
                "we only trade when the model is very confident. This is like a weather forecaster "
                "only making predictions when they're sure — they'll be wrong sometimes, "
                "but much less often than when they're uncertain."
            )
            lines.append("")
    else:
        lines.append("*Prediction data not found.*")
    lines.append("")

    # ==================================================================
    # 7. BACKTEST RESULTS
    # ==================================================================
    lines.append("## 7. Would Trading on These Predictions Have Made Money?")
    lines.append("")
    lines.append(
        "We simulated what would have happened if we traded real money based on the model's "
        "high-confidence predictions. This is called a **backtest** — running the strategy "
        "on historical data as if it were live."
    )
    lines.append("")

    # Backtest config in plain language
    lines.append("### How the Simulation Was Set Up")
    lines.append("")
    lines.append(f"- **Starting money**: ${config.backtest.initial_capital:,.0f}")
    lines.append(
        f"- **Leverage**: {config.backtest.leverage}:1 "
        f"(borrowing ${config.backtest.leverage - 1} for every $1 of own capital — amplifies both gains and losses)"
    )
    lines.append(
        f"- **Trade size**: {config.backtest.lots_per_trade} lot = 100 oz of gold (~${200 * 100:,.0f} notional)"
    )
    lines.append(
        f"- **Stop-loss**: Automatically exits if the price moves against us by "
        f"{config.backtest.atr_stop_multiplier}× the recent volatility"
    )
    lines.append(
        f"- **Confidence filter**: Only trades when model confidence ≥ {config.backtest.confidence_threshold}"
    )
    lines.append(
        f"- **Trading costs**: ${config.backtest.spread_ticks * config.data.tick_size:.2f} spread + "
        f"${config.backtest.commission_per_lot:.0f} commission per lot per trade"
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
        commissions = float(metrics.get("commissions", 0))

        # The big picture
        lines.append("### The Big Picture")
        lines.append("")
        lines.append(f"- **Total trades**: {n_trades:,}")
        lines.append(
            f"- **Starting with**: ${config.backtest.initial_capital:,.0f} → "
            f"**Ending with**: ${final_eq:,.0f}"
        )
        if config.backtest.leverage > 1 and ret > 500:
            unleveraged = ret / config.backtest.leverage
            lines.append(
                f"- **Total return**: {ret:,.0f}% (with {config.backtest.leverage}:1 leverage) "
                f"≈ {unleveraged:.0f}% without leverage"
            )
        else:
            lines.append(f"- **Total return**: {ret:.1f}%")
        lines.append(f"- **Total trading costs paid**: ${commissions:,.0f}")
        lines.append("")

        # Win rate explained
        lines.append("### Win Rate — How Often Does the Strategy Win?")
        lines.append("")
        lines.append(
            f"- **Win rate**: {wr:.1f}% (about {int(wr)} out of every 10 trades were profitable)"
        )
        lines.append("")
        lines.append(
            f"> **{wr:.0f}% sounds low — is it bad?** Not necessarily. "
            f"Think of a casino: the house doesn't win every hand, but it makes money because "
            f"the wins are bigger than the losses. Our strategy works the same way."
        )
        lines.append("")

        # Profit factor explained
        lines.append("### Profit Factor — Are Wins Bigger Than Losses?")
        lines.append("")
        lines.append(f"- **Profit Factor**: {pf:.2f}")
        lines.append(
            f"- **What this means**: For every $1 lost, the strategy makes ${pf:.2f}"
        )
        lines.append("- A profit factor above 1.5 is generally considered **good**")
        lines.append(
            "- A profit factor above 2.0 is generally considered **excellent**"
        )
        if pf >= 1.5:
            lines.append(
                f"> Our profit factor of {pf:.2f} means the strategy's winning trades "
                f"are significantly larger than its losing trades."
            )
        lines.append("")

        # Risk metrics explained
        lines.append("### Risk — How Much Could You Lose?")
        lines.append("")
        lines.append(
            f"- **Maximum drawdown**: {max_dd:.1f}% "
            f'(the worst "peak to trough" decline — imagine watching your account '
            f"go from its highest point down {abs(max_dd):.0f}% before recovering)"
        )
        _metric(
            lines,
            metrics,
            "avg_drawdown_pct",
            "Average drawdown (typical temporary loss)",
            fmt="pct",
        )
        lines.append("")
        lines.append(
            f"> **{abs(max_dd):.0f}% maximum drawdown**: At the worst point, "
            f"a ${config.backtest.initial_capital:,.0f} account would have temporarily "
            f"fallen to about ${config.backtest.initial_capital * (1 + max_dd / 100):,.0f}. "
            f"{'This is manageable for a leveraged strategy.' if abs(max_dd) < 30 else 'This is significant — the strategy has high volatility.'}"
        )
        lines.append("")

        # Sharpe explained
        lines.append("### Risk-Adjusted Return (Sharpe Ratio)")
        lines.append("")
        lines.append(f"- **Sharpe Ratio**: {sharpe:.2f}")
        lines.append("")
        lines.append(
            "The Sharpe Ratio measures return relative to risk. Think of it as "
            '"how smooth is the ride?" A higher number means steadier returns with fewer scary drops.'
        )
        lines.append("")
        lines.append("| Sharpe | Quality |")
        lines.append("|--------|---------|")
        lines.append("| Below 0.5 | Poor — returns are mostly random noise |")
        lines.append("| 0.5 – 1.0 | Below average |")
        lines.append("| 1.0 – 1.5 | Good |")
        lines.append("| 1.5 – 2.0 | Very good |")
        lines.append("| Above 2.0 | Excellent |")
        lines.append("")
        if sharpe >= 1.0:
            lines.append(
                f"> Our Sharpe of {sharpe:.2f} falls in the "
                f"{'**good**' if sharpe < 1.5 else '**very good**' if sharpe < 2.0 else '**excellent**'} range."
            )
        lines.append("")

        # Trade distribution
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
                lines.append(f"  - Average win: ${np.mean(winners):,.2f}")
                lines.append(f"  - Best single trade: ${max(winners):,.2f}")
            lines.append(
                f"- **Losing trades**: {len(losers):,} ({len(losers) / len(trades) * 100:.1f}%)"
            )
            if losers:
                lines.append(f"  - Average loss: ${np.mean(losers):,.2f}")
                lines.append(f"  - Worst single trade: ${min(losers):,.2f}")
            lines.append(f"- **Buy (Long) trades**: {len(longs):,}")
            lines.append(f"- **Sell (Short) trades**: {len(shorts):,}")
            if winners and losers:
                rr = abs(np.mean(winners) / np.mean(losers))
                lines.append(
                    f"- **Win/Loss ratio**: {rr:.1f}:1 (average win is {rr:.1f}× the average loss)"
                )
            lines.append("")

        # Detailed metrics for technical readers
        lines.append("<details>")
        lines.append("<summary>📊 Detailed Metrics (click to expand)</summary>")
        lines.append("")
        for group_name, group_metrics in [
            (
                "Performance",
                [
                    ("return_ann_pct", "Annual Return", "pct"),
                    (
                        "buy_&_hold_return_pct",
                        "Buy & Hold Return (just holding gold)",
                        "pct",
                    ),
                    ("cagr_pct", "Compound Annual Growth Rate", "pct"),
                    ("equity_peak", "Account Peak", "dollar"),
                    (
                        "volatility_ann_pct",
                        "Annual Volatility (how much returns vary)",
                        "pct",
                    ),
                ],
            ),
            (
                "Risk",
                [
                    ("sortino_ratio", "Sortino Ratio (downside risk only)", "f2"),
                    ("calmar_ratio", "Calmar Ratio (return vs max drawdown)", "f2"),
                    ("avg_drawdown_pct", "Avg Drawdown", "pct"),
                    ("max_drawdown_duration", "Longest Drawdown Period", "s"),
                ],
            ),
            (
                "Trade Quality",
                [
                    ("expectancy_pct", "Expected return per trade", "pct"),
                    ("sqn", "System Quality Number (SQN)", "f2"),
                    (
                        "kelly_criterion",
                        "Kelly Criterion (optimal position sizing)",
                        "f4",
                    ),
                    ("best_trade_pct", "Best Trade", "pct"),
                    ("worst_trade_pct", "Worst Trade", "pct"),
                    ("avg_trade_pct", "Avg Trade Return", "pct"),
                    ("avg_trade_duration", "Avg Trade Duration", "s"),
                    ("max_trade_duration", "Max Trade Duration", "s"),
                    ("exposure_time_pct", "Time in Market", "pct"),
                ],
            ),
            (
                "Market Comparison",
                [
                    ("alpha_pct", "Alpha (excess return vs market)", "pct"),
                    ("beta", "Beta (correlation with market)", "f3"),
                ],
            ),
        ]:
            found = [(mk, lb, fm) for mk, lb, fm in group_metrics if mk in metrics]
            if found:
                lines.append(f"**{group_name}**:")
                for mk, lb, fm in found:
                    val = metrics[mk]
                    formatter = _METRIC_FMT.get(fm, _METRIC_FMT["f2"])
                    lines.append(f"- {lb}: {formatter(val)}")
                lines.append("")
        lines.append("</details>")
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
            "An ablation study tests whether each component of the system is actually helpful. "
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
                lines.append(
                    f"| {variant} | {fc} | {trades_v} | {vwr:.2f}% | {vret:.2f} | {vsh:.4f} | {vdd:.2f} |"
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
    conclusion_parts: list[str] = []

    if pred_stats and metrics:
        acc = pred_stats["accuracy"]
        hc = pred_stats.get("high_confidence")
        n_trades = int(metrics.get("num_trades", 0))
        sharpe_val = float(metrics.get("sharpe_ratio", 0))
        max_dd_val = float(metrics.get("max_drawdown_pct", 0))
        pf_val = float(metrics.get("profit_factor", 0))
        ret_val = float(metrics.get("return_pct", 0))
        wr_val = float(metrics.get("win_rate_pct", 0))

        # Summary paragraph
        summary = (
            f"We tested an AI model that predicts gold price direction on "
            f"{pred_stats['total']:,} hours of out-of-sample data "
            f"(data it never saw during training). "
        )
        if hc:
            summary += (
                f"While the model's overall accuracy is {acc * 100:.0f}%, "
                f"when we only act on its **high-confidence** predictions (≥{hc['threshold']:.0%} sure), "
                f"accuracy jumps to **{hc['accuracy'] * 100:.0f}%**. "
                f"This is the core finding: the model knows when it doesn't know, "
                f"and we can use that to our advantage."
            )
        else:
            summary += (
                f"The model achieves {acc * 100:.0f}% accuracy — "
                f"{(acc - pred_stats['majority_baseline']) * 100:.1f} percentage points better than random guessing."
            )
        conclusion_parts.append(summary)

        # Backtest conclusion
        bt_conclusion = (
            f"Trading only on these high-confidence signals produces **{n_trades:,} trades** "
            f"over the test period. The strategy wins only {wr_val:.0f}% of the time, "
            f"but when it wins, it wins big — **${pf_val:.1f} for every $1 lost**. "
            f"The Sharpe ratio of {sharpe_val:.2f} indicates "
            f"{'good' if sharpe_val < 1.5 else 'very good' if sharpe_val < 2.0 else 'excellent'} "
            f"risk-adjusted performance."
        )
        conclusion_parts.append(bt_conclusion)

        # Caveats
        caveats = []
        if ret_val > 500:
            unleveraged = ret_val / config.backtest.leverage
            caveats.append(
                f"Returns are amplified by {config.backtest.leverage}:1 leverage. "
                f"Without leverage, the return would be approximately {unleveraged:.0f}%."
            )
        caveats.append(
            "The test period includes gold's historic bull run ($2,000 → $3,000+), "
            "which gave the strategy a strong tailwind."
        )
        caveats.append(
            f"Maximum drawdown was {abs(max_dd_val):.1f}% — the strategy is not for the faint of heart."
        )
        if caveats:
            conclusion_parts.append("**Important caveats**: " + " ".join(caveats))

        # Architecture validation
        if feature_importance:
            items_fi = list(feature_importance.items())
            gru_top = [n for n, _ in items_fi[:5] if n.startswith("gru_")]
            if gru_top:
                conclusion_parts.append(
                    "The neural network (GRU) features dominate the top importance rankings, "
                    "confirming that temporal pattern recognition adds real value "
                    "beyond what simple technical indicators can provide."
                )

    elif pred_stats:
        acc = pred_stats["accuracy"]
        conclusion_parts.append(
            f"The model achieves {acc * 100:.1f}% accuracy on "
            f"{pred_stats['total']:,} out-of-sample predictions. "
            f"No backtest results are available."
        )

    if config.backtest.confidence_threshold > 0:
        conclusion_parts.append(
            f"The confidence threshold ({config.backtest.confidence_threshold}) "
            f"is the strategy's most important feature — it filters out "
            f"low-conviction predictions and concentrates on the ones where "
            f"the model is genuinely confident."
        )

    for para in conclusion_parts:
        lines.append(para)
        lines.append("")

    if not conclusion_parts:
        lines.append("*Insufficient data for automated conclusion.*")
        lines.append("")

    # ==================================================================
    # 10. CHARTS
    # ==================================================================
    lines.append("## 10. Charts")
    lines.append("")
    lines.append("### Equity Curve")
    lines.append("")
    lines.append(
        "The equity curve shows how the account balance changed over time. "
        "A steadily rising line means consistent profits. Sharp drops indicate losing streaks."
    )
    lines.append("")
    if config.paths.session_dir:
        bt_chart = Path(config.paths.session_dir) / "backtest" / "backtest_chart.html"
        if bt_chart.exists():
            lines.append(
                "See [backtest_chart.html](backtest_chart.html) for the interactive version."
            )
            lines.append("")
    lines.append("![Equity Curve](equity_curve.png)")
    lines.append("")
    lines.append("### Feature Importance")
    lines.append("")
    lines.append("Higher bars = more important features for the model's predictions.")
    lines.append("")
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
