"""Metric zone classification for backtest benchmarks.

Pure Python — no Streamlit dependency. Can be unit-tested independently.
"""

import math


def is_extreme_value(metric_name: str, value: float) -> tuple[bool, float]:
    """Check if metric value is extreme."""
    thresholds = {
        "recovery_factor": 20.0,
        "sharpe_ratio": 10.0,
        "sortino_ratio": 20.0,
        "calmar_ratio": 15.0,
        "profit_factor": 10.0,
        "sqn": 5.0,
        "kelly_criterion": 0.8,
        "return_pct": 1000.0,
        "cagr_pct": 500.0,
        "return_ann_pct": 500.0,
    }
    threshold = thresholds.get(metric_name, float("inf"))
    return value > threshold, threshold


def get_metric_zone(metric_name: str, value: float) -> tuple[str, str, str]:
    """Return (color, zone_label, recommendation) for a metric value."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ("moderate", "N/A", "No data available")

    is_extreme, threshold = is_extreme_value(metric_name, value)
    if is_extreme:
        return (
            "dangerous",
            "Extreme",
            f"Value {value:.1f} exceeds threshold {threshold:.1f} "
            "— verify for overfitting/data issues",
        )

    # ---- helper bands ----
    def band(val, *ranges):
        for lo, hi, color, label, rec in ranges:
            if lo <= val < hi:
                return color, label, rec
        return ranges[-1][2:]

    # ---- per-metric rules ----
    if metric_name == "sharpe_ratio":
        return band(
            value,
            (0, 0.5, "dangerous", "Poor", "<0.5 — high risk-adjusted cost"),
            (
                0.5,
                1.0,
                "moderate",
                "Acceptable",
                "0.5-1.0 — acceptable risk-adjusted returns",
            ),
            (1.0, 2.0, "good", "Good", "1.0-2.0 — solid risk-adjusted returns"),
            (
                2.0,
                3.0,
                "excellent",
                "Excellent",
                "2.0-3.0 — hedge fund target (verify no overfitting)",
            ),
            (
                3.0,
                float("inf"),
                "dangerous",
                "Suspicious",
                ">3.0 — verify no overfitting",
            ),
        )

    if metric_name == "sortino_ratio":
        return band(
            value,
            (0, 0.5, "dangerous", "Poor", "<0.5 — excessive downside risk"),
            (
                0.5,
                1.5,
                "moderate",
                "Acceptable",
                "0.5-1.5 — acceptable downside-adjusted returns",
            ),
            (1.5, 2.5, "good", "Good", "1.5-2.5 — solid downside-adjusted returns"),
            (2.5, 4.0, "excellent", "Excellent", "2.5-4.0 — very good"),
            (
                4.0,
                float("inf"),
                "excellent",
                "Exceptional",
                ">4.0 — exceptional downside protection",
            ),
        )

    if metric_name == "max_drawdown_pct":
        return band(
            value,
            (
                float("-inf"),
                -50,
                "dangerous",
                "Critical",
                ">50% — aggressive, question viability",
            ),
            (-50, -35, "poor", "Significant", "35-50% — high, assess suitability"),
            (
                -35,
                -20,
                "moderate",
                "Moderate",
                "20-35% — typical for volatile instruments",
            ),
            (-20, -10, "good", "Good", "10-20% — conservative drawdown"),
            (
                -10,
                float("inf"),
                "excellent",
                "Excellent",
                "<10% — exceptional capital preservation",
            ),
        )

    if metric_name == "profit_factor":
        return band(
            value,
            (0, 1.0, "dangerous", "Losing", "<1.0 — strategy loses money"),
            (1.0, 1.2, "poor", "Marginal", "1.0-1.2 — barely covers costs"),
            (1.2, 1.5, "moderate", "Acceptable", "1.2-1.5 — covers costs with margin"),
            (1.5, 2.0, "good", "Good", "1.5-2.0 — strong profitability"),
            (2.0, 3.0, "excellent", "Excellent", "2.0-3.0 — very efficient"),
            (
                3.0,
                float("inf"),
                "dangerous",
                "Suspicious",
                ">3.0 — verify no overfitting",
            ),
        )

    if metric_name == "win_rate_pct":
        return band(
            value,
            (0, 35, "poor", "Low", "<35% — requires large risk/reward ratio"),
            (35, 45, "moderate", "Acceptable", "35-45% — typical for trend-following"),
            (45, 55, "good", "Good", "45-55% — solid win rate"),
            (55, 65, "excellent", "Excellent", "55-65% — strong (verify if >65%)"),
            (
                65,
                float("inf"),
                "dangerous",
                "Suspicious",
                ">65% — verify no overfitting",
            ),
        )

    if metric_name in ("cagr_pct", "return_ann_pct"):
        return band(
            value,
            (
                float("-inf"),
                0,
                "dangerous",
                "Negative",
                "Negative returns — strategy losing money",
            ),
            (0, 5, "poor", "Very Low", "<5% — underperforms inflation"),
            (5, 15, "moderate", "Conservative", "5-15% — conservative but acceptable"),
            (15, 30, "good", "Strong", "15-30% — strong risk-adjusted returns"),
            (30, 50, "excellent", "Excellent", "30-50% — exceptional performance"),
            (
                50,
                float("inf"),
                "dangerous",
                "Suspicious",
                ">50% — verify for overfitting",
            ),
        )

    if metric_name == "return_pct":
        return band(
            value,
            (float("-inf"), 0, "dangerous", "Loss", "Negative returns — capital loss"),
            (0, 50, "poor", "Low", "<50% — minimal growth over period"),
            (50, 100, "moderate", "Moderate", "50-100% — doubled capital at best"),
            (100, 200, "good", "Good", "100-200% — solid growth"),
            (200, 500, "excellent", "Strong", "200-500% — strong performance"),
            (
                500,
                float("inf"),
                "dangerous",
                "Extreme",
                ">500% — verify for data issues",
            ),
        )

    if metric_name == "num_trades":
        return band(
            value,
            (0, 30, "poor", "Small Sample", "<30 trades — statistically weak"),
            (30, 100, "moderate", "Limited", "30-100 trades — use caution"),
            (100, 500, "good", "Useful", "100-500 trades — useful sample size"),
            (
                500,
                float("inf"),
                "excellent",
                "Robust",
                "≥500 trades — robust backtest sample",
            ),
        )

    if metric_name == "calmar_ratio":
        return band(
            value,
            (
                float("-inf"),
                0,
                "dangerous",
                "Negative",
                "Negative — losses exceed returns",
            ),
            (0, 0.5, "poor", "Weak", "<0.5 — risk outweighs reward"),
            (
                0.5,
                1.0,
                "moderate",
                "Acceptable",
                "0.5-1.0 — minimum acceptable threshold",
            ),
            (1.0, 2.0, "good", "Good", "1.0-2.0 — healthy risk/reward balance"),
            (
                2.0,
                3.0,
                "excellent",
                "Excellent",
                "2.0-3.0 — very strong risk-adjusted returns",
            ),
            (
                3.0,
                float("inf"),
                "excellent",
                "Exceptional",
                ">3.0 — exceptional risk/reward",
            ),
        )

    if metric_name == "sqn":
        return band(
            value,
            (0, 1.0, "poor", "Poor", "<1.0 — system has no edge"),
            (1.0, 1.5, "moderate", "Average", "1.0-1.5 — acceptable system quality"),
            (1.5, 2.0, "moderate", "Average", "1.5-2.0 — acceptable system"),
            (2.0, 3.0, "good", "Good", "2.0-3.0 — good system quality"),
            (3.0, float("inf"), "excellent", "Excellent", ">3.0 — excellent system"),
        )

    if metric_name == "exposure_time_pct":
        return band(
            value,
            (0, 15, "poor", "Too Selective", "<15% — may miss opportunities"),
            (15, 30, "moderate", "Low", "15-30% — conservative exposure"),
            (30, 60, "good", "Good", "30-60% — typical market exposure"),
            (60, 80, "moderate", "High", "60-80% — significant market commitment"),
            (80, float("inf"), "poor", "Overexposed", ">80% — almost always in trade"),
        )

    if metric_name == "kelly_criterion":
        return band(
            value,
            (float("-inf"), 0, "dangerous", "Invalid", "0 or negative — no edge"),
            (
                0,
                0.15,
                "moderate",
                "Conservative",
                "<15% — conservative position sizing",
            ),
            (0.15, 0.25, "good", "Optimal", "15-25% — textbook optimal sizing"),
            (0.25, 0.4, "moderate", "Aggressive", "25-40% — aggressive, high variance"),
            (
                0.4,
                float("inf"),
                "dangerous",
                "Very Aggressive",
                ">40% — very aggressive, high risk",
            ),
        )

    if metric_name == "recovery_factor":
        return band(
            value,
            (0, 1.0, "dangerous", "Bad", "<1.0 — never recovered worst loss"),
            (1.0, 2.0, "poor", "Weak", "1.0-2.0 — slow recovery"),
            (2.0, 4.0, "good", "Good", "2.0-4.0 — reasonable recovery"),
            (
                4.0,
                float("inf"),
                "excellent",
                "Excellent",
                ">4.0 — quick recovery from drawdowns",
            ),
        )

    if metric_name == "volatility_ann_pct":
        return band(
            value,
            (0, 10, "excellent", "Low", "<10% — very stable"),
            (10, 20, "good", "Moderate", "10-20% — acceptable range"),
            (20, 35, "moderate", "High", "20-35% — elevated risk"),
            (35, float("inf"), "poor", "Very High", ">35% — excessive volatility"),
        )

    if metric_name == "avg_win":
        return band(
            value,
            (
                0,
                50,
                "poor",
                "Low",
                "<1% of initial capital — small wins, may not cover costs",
            ),
            (
                50,
                200,
                "moderate",
                "Moderate",
                "1-4% of initial capital — decent win size",
            ),
            (
                200,
                500,
                "good",
                "Good",
                "4-10% of initial capital — strong average wins",
            ),
            (
                500,
                float("inf"),
                "excellent",
                "High",
                ">10% of initial capital — excellent win size",
            ),
        )

    if metric_name == "avg_loss":
        v = abs(value)
        return band(
            v,
            (
                0,
                50,
                "excellent",
                "Low",
                "<1% of initial capital — excellent risk control",
            ),
            (
                50,
                200,
                "good",
                "Moderate",
                "1-4% of initial capital — reasonable losses",
            ),
            (
                200,
                500,
                "moderate",
                "High",
                "4-10% of initial capital — large average losses",
            ),
            (
                500,
                float("inf"),
                "poor",
                "Severe",
                ">10% of initial capital — concerning loss size",
            ),
        )

    if metric_name == "equity_final":
        return ("moderate", "Absolute", "Absolute value — compare to initial capital")

    if metric_name == "equity_peak":
        return ("moderate", "Peak", "Peak equity reached")

    if metric_name == "commissions":
        return (
            "moderate",
            "Cost",
            "Compare to total return — should be <5% of profits",
        )

    if metric_name == "avg_trade_pct":
        return band(
            value,
            (float("-inf"), 0, "poor", "Negative", "<0% — average trade loses money"),
            (0, 0.3, "moderate", "Low", "0-0.3% — small per-trade edge"),
            (0.3, 1.0, "good", "Good", "0.3-1% — solid average"),
            (
                1.0,
                float("inf"),
                "excellent",
                "Excellent",
                ">1% — strong per-trade returns",
            ),
        )

    if metric_name == "best_trade_pct":
        return band(
            value,
            (0, 0.5, "poor", "Weak", "<0.5% — small best trade, limited upside"),
            (0.5, 1.5, "moderate", "Moderate", "0.5-1.5% — decent single trade"),
            (1.5, 3.0, "good", "Strong", "1.5-3.0% — strong best trade"),
            (3.0, 5.0, "excellent", "Excellent", "3.0-5.0% — exceptional single trade"),
            (
                5.0,
                float("inf"),
                "dangerous",
                "Suspicious",
                ">5.0% — verify for data errors",
            ),
        )

    if metric_name == "worst_trade_pct":
        return band(
            value,
            (
                float("-inf"),
                -5.0,
                "dangerous",
                "Dangerous",
                "<-5% — catastrophic risk management",
            ),
            (-5.0, -3.0, "poor", "Poor", "-3% to -5% — large single loss"),
            (-3.0, -1.0, "moderate", "Moderate", "-1% to -3% — acceptable"),
            (-1.0, float("inf"), "good", "Good", ">-1% — manageable worst case"),
        )

    if metric_name == "risk_reward_ratio":
        return band(
            value,
            (0, 1.0, "moderate", "Fair", "1.0-1.5 — marginal edge"),
            (1.0, 1.5, "moderate", "Fair", "1.0-1.5 — marginal edge"),
            (1.5, 2.0, "good", "Good", "1.5-2.0 — solid R/R"),
            (2.0, float("inf"), "excellent", "Excellent", "≥2.0 — strong R/R"),
        )

    if metric_name == "accuracy":
        return band(
            value,
            (0, 0.50, "poor", "Poor", "<50% — no predictive edge"),
            (0.50, 0.55, "moderate", "Moderate", "50-55% — near random, marginal edge"),
            (
                0.55,
                float("inf"),
                "excellent",
                "Excellent",
                "≥55% — very accurate predictions",
            ),
        )

    if metric_name == "directional_accuracy":
        return band(
            value,
            (0, 0.55, "poor", "Poor", "<55% — weak directional predictions"),
            (
                0.55,
                0.60,
                "moderate",
                "Moderate",
                "55-60% — decent directional accuracy",
            ),
            (
                0.60,
                float("inf"),
                "excellent",
                "Excellent",
                "≥60% — strong directional accuracy",
            ),
        )

    if metric_name == "expectancy_pct":
        return band(
            value,
            (
                float("-inf"),
                0,
                "poor",
                "Negative",
                "<0% — negative expected value per trade",
            ),
            (0, 0.5, "moderate", "Low", "0-0.5% — small per-trade edge"),
            (0.5, 1.0, "good", "Good", "0.5-1% — solid per-trade edge"),
            (
                1.0,
                float("inf"),
                "excellent",
                "Excellent",
                "≥1% — strong per-trade edge",
            ),
        )

    if metric_name == "avg_drawdown_pct":
        return band(
            value,
            (float("-inf"), -15, "poor", "High", "≤-15% — high average drawdown"),
            (
                -15,
                -10,
                "moderate",
                "Moderate",
                "-10% to -15% — typical average drawdown",
            ),
            (-10, -5, "good", "Good", "-5% to -10% — conservative"),
            (-5, float("inf"), "excellent", "Excellent", ">-5% — very stable"),
        )

    return ("moderate", "Neutral", "No benchmark available")


ZONE_COLORS = {
    "excellent": "#22c55e",
    "good": "#84cc16",
    "moderate": "#eab308",
    "poor": "#f97316",
    "dangerous": "#ef4444",
}
