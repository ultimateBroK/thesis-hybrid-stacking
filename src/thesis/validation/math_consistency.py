"""Validate that backtest returns match model accuracy expectations.

This module implements mathematical consistency checks to detect impossible
performance claims. If a system claims 1,987% returns with 53% accuracy,
this validator will flag it as mathematically inconsistent.
"""

import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger("thesis.validation")


@dataclass
class TradingMathValidator:
    """Validate trading performance against mathematical expectations."""

    accuracy: float
    win_rate: float
    avg_win: float
    avg_loss: float
    num_trades: int
    risk_per_trade: float

    def expected_return(self) -> float:
        """Calculate expected return based on win rate and R-multiples.

        Returns:
            Expected total return as decimal (e.g., 0.50 for 50%)
        """
        win_prob = self.win_rate
        loss_prob = 1 - win_prob

        # Expected value per trade (as decimal)
        ev_per_trade = win_prob * self.avg_win - loss_prob * self.avg_loss

        # Total expected return
        total_return = ev_per_trade * self.num_trades
        return total_return

    def kelly_fraction(self) -> float:
        """Calculate optimal Kelly Criterion fraction.

        Returns:
            Kelly fraction (0.0 to 1.0), or 0 if edge is negative
        """
        win_prob = self.win_rate
        loss_prob = 1 - win_prob

        if self.avg_loss == 0:
            return 0.0

        # Kelly formula: (win_prob * avg_win - loss_prob * avg_loss) / avg_win
        # Simplified for R-multiples: edge / avg_win
        edge = win_prob * self.avg_win - loss_prob * self.avg_loss

        if edge <= 0:
            return 0.0

        kelly = edge / self.avg_win
        return max(0.0, min(1.0, kelly))  # Clamp between 0 and 1

    def validate_consistency(self, actual_return: float) -> dict:
        """Check if actual return is within statistical bounds.

        Args:
            actual_return: Actual total return as decimal (e.g., 19.87 for 1987%)

        Returns:
            Dictionary with validation results
        """
        expected = self.expected_return()

        # Calculate variance (simplified model)
        # Assumes independent trades with binary outcomes
        win_prob = self.win_rate
        loss_prob = 1 - self.win_rate

        variance_per_trade = (
            (
                win_prob * (self.avg_win - expected / self.num_trades) ** 2
                + loss_prob * (self.avg_loss - expected / self.num_trades) ** 2
            )
            if self.num_trades > 0
            else 0
        )

        variance = self.num_trades * variance_per_trade
        std_dev = np.sqrt(variance) if variance > 0 else 1e-10

        # Sigma bounds
        sigma_1_lower = expected - 1 * std_dev
        sigma_1_upper = expected + 1 * std_dev
        sigma_2_lower = expected - 2 * std_dev
        sigma_2_upper = expected + 2 * std_dev
        sigma_3_lower = expected - 3 * std_dev
        sigma_3_upper = expected + 3 * std_dev

        # Determine bounds violation
        within_1_sigma = sigma_1_lower <= actual_return <= sigma_1_upper
        within_2_sigma = sigma_2_lower <= actual_return <= sigma_2_upper
        within_3_sigma = sigma_3_lower <= actual_return <= sigma_3_upper

        # Calculate sigma deviation
        sigma_deviation = (
            (actual_return - expected) / std_dev if std_dev > 0 else float("inf")
        )

        # Red flag criteria
        # If actual is >2× the 3-sigma upper bound, it's physically impossible
        red_flag = actual_return > sigma_3_upper * 2 or sigma_deviation > 6

        return {
            "expected": expected,
            "actual": actual_return,
            "std_dev": std_dev,
            "sigma_deviation": sigma_deviation,
            "within_1_sigma": within_1_sigma,
            "within_2_sigma": within_2_sigma,
            "within_3_sigma": within_3_sigma,
            "sigma_3_lower": sigma_3_lower,
            "sigma_3_upper": sigma_3_upper,
            "red_flag": red_flag,
            "warning": not within_3_sigma and not red_flag,
        }

    def validate_calmar(self, calmar_ratio: float, max_drawdown: float) -> dict:
        """Validate Calmar ratio is within realistic bounds.

        Args:
            calmar_ratio: Calmar ratio (return / max_drawdown)
            max_drawdown: Maximum drawdown as decimal

        Returns:
            Dictionary with validation results
        """
        # Industry benchmarks
        retail_benchmark = 1.5
        professional_benchmark = 3.0
        impossible_threshold = 10.0

        result = {
            "calmar": calmar_ratio,
            "max_drawdown": max_drawdown,
            "retail_benchmark": retail_benchmark,
            "professional_benchmark": professional_benchmark,
            "impossible_threshold": impossible_threshold,
        }

        if calmar_ratio > impossible_threshold:
            result["status"] = "IMPOSSIBLE"
            result["passed"] = False
            result["message"] = (
                f"Calmar ratio {calmar_ratio:.2f} is impossible (> {impossible_threshold}). "
                f"Best real systems (Renaissance) achieve ~3-5. "
                f"This indicates curve-fitting or data leakage."
            )
        elif calmar_ratio > professional_benchmark:
            result["status"] = "EXCEPTIONAL"
            result["passed"] = True
            result["warning"] = True
            result["message"] = (
                f"Calmar ratio {calmar_ratio:.2f} exceeds professional benchmark "
                f"({professional_benchmark}). Requires extreme scrutiny."
            )
        elif calmar_ratio > retail_benchmark:
            result["status"] = "GOOD"
            result["passed"] = True
            result["warning"] = False
            result["message"] = (
                f"Calmar ratio {calmar_ratio:.2f} is good (> {retail_benchmark})."
            )
        else:
            result["status"] = "ACCEPTABLE"
            result["passed"] = True
            result["warning"] = False
            result["message"] = f"Calmar ratio {calmar_ratio:.2f} is acceptable."

        return result


def validate_backtest_math(
    backtest_results: dict,
    model_accuracy: float | None = None,
    risk_per_trade: float = 0.01,
) -> dict:
    """Validate backtest results for mathematical consistency.

    Args:
        backtest_results: Dictionary with backtest metrics
        model_accuracy: Model validation accuracy (if known)
        risk_per_trade: Risk per trade as decimal

    Returns:
        Comprehensive validation results
    """
    # Extract metrics from backtest results (handle nested structure)
    metrics = backtest_results.get("metrics", backtest_results)
    total_return = (
        metrics.get("total_return", metrics.get("total_return_pct", 0)) / 100
    )  # Convert from %
    win_rate = metrics.get("win_rate", 0)
    num_trades = metrics.get("num_trades", metrics.get("total_trades", 0))
    calmar = metrics.get("calmar_ratio", 0)
    max_drawdown = (
        metrics.get("max_drawdown", metrics.get("max_drawdown_pct", 0)) / 100
    )  # Convert from %

    # Estimate avg win/loss from backtest
    avg_winner = metrics.get("avg_winner", metrics.get("avg_win_dollar", 0.01))
    avg_loser = metrics.get("avg_loser", metrics.get("avg_loss_dollar", 0.01))

    # If we have per-trade data, calculate more precisely
    if (
        "trades" in backtest_results
        and isinstance(backtest_results["trades"], list)
        and backtest_results["trades"]
    ):
        trades = backtest_results["trades"]
        winners = [t for t in trades if t.get("pnl", 0) > 0]
        losers = [t for t in trades if t.get("pnl", 0) <= 0]

        if winners:
            avg_win = np.mean([t["pnl"] / t.get("entry_price", 1) for t in winners])
        else:
            avg_win = 0.01

        if losers:
            avg_loss = np.mean(
                [abs(t["pnl"]) / t.get("entry_price", 1) for t in losers]
            )
        else:
            avg_loss = 0.01
    else:
        avg_win = avg_winner
        avg_loss = avg_loser

    # Create validator
    validator = TradingMathValidator(
        accuracy=model_accuracy or win_rate,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        num_trades=num_trades,
        risk_per_trade=risk_per_trade,
    )

    # Run validations
    return_validation = validator.validate_consistency(total_return)
    calmar_validation = validator.validate_calmar(calmar, max_drawdown)

    # Compile results
    results = {
        "return_validation": return_validation,
        "calmar_validation": calmar_validation,
        "kelly_fraction": validator.kelly_fraction(),
        "expected_return": validator.expected_return(),
        "passed": not return_validation["red_flag"] and calmar_validation["passed"],
    }

    # Add warnings
    warnings = []
    if return_validation["red_flag"]:
        warnings.append(
            f"RED FLAG: Actual return {total_return:.1%} is {return_validation['sigma_deviation']:.1f}σ "
            f"above expected ({return_validation['expected']:.1%}). Mathematical impossibility detected."
        )
    elif return_validation["warning"]:
        warnings.append(
            f"WARNING: Return outside 3-sigma bounds. Actual: {total_return:.1%}, "
            f"Expected: {return_validation['expected']:.1%} ± {return_validation['std_dev']:.1%}"
        )

    if calmar_validation.get("warning"):
        warnings.append(calmar_validation["message"])

    if not calmar_validation["passed"]:
        warnings.append(calmar_validation["message"])

    results["warnings"] = warnings

    # Log results
    logger.info("Math consistency validation:")
    logger.info(f"  Expected return: {results['expected_return']:.1%}")
    logger.info(f"  Actual return: {total_return:.1%}")
    logger.info(f"  Sigma deviation: {return_validation['sigma_deviation']:.1f}σ")
    logger.info(f"  Calmar ratio: {calmar:.2f} ({calmar_validation['status']})")
    logger.info(f"  Kelly fraction: {results['kelly_fraction']:.2%}")
    logger.info(f"  Passed: {results['passed']}")

    if warnings:
        for warning in warnings:
            logger.warning(warning)

    return results


def run_math_validation_from_files(
    backtest_path: str = "results/backtest_results.json",
    model_accuracy: float | None = None,
) -> dict:
    """Run math validation from saved backtest results.

    Args:
        backtest_path: Path to backtest results JSON
        model_accuracy: Model accuracy if known

    Returns:
        Validation results
    """
    import json
    from pathlib import Path

    backtest_file = Path(backtest_path)

    if not backtest_file.exists():
        logger.error(f"Backtest results not found: {backtest_path}")
        return {"error": "Backtest results not found", "passed": False}

    with open(backtest_file) as f:
        backtest_results = json.load(f)

    results = validate_backtest_math(backtest_results, model_accuracy)

    if not results["passed"]:
        raise ValueError(
            f"Math consistency check failed: {results.get('warnings', 'Unknown issue')}"
        )

    return results
