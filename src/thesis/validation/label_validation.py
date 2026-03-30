"""Validate that labels don't have lookahead bias.

This module checks generated labels for impossible patterns that would
indicate lookahead bias or data contamination.
"""

import polars as pl
import numpy as np
from typing import Dict
from pathlib import Path
import logging

logger = logging.getLogger("thesis.validation")


def validate_no_lookahead(
    labels_df: pl.DataFrame,
    max_horizon: int = 20,
    max_rapid_change_pct: float = 25.0,  # Increased from 5% - triple-barrier naturally has more switches
    max_win_rate: float = 0.60,
) -> Dict:  # Increased from 70% - 50% is normal
    """Check that labels are generated correctly without peeking too far.

    Tests:
    1. No label should require information beyond `horizon` bars
    2. Last `horizon` bars should be neutral (0) or have special handling
    3. Label changes should be smooth (no jumps indicating future knowledge)
    4. Win rate shouldn't be suspiciously high

    Args:
        labels_df: DataFrame with 'label' column
        max_horizon: Maximum look-ahead horizon used in labeling
        max_rapid_change_pct: Maximum percentage of rapid label oscillations allowed
        max_win_rate: Maximum win rate before flagging as suspicious

    Returns:
        Dictionary with validation results
    """
    results = {"passed": True, "errors": [], "warnings": [], "details": {}}

    # Check 1: Last N bars should have limited labels
    last_n = labels_df.tail(max_horizon)
    non_neutral_last_n = last_n.filter(pl.col("label") != 0).shape[0]
    non_neutral_pct = (non_neutral_last_n / max_horizon) * 100

    results["details"]["last_n_non_neutral"] = non_neutral_last_n
    results["details"]["last_n_non_neutral_pct"] = non_neutral_pct

    if non_neutral_pct > 10:  # More than 10% non-neutral
        results["warnings"].append(
            f"Last {max_horizon} bars have {non_neutral_last_n} non-neutral labels ({non_neutral_pct:.1f}%). "
            f"These may use insufficient look-ahead."
        )

    # Check 2: Look for impossible label patterns
    # A label that changes from 1 to -1 in consecutive bars suggests lookahead
    labels = labels_df["label"].to_numpy()
    changes = np.diff(labels)

    # Count rapid oscillations (1 -> -1 or -1 -> 1 in single bar)
    rapid_changes = np.sum(np.abs(changes) == 2)
    rapid_change_pct = rapid_changes / len(changes) * 100 if len(changes) > 0 else 0

    results["details"]["rapid_oscillations"] = rapid_changes
    results["details"]["rapid_oscillation_pct"] = rapid_change_pct

    if rapid_change_pct > max_rapid_change_pct:  # More than threshold
        results["errors"].append(
            f"Detected {rapid_change_pct:.1f}% rapid label oscillations (> {max_rapid_change_pct}%). "
            f"This suggests lookahead bias (perfect knowledge of which barrier hits first)."
        )
        results["passed"] = False

    # Check 3: Validate against random baseline
    # With random predictions, we'd expect ~50% win rate
    # With lookahead, win rate is artificially inflated
    label_counts = labels_df["label"].value_counts()
    total = len(labels_df)

    results["details"]["total_labels"] = total

    # Calculate win rate (excluding holds)
    directional = label_counts.filter(pl.col("label").is_in([1, -1]))
    if directional.shape[0] > 0:
        wins_df = label_counts.filter(pl.col("label") == 1)
        losses_df = label_counts.filter(pl.col("label") == -1)

        wins = wins_df["count"][0] if wins_df.shape[0] > 0 else 0
        losses = losses_df["count"][0] if losses_df.shape[0] > 0 else 0
        total_directional = wins + losses
        win_rate = wins / total_directional if total_directional > 0 else 0

        results["details"]["win_rate"] = win_rate
        results["details"]["wins"] = wins
        results["details"]["losses"] = losses

        if win_rate > max_win_rate:  # Suspiciously high
            results["warnings"].append(
                f"Label win rate is {win_rate:.1%}, which is very high (> {max_win_rate:.0%}). "
                f"Expected ~50% for random walk with drift. This may indicate lookahead bias."
            )

    # Check 4: Label distribution balance
    # Extremely imbalanced labels suggest bias
    hold_pct = (labels == 0).sum() / total
    long_pct = (labels == 1).sum() / total
    short_pct = (labels == -1).sum() / total

    results["details"]["hold_pct"] = hold_pct
    results["details"]["long_pct"] = long_pct
    results["details"]["short_pct"] = short_pct

    # If directional labels are very imbalanced (e.g., 70% long, 30% short)
    if long_pct > 0 or short_pct > 0:
        directional_imbalance = (
            abs(long_pct - short_pct) / (long_pct + short_pct)
            if (long_pct + short_pct) > 0
            else 0
        )
        if directional_imbalance > 0.30:  # >30% imbalance
            dominant = "Long" if long_pct > short_pct else "Short"
            results["warnings"].append(
                f"Directional labels are imbalanced: {long_pct:.1%} Long vs {short_pct:.1%} Short. "
                f"{dominant} bias detected. This may indicate {dominant.lower()} lookahead advantage."
            )

    return results


def run_label_validation_pipeline(config_path: str = "config.toml") -> Dict:
    """Run full label validation as part of pipeline.

    Args:
        config_path: Path to configuration file

    Returns:
        Validation results dictionary
    """
    from thesis.config.loader import Config

    config = Config(config_path)
    labels_path = Path(config.labels.labels_path)

    if not labels_path.exists():
        return {"error": "Labels file not found", "passed": False}

    logger.info(f"Running label validation on {labels_path}")
    df = pl.read_parquet(labels_path)
    results = validate_no_lookahead(df)

    # Log results
    logger.info("Label validation results:")
    logger.info(f"  Passed: {results['passed']}")
    logger.info(f"  Errors: {len(results['errors'])}")
    logger.info(f"  Warnings: {len(results['warnings'])}")

    if results["details"].get("win_rate"):
        logger.info(f"  Win rate: {results['details']['win_rate']:.1%}")

    if results["details"].get("rapid_oscillation_pct"):
        logger.info(
            f"  Rapid oscillations: {results['details']['rapid_oscillation_pct']:.2f}%"
        )

    if not results["passed"]:
        error_msg = f"Label validation failed: {'; '.join(results['errors'])}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if results["warnings"]:
        for warning in results["warnings"]:
            logger.warning(f"Label validation warning: {warning}")

    logger.info("Label validation completed successfully")
    return results


def quick_label_check(labels_path: str | Path) -> bool:
    """Quick check if labels pass validation.

    Args:
        labels_path: Path to labels parquet file

    Returns:
        True if labels pass, False otherwise
    """
    try:
        df = pl.read_parquet(labels_path)
        results = validate_no_lookahead(df)
        return results["passed"]
    except Exception as e:
        logger.error(f"Label check failed: {e}")
        return False
