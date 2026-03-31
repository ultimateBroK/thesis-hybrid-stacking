#!/usr/bin/env python3
"""Run all validation checks on the trading system.

This script runs comprehensive validations to ensure the system is
mathematically consistent and free from data leakage.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from thesis.validation import (
    verify_pipeline_integrity,
    generate_integrity_report,
    run_label_validation_pipeline,
    run_math_validation_from_files,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("thesis.validation.runner")


def run_all_validations(config_path: str = "config.toml") -> dict:
    """Run all validation checks.

    Args:
        config_path: Path to configuration file

    Returns:
        Dictionary with all validation results
    """
    results = {"passed": True, "checks": {}}

    logger.info("\n" + "=" * 80)
    logger.info("COMPREHENSIVE VALIDATION SUITE")
    logger.info("=" * 80)

    # 1. Pipeline Integrity Check
    logger.info("\n[1/3] Running pipeline integrity check...")
    try:
        integrity_report = verify_pipeline_integrity()
        results["checks"]["integrity"] = {
            "passed": integrity_report.passed,
            "errors": len(integrity_report.errors),
            "warnings": len(integrity_report.warnings),
        }

        if not integrity_report.passed:
            results["passed"] = False
            logger.info(generate_integrity_report(integrity_report))
            logger.info("\n❌ Pipeline integrity check FAILED")
            logger.info("   Fix: Run force_clean_rebuild() and re-run pipeline")
            return results
        else:
            logger.info("✅ Pipeline integrity: PASSED")
    except Exception as e:
        logger.error(f"Integrity check failed: {e}")
        results["checks"]["integrity"] = {"passed": False, "error": str(e)}
        results["passed"] = False

    # 2. Label Validation
    logger.info("\n[2/3] Running label validation...")
    try:
        label_results = run_label_validation_pipeline(config_path)
        results["checks"]["labels"] = {
            "passed": label_results["passed"],
            "warnings": len(label_results.get("warnings", [])),
        }

        if not label_results["passed"]:
            results["passed"] = False
            logger.error("❌ Label validation FAILED")
            for error in label_results.get("errors", []):
                logger.error(f"   Error: {error}")
        else:
            logger.info("✅ Label validation: PASSED")
            if label_results.get("warnings"):
                logger.warning(f"   Warnings: {len(label_results['warnings'])}")
    except Exception as e:
        logger.error(f"Label validation failed: {e}")
        results["checks"]["labels"] = {"passed": False, "error": str(e)}
        results["passed"] = False

    # 3. Math Consistency Check
    logger.info("\n[3/3] Running math consistency check...")
    try:
        math_results = run_math_validation_from_files()
        results["checks"]["math"] = {
            "passed": math_results["passed"],
            "expected_return": math_results.get("expected_return", 0),
            "kelly_fraction": math_results.get("kelly_fraction", 0),
        }

        if not math_results["passed"]:
            results["passed"] = False
            logger.error("❌ Math consistency check FAILED")
            for warning in math_results.get("warnings", []):
                logger.warning(f"   Warning: {warning}")
        else:
            logger.info("✅ Math consistency: PASSED")
            logger.info(f"   Expected return: {math_results['expected_return']:.1%}")
            logger.info(f"   Kelly fraction: {math_results['kelly_fraction']:.1%}")

            if math_results.get("warnings"):
                logger.warning(f"   Warnings: {len(math_results['warnings'])}")
                for warning in math_results["warnings"]:
                    logger.warning(f"     - {warning}")
    except Exception as e:
        logger.error(f"Math validation failed: {e}")
        results["checks"]["math"] = {"passed": False, "error": str(e)}
        results["passed"] = False

    # Summary
    logger.info("\n" + "=" * 80)
    if results["passed"]:
        logger.info("✅ ALL VALIDATIONS PASSED")
        logger.info("=" * 80)
        logger.info("\nSystem is ready for further testing.")
    else:
        logger.error("❌ SOME VALIDATIONS FAILED")
        logger.info("=" * 80)
        logger.info("\nPlease fix the issues above before proceeding.")
        logger.info("\nRecommended actions:")
        logger.info(
            "  1. If label validation failed: Fix triple_barrier.py and regenerate"
        )
        logger.info(
            "  2. If math check failed: Investigate backtest for hidden advantages"
        )
        logger.info("  3. If integrity failed: Run force_clean_rebuild()")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run all validation checks on trading system"
    )
    parser.add_argument(
        "--config",
        default="config.toml",
        help="Path to configuration file (default: config.toml)",
    )
    parser.add_argument(
        "--exit-on-fail",
        action="store_true",
        help="Exit with non-zero code if any check fails",
    )

    args = parser.parse_args()

    results = run_all_validations(args.config)

    if args.exit_on_fail and not results["passed"]:
        sys.exit(1)
    else:
        sys.exit(0)
