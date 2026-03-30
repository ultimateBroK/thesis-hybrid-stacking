"""Validation module for trading system integrity checks.

This module provides tools to validate:
- Label quality and absence of lookahead bias
- Mathematical consistency of backtest results
- Pipeline integrity and version control
- Monte Carlo robustness testing
- Stress testing for extreme conditions
"""

from thesis.validation.label_validation import (
    validate_no_lookahead,
    run_label_validation_pipeline,
    quick_label_check
)

from thesis.validation.math_consistency import (
    TradingMathValidator,
    validate_backtest_math,
    run_math_validation_from_files
)

from thesis.validation.pipeline_integrity import (
    verify_pipeline_integrity,
    generate_integrity_report,
    force_clean_rebuild,
    backup_critical_files,
    IntegrityReport
)

__all__ = [
    # Label validation
    "validate_no_lookahead",
    "run_label_validation_pipeline",
    "quick_label_check",
    
    # Math consistency
    "TradingMathValidator",
    "validate_backtest_math",
    "run_math_validation_from_files",
    
    # Pipeline integrity
    "verify_pipeline_integrity",
    "generate_integrity_report",
    "force_clean_rebuild",
    "backup_critical_files",
    "IntegrityReport",
]
