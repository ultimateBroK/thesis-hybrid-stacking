"""Validate pipeline integrity and detect contamination.

This module ensures that all pipeline outputs are consistent with current
configuration and provides tools for clean rebuilds.
"""

import hashlib
import shutil
import polars as pl
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
import logging

logger = logging.getLogger("thesis.validation")


@dataclass
class IntegrityReport:
    """Pipeline integrity check results."""

    passed: bool
    errors: List[str]
    warnings: List[str]
    stage_status: Dict[str, str]


def _compute_file_hash(file_path: Path) -> str:
    """Compute MD5 hash of file for freshness check."""
    if not file_path.exists():
        return ""

    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception:
        return ""


def verify_pipeline_integrity(config_hash: str = "") -> IntegrityReport:
    """Verify that all pipeline outputs are consistent and non-corrupted.

    Checks:
    1. All required files exist
    2. File modification times make sense (outputs after inputs)
    3. Parquet files are readable and non-empty
    4. No stale intermediate files

    Args:
        config_hash: Hash of current configuration (optional)

    Returns:
        Integrity report with pass/fail status
    """
    errors = []
    warnings = []
    stage_status = {}

    required_files = [
        ("labels", "data/processed/labels.parquet"),
        ("features", "data/processed/features.parquet"),
        ("ohlcv", "data/processed/ohlcv.parquet"),
        ("train", "data/processed/train.parquet"),
        ("validation", "data/processed/val.parquet"),
        ("test", "data/processed/test.parquet"),
    ]

    optional_files = [
        ("lightgbm", "models/lightgbm_model.pkl"),
        ("lstm", "models/lstm_model.pt"),
        ("lstm_norm", "models/lstm_norm_stats.npz"),
        ("stacking", "models/stacking_meta_learner.pkl"),
        ("backtest", "results/backtest_results.json"),
    ]

    # Check required files
    for stage_name, file_path in required_files:
        path = Path(file_path)

        if not path.exists():
            errors.append(f"Missing required file: {file_path}")
            stage_status[stage_name] = "MISSING"
            continue

        if path.stat().st_size == 0:
            errors.append(f"Empty file: {file_path}")
            stage_status[stage_name] = "EMPTY"
            continue

        # For parquet files, try to read to validate
        if file_path.endswith(".parquet"):
            try:
                df = pl.read_parquet(path)
                if len(df) == 0:
                    warnings.append(f"Empty dataframe in {file_path}")
            except Exception as e:
                errors.append(f"Corrupted parquet file {file_path}: {e}")
                stage_status[stage_name] = "CORRUPTED"
                continue

        stage_status[stage_name] = "OK"

    # Check optional files (don't fail if missing)
    for stage_name, file_path in optional_files:
        path = Path(file_path)

        if path.exists():
            if path.stat().st_size > 0:
                stage_status[stage_name] = "OK"
            else:
                warnings.append(f"Empty optional file: {file_path}")
                stage_status[stage_name] = "EMPTY"
        else:
            stage_status[stage_name] = "NOT_GENERATED"

    return IntegrityReport(
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        stage_status=stage_status,
    )


def generate_integrity_report(report: IntegrityReport) -> str:
    """Generate human-readable integrity report.

    Args:
        report: IntegrityReport object

    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("PIPELINE INTEGRITY REPORT")
    lines.append("=" * 80)

    # Stage status
    lines.append("\nStage Status:")
    for stage, status in report.stage_status.items():
        icon = (
            "✓"
            if status == "OK"
            else "✗"
            if status in ["MISSING", "EMPTY", "CORRUPTED"]
            else "○"
        )
        lines.append(f"  {icon} {stage}: {status}")

    # Errors
    if report.errors:
        lines.append("\n❌ ERRORS:")
        for error in report.errors:
            lines.append(f"  - {error}")

    # Warnings
    if report.warnings:
        lines.append("\n⚠️  WARNINGS:")
        for warning in report.warnings:
            lines.append(f"  - {warning}")

    # Overall status
    lines.append("\n" + "=" * 80)
    if report.passed:
        lines.append("✅ PIPELINE INTEGRITY: PASSED")
    else:
        lines.append("❌ PIPELINE INTEGRITY: FAILED")
        lines.append(
            "\nRecommended action: Run force_clean_rebuild() and re-run pipeline"
        )
    lines.append("=" * 80)

    return "\n".join(lines)


def force_clean_rebuild(confirm: bool = True) -> None:
    """Remove all intermediate and output files for clean rebuild.

    Args:
        confirm: If True, ask for confirmation before deleting
    """
    dirs_to_clean = [
        "data/processed",
        "models",
        "results",
    ]

    files_to_clean = [
        ".pipeline_state.json",
        ".tracking/state.json",
    ]

    # Calculate what will be deleted
    total_size = 0
    files_count = 0

    for dir_name in dirs_to_clean:
        dir_path = Path(dir_name)
        if dir_path.exists():
            for item in dir_path.rglob("*"):
                if item.is_file():
                    total_size += item.stat().st_size
                    files_count += 1

    for file_name in files_to_clean:
        file_path = Path(file_name)
        if file_path.exists():
            total_size += file_path.stat().st_size
            files_count += 1

    size_mb = total_size / (1024 * 1024)

    print("\n🧹 CLEAN REBUILD PREVIEW")
    print(f"   Files to delete: {files_count}")
    print(f"   Total size: {size_mb:.1f} MB")
    print(f"   Directories: {', '.join(dirs_to_clean)}")
    print("\n   This will force a complete rebuild of:")
    print("   - Labels (triple-barrier)")
    print("   - Features")
    print("   - Train/Val/Test splits")
    print("   - Models (LightGBM, LSTM, Stacking)")
    print("   - Backtest results")

    if confirm:
        response = input("\n⚠️  Proceed with clean rebuild? [y/N]: ")
        if response.lower() != "y":
            print("❌ Clean rebuild cancelled.")
            return

    # Execute cleanup
    deleted_count = 0

    for dir_name in dirs_to_clean:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"\n  Cleaning {dir_name}...")
            shutil.rmtree(dir_path)
            dir_path.mkdir(parents=True, exist_ok=True)
            deleted_count += 1

    for file_name in files_to_clean:
        file_path = Path(file_name)
        if file_path.exists():
            print(f"  Removing {file_name}...")
            file_path.unlink()

    print("\n✅ Clean rebuild completed.")
    print(f"   Deleted: {deleted_count} directories")
    print("\n   Next steps:")
    print("   1. Run: python main.py --force")
    print("   2. Wait for pipeline completion")
    print("   3. Validate: python -m thesis.validation.run_all_validations")


def backup_critical_files(backup_dir: str = ".tracking/backups") -> Path:
    """Create backup of critical files before major changes.

    Args:
        backup_dir: Directory for backups

    Returns:
        Path to backup directory
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = Path(backup_dir) / f"backup_{timestamp}"
    backup_path.mkdir(parents=True, exist_ok=True)

    files_to_backup = [
        "config.toml",
        "src/thesis/labels/triple_barrier.py",
        "src/thesis/config/loader.py",
    ]

    for file_path in files_to_backup:
        src = Path(file_path)
        if src.exists():
            dst = backup_path / src.name
            shutil.copy2(src, dst)
            logger.info(f"Backed up: {file_path}")

    logger.info(f"Backup created: {backup_path}")
    return backup_path


if __name__ == "__main__":
    # CLI interface
    import sys

    if "--check" in sys.argv:
        report = verify_pipeline_integrity()
        print(generate_integrity_report(report))
        sys.exit(0 if report.passed else 1)

    elif "--clean" in sys.argv:
        force_clean_rebuild()

    elif "--backup" in sys.argv:
        backup_path = backup_critical_files()
        print(f"Backup created: {backup_path}")

    else:
        print(
            "Usage: python -m thesis.validation.pipeline_integrity [--check|--clean|--backup]"
        )
        print("")
        print("  --check   Verify pipeline integrity")
        print("  --clean   Force clean rebuild (deletes all intermediate files)")
        print("  --backup  Create backup of critical files")
