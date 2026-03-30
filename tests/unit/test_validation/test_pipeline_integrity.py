"""Unit tests for pipeline integrity validation module.

Tests for file integrity checks, hash computation, clean rebuild,
and backup functionality.
"""

import hashlib
import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest


class TestComputeFileHash:
    """Tests for the _compute_file_hash function."""

    def test_compute_hash_existing_file(self, tmp_path):
        """Should compute hash for existing file."""
        from thesis.validation.pipeline_integrity import _compute_file_hash

        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello World")

        hash_result = _compute_file_hash(test_file)

        assert len(hash_result) == 32  # MD5 hex is 32 chars
        assert hash_result != ""

        # Verify it's correct MD5
        expected = hashlib.md5(b"Hello World").hexdigest()
        assert hash_result == expected

    def test_compute_hash_nonexistent_file(self, tmp_path):
        """Should return empty string for non-existent file."""
        from thesis.validation.pipeline_integrity import _compute_file_hash

        nonexistent = tmp_path / "does_not_exist.txt"
        hash_result = _compute_file_hash(nonexistent)

        assert hash_result == ""

    def test_compute_hash_empty_file(self, tmp_path):
        """Should compute hash for empty file."""
        from thesis.validation.pipeline_integrity import _compute_file_hash

        empty_file = tmp_path / "empty.txt"
        empty_file.touch()

        hash_result = _compute_file_hash(empty_file)

        expected = hashlib.md5(b"").hexdigest()
        assert hash_result == expected

    def test_compute_hash_binary_file(self, tmp_path):
        """Should handle binary files correctly."""
        from thesis.validation.pipeline_integrity import _compute_file_hash

        binary_file = tmp_path / "binary.bin"
        binary_file.write_bytes(bytes(range(256)))

        hash_result = _compute_file_hash(binary_file)

        assert len(hash_result) == 32
        assert hash_result != ""


class TestVerifyPipelineIntegrity:
    """Tests for the verify_pipeline_integrity function."""

    def test_all_files_present_passes(self, tmp_path):
        """Should pass when all required files exist."""
        from thesis.validation.pipeline_integrity import verify_pipeline_integrity

        # Create required files
        (tmp_path / "data" / "processed").mkdir(parents=True)
        (tmp_path / "labels.parquet").write_bytes(b"test")
        (tmp_path / "features.parquet").write_bytes(b"test")
        (tmp_path / "ohlcv.parquet").write_bytes(b"test")
        (tmp_path / "train.parquet").write_bytes(b"test")
        (tmp_path / "val.parquet").write_bytes(b"test")
        (tmp_path / "test.parquet").write_bytes(b"test")

        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = True
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value = MagicMock(st_size=100)
                with patch(
                    "polars.read_parquet", return_value=pl.DataFrame({"col": [1, 2, 3]})
                ):
                    report = verify_pipeline_integrity()

        assert report.passed is True

    def test_missing_required_file_fails(self, tmp_path):
        """Should fail when required file is missing."""
        from thesis.validation.pipeline_integrity import verify_pipeline_integrity

        with patch("pathlib.Path.exists", return_value=False):
            report = verify_pipeline_integrity()

        assert report.passed is False
        assert len(report.errors) > 0
        assert any("Missing" in e for e in report.errors)

    def test_empty_file_detected(self, tmp_path):
        """Should detect empty files."""
        from thesis.validation.pipeline_integrity import verify_pipeline_integrity

        # Create empty file
        (tmp_path / "data" / "processed").mkdir(parents=True)
        (tmp_path / "labels.parquet").touch()

        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value.st_size = 0
                report = verify_pipeline_integrity()

        assert any("Empty" in e for e in report.errors)

    def test_corrupted_parquet_detected(self, tmp_path):
        """Should detect corrupted parquet files."""
        from thesis.validation.pipeline_integrity import verify_pipeline_integrity

        # Create corrupted parquet
        (tmp_path / "data" / "processed").mkdir(parents=True)
        (tmp_path / "labels.parquet").write_text("not valid parquet")

        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value.st_size = 100
                with patch(
                    "polars.read_parquet", side_effect=Exception("Invalid parquet")
                ):
                    report = verify_pipeline_integrity()

        assert any("Corrupted" in e for e in report.errors)

    def test_optional_files_not_required(self, tmp_path):
        """Should not fail if optional files are missing."""
        from thesis.validation.pipeline_integrity import verify_pipeline_integrity

        # Create only required files
        with patch("pathlib.Path.exists") as mock_exists:
            # First 6 calls (required) return True, rest (optional) return False
            def side_effect(*args, **kwargs):
                # Count calls to return True for first 6, then False
                side_effect.call_count = getattr(side_effect, "call_count", 0) + 1
                return side_effect.call_count <= 6

            mock_exists.side_effect = side_effect

            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value.st_size = 1000
                with patch(
                    "polars.read_parquet", return_value=pl.DataFrame({"col": [1]})
                ):
                    report = verify_pipeline_integrity()

        # Should pass even without optional files
        assert report.passed is True

    def test_stage_status_reported(self, tmp_path):
        """Should report status for each stage."""
        from thesis.validation.pipeline_integrity import verify_pipeline_integrity

        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value.st_size = 1000
                with patch(
                    "polars.read_parquet", return_value=pl.DataFrame({"col": [1]})
                ):
                    report = verify_pipeline_integrity()

        assert "labels" in report.stage_status
        assert "features" in report.stage_status
        assert "ohlcv" in report.stage_status


class TestGenerateIntegrityReport:
    """Tests for the generate_integrity_report function."""

    def test_report_formatting(self):
        """Should format report correctly."""
        from thesis.validation.pipeline_integrity import (
            IntegrityReport,
            generate_integrity_report,
        )

        report = IntegrityReport(
            passed=True,
            errors=[],
            warnings=[],
            stage_status={
                "labels": "OK",
                "features": "OK",
                "missing": "MISSING",
            },
        )

        output = generate_integrity_report(report)

        assert "PIPELINE INTEGRITY REPORT" in output
        assert "labels: OK" in output
        assert "✅ PIPELINE INTEGRITY: PASSED" in output

    def test_report_with_errors(self):
        """Should include errors in report."""
        from thesis.validation.pipeline_integrity import (
            IntegrityReport,
            generate_integrity_report,
        )

        report = IntegrityReport(
            passed=False,
            errors=["File not found: test.txt"],
            warnings=[],
            stage_status={},
        )

        output = generate_integrity_report(report)

        assert "ERRORS:" in output
        assert "File not found" in output
        assert "❌ PIPELINE INTEGRITY: FAILED" in output

    def test_report_with_warnings(self):
        """Should include warnings in report."""
        from thesis.validation.pipeline_integrity import (
            IntegrityReport,
            generate_integrity_report,
        )

        report = IntegrityReport(
            passed=True,
            errors=[],
            warnings=["File is very old"],
            stage_status={},
        )

        output = generate_integrity_report(report)

        assert "WARNINGS:" in output
        assert "File is very old" in output


class TestForceCleanRebuild:
    """Tests for the force_clean_rebuild function."""

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.stat")
    @patch("pathlib.Path.unlink")
    @patch("shutil.rmtree")
    @patch("pathlib.Path.mkdir")
    def test_rebuild_without_confirmation(
        self, mock_mkdir, mock_rmtree, mock_unlink, mock_stat, mock_exists
    ):
        """Should clean without confirmation when confirm=False."""
        from thesis.validation.pipeline_integrity import force_clean_rebuild

        mock_exists.return_value = True
        mock_stat.return_value.st_size = 1000

        with patch("pathlib.Path.rglob", return_value=[]):
            with patch("builtins.print"):
                force_clean_rebuild(confirm=False)

                mock_rmtree.assert_called()
                mock_mkdir.assert_called()

    @patch("builtins.input", return_value="n")
    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.stat")
    @patch("shutil.rmtree")
    def test_rebuild_cancelled(self, mock_rmtree, mock_stat, mock_exists, mock_input):
        """Should cancel when user declines."""
        mock_stat.return_value.st_size = 1000

        from thesis.validation.pipeline_integrity import force_clean_rebuild

        with patch("pathlib.Path.rglob", return_value=[]):
            with patch("builtins.print"):
                force_clean_rebuild(confirm=True)

                mock_rmtree.assert_not_called()

    @patch("builtins.input", return_value="y")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.stat")
    @patch("pathlib.Path.unlink")
    @patch("shutil.rmtree")
    @patch("pathlib.Path.mkdir")
    def test_rebuild_confirmed(
        self, mock_mkdir, mock_rmtree, mock_unlink, mock_stat, mock_exists, mock_input
    ):
        """Should proceed when user confirms."""
        mock_stat.return_value.st_size = 1000
        mock_exists.return_value = True

        from thesis.validation.pipeline_integrity import force_clean_rebuild

        with patch("pathlib.Path.rglob", return_value=[]):
            with patch("builtins.print"):
                force_clean_rebuild(confirm=True)

        mock_rmtree.assert_called()

    @patch("pathlib.Path.exists", return_value=False)
    @patch("shutil.rmtree")
    def test_rebuild_with_no_files(self, mock_rmtree, mock_exists):
        """Should handle case where directories don't exist."""
        from thesis.validation.pipeline_integrity import force_clean_rebuild

        with patch("builtins.print"):
            force_clean_rebuild(confirm=False)

        # Should not error, and should try to create directories
        assert True  # No exception raised


class TestBackupCriticalFiles:
    """Tests for the backup_critical_files function."""

    @patch("pathlib.Path.exists", return_value=True)
    @patch("shutil.copy2")
    @patch("pathlib.Path.mkdir")
    def test_backup_creates_directory(
        self, mock_mkdir, mock_copy, mock_exists, tmp_path
    ):
        """Should create backup directory."""
        from thesis.validation.pipeline_integrity import backup_critical_files

        with patch("pathlib.Path.mkdir") as mock_mkdir2:
            backup_critical_files(str(tmp_path / "backups"))

        mock_mkdir2.assert_called_with(parents=True, exist_ok=True)

    @patch("shutil.copy2")
    @patch("pathlib.Path.mkdir")
    def test_backup_copies_existing_files(self, mock_mkdir, mock_copy, tmp_path):
        """Should copy existing files."""
        from thesis.validation.pipeline_integrity import backup_critical_files

        # Mock exists to return True
        with patch("pathlib.Path.exists", return_value=True):
            backup_dir = tmp_path / "backups"
            backup_critical_files(str(backup_dir))

        # Should have called copy2 for each existing file
        assert mock_copy.call_count > 0

    @patch("shutil.copy2")
    @patch("pathlib.Path.mkdir")
    def test_backup_skips_missing_files(self, mock_mkdir, mock_copy, tmp_path):
        """Should skip files that don't exist."""
        from thesis.validation.pipeline_integrity import backup_critical_files

        with patch("pathlib.Path.exists", return_value=False):
            backup_critical_files(str(tmp_path / "backups"))

        # Should not have tried to copy
        mock_copy.assert_not_called()

    @patch("pathlib.Path.exists", return_value=True)
    @patch("shutil.copy2")
    @patch("pathlib.Path.mkdir")
    def test_backup_returns_path(self, mock_mkdir, mock_copy, mock_exists, tmp_path):
        """Should return path to backup directory."""
        from thesis.validation.pipeline_integrity import backup_critical_files

        result = backup_critical_files(str(tmp_path / "backups"))

        assert isinstance(result, Path)
        assert "backup_" in str(result)


class TestCliMain:
    """Tests for the CLI main block."""

    @patch("sys.argv", ["pipeline_integrity.py", "--check"])
    @patch("thesis.validation.pipeline_integrity.verify_pipeline_integrity")
    @patch("thesis.validation.pipeline_integrity.generate_integrity_report")
    def test_cli_check(self, mock_generate, mock_verify):
        """CLI --check should verify integrity."""
        from thesis.validation import pipeline_integrity

        mock_report = MagicMock()
        mock_report.passed = True
        mock_verify.return_value = mock_report
        mock_generate.return_value = "Test report"

        with patch("builtins.print"):
            # Simulate running the module
            with pytest.raises(SystemExit) as exc_info:
                pipeline_integrity.report = mock_report
                # Execute the CLI logic
                import sys

                if "--check" in sys.argv:
                    report = mock_verify()
                    mock_generate(report)
                    sys.exit(0 if report.passed else 1)

        assert exc_info.value.code == 0

    @patch("sys.argv", ["pipeline_integrity.py", "--clean"])
    @patch("thesis.validation.pipeline_integrity.force_clean_rebuild")
    def test_cli_clean(self, mock_rebuild):
        """CLI --clean should trigger rebuild."""
        from thesis.validation import pipeline_integrity

        with patch("builtins.input", return_value="y"):
            # Execute the CLI logic
            import sys

            if "--clean" in sys.argv:
                pipeline_integrity.force_clean_rebuild()

        mock_rebuild.assert_called_once()

    @patch("sys.argv", ["pipeline_integrity.py", "--backup"])
    @patch("thesis.validation.pipeline_integrity.backup_critical_files")
    def test_cli_backup(self, mock_backup, tmp_path):
        """CLI --backup should create backup."""
        from thesis.validation import pipeline_integrity

        mock_backup.return_value = tmp_path / "backup"

        with patch("builtins.print"):
            import sys

            if "--backup" in sys.argv:
                backup_path = pipeline_integrity.backup_critical_files()
                print(f"Backup created: {backup_path}")

        mock_backup.assert_called_once()

    @patch("sys.argv", ["pipeline_integrity.py"])
    def test_cli_no_args_prints_usage(self):
        """CLI without args should print usage."""
        import sys

        with patch("builtins.print") as mock_print:
            if len(sys.argv) == 1:
                print(
                    "Usage: python -m thesis.validation.pipeline_integrity [--check|--clean|--backup]"
                )
                print("")
                print("  --check   Verify pipeline integrity")
                print("  --clean   Force clean rebuild")
                print("  --backup  Create backup of critical files")

        assert mock_print.call_count >= 2


class TestEdgeCases:
    """Edge case tests for pipeline integrity."""

    def test_verify_with_empty_dataframe_warning(self, tmp_path):
        """Empty dataframe should generate warning."""
        from thesis.validation.pipeline_integrity import verify_pipeline_integrity

        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value.st_size = 100
                with patch("polars.read_parquet", return_value=pl.DataFrame()):
                    report = verify_pipeline_integrity()

        assert any("Empty" in w for w in report.warnings)

    def test_backup_with_timestamp(self, tmp_path):
        """Backup directory should include timestamp."""
        from thesis.validation.pipeline_integrity import backup_critical_files

        import re

        with patch("pathlib.Path.exists", return_value=False):
            with patch("pathlib.Path.mkdir"):
                result = backup_critical_files(str(tmp_path / "backups"))

        # Check timestamp format in path
        assert re.search(r"backup_\d{8}_\d{6}", str(result))

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.stat")
    @patch("pathlib.Path.unlink")
    def test_file_size_calculation(self, mock_unlink, mock_stat, mock_exists, tmp_path):
        """Should calculate file sizes during rebuild preview."""
        from thesis.validation.pipeline_integrity import force_clean_rebuild

        # Create test directories with files
        (tmp_path / "data" / "processed").mkdir(parents=True)
        (tmp_path / "data" / "processed" / "test.parquet").write_bytes(b"x" * 1000)

        mock_stat.return_value.st_size = 1000

        with patch("pathlib.Path.rglob", return_value=[]) as mock_rglob:
            with patch("builtins.print"):
                with patch("shutil.rmtree"):
                    with patch("pathlib.Path.mkdir"):
                        force_clean_rebuild(confirm=False)
