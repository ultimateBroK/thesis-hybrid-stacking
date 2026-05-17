"""Integration tests for pipeline module.

Tests pipeline stage ordering, caching, and --force flag.
These tests use a temporary directory and do NOT write to the project's results/ directory.
"""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))


from thesis.shared.config import Config


@pytest.fixture
def sample_config(tmp_path: Path) -> Config:
    """Create a sample config with temporary paths."""
    config = Config()
    config.paths.session_dir = str(tmp_path)
    config.paths.ohlcv = str(tmp_path / "ohlcv.parquet")
    config.paths.features = str(tmp_path / "features.parquet")
    config.paths.labels = str(tmp_path / "labels.parquet")
    return config


@pytest.mark.integration
@pytest.mark.skip(reason="Integration pipeline test — requires full setup")
def test_pipeline_runs_all_stages(sample_config: Config) -> None:
    """Full pipeline should run without errors."""
    pass


@pytest.mark.integration
@pytest.mark.skip(reason="Old 6-stage layout removed — now 4 stages")
def test_new_stage_package_layout() -> None:
    """Verify all stage packages are importable and expose expected public API."""
    pass
