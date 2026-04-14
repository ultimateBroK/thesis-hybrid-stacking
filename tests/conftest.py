"""Shared test fixtures."""

import sys
from pathlib import Path

import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def config():
    """Load the default config for testing."""
    from thesis.config import load_config

    return load_config(Path(__file__).parent.parent / "config.toml")
