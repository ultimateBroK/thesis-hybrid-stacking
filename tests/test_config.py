"""Tests for config module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from thesis.config import load_config, Config, LGBMConfig, ModelConfig


def test_load_config_default():
    cfg = load_config(Path(__file__).parent.parent / "config.toml")
    assert isinstance(cfg, Config)
    assert cfg.data.symbol == "XAUUSD"
    assert cfg.data.timeframe == "1H"


def test_config_sections_exist():
    cfg = load_config(Path(__file__).parent.parent / "config.toml")
    assert hasattr(cfg, "data")
    assert hasattr(cfg, "splitting")
    assert hasattr(cfg, "features")
    assert hasattr(cfg, "labels")
    assert hasattr(cfg, "model")
    assert hasattr(cfg, "backtest")
    assert hasattr(cfg, "workflow")
    assert hasattr(cfg, "paths")


def test_model_config_flat():
    cfg = load_config(Path(__file__).parent.parent / "config.toml")
    # Model config is a flat dataclass, not a dict
    assert isinstance(cfg.model, LGBMConfig)
    assert cfg.model.num_leaves > 0
    assert cfg.model.learning_rate > 0


def test_labels_no_session_atr():
    cfg = load_config(Path(__file__).parent.parent / "config.toml")
    # No session_atr attribute
    assert not hasattr(cfg.labels, "session_atr")
    assert cfg.labels.atr_multiplier > 0


def test_paths_basic():
    cfg = load_config(Path(__file__).parent.parent / "config.toml")
    assert cfg.paths.train_data.endswith(".parquet")
    assert cfg.paths.val_data.endswith(".parquet")
    assert cfg.paths.test_data.endswith(".parquet")


def test_missing_config_raises():
    from thesis.config import load_config
    import pytest

    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/config.toml")
