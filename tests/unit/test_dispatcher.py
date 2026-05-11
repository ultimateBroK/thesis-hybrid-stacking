"""Tests for walk-forward dispatcher — architecture routing."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from thesis.shared.config import Config
from thesis.stage_4_training.walk_forward.dispatcher import train_walk_forward


@pytest.mark.unit
class TestDispatcher:
    @patch("thesis.stage_4_training.walk_forward.dispatcher.train_lgbm_walk_forward")
    def test_lgbm_architecture_routes_correctly(self, mock_lgbm) -> None:
        config = Config()
        config.model.architecture = "lgbm"
        train_walk_forward(config)
        mock_lgbm.assert_called_once_with(config, expanded_features=False)

    @patch("thesis.stage_4_training.walk_forward.dispatcher.train_stacking_walk_forward")
    def test_stacking_architecture_routes_correctly(self, mock_stacking) -> None:
        config = Config()
        config.model.architecture = "stacking"
        train_walk_forward(config)
        mock_stacking.assert_called_once_with(config)

    def test_unsupported_architecture_raises(self) -> None:
        config = Config()
        config.model.architecture = "unknown_arch"
        with pytest.raises(ValueError, match="Unsupported model.architecture"):
            train_walk_forward(config)
