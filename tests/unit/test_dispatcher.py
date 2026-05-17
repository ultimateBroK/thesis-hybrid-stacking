"""Tests for walk-forward dispatcher — architecture routing.

After refactor, train_walk_forward always delegates to stacking.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from thesis.models.train import train_walk_forward
from thesis.shared.config import Config


@pytest.mark.unit
class TestDispatcher:
    @patch("thesis.models.stacking.train_stacking_walk_forward")
    def test_always_routes_to_stacking(self, mock_stacking) -> None:
        config = Config()
        train_walk_forward(config)
        mock_stacking.assert_called_once_with(config)

    @pytest.mark.skip(reason="lgbm routing removed — always stacking now")
    def test_lgbm_architecture_routes_correctly(self) -> None:
        pass

    def test_unsupported_architecture_raises(self) -> None:
        """train_walk_forward no longer routes by architecture — always stacking."""
        config = Config()
        config.model.architecture = "unknown_arch"
        # No longer raises — always uses stacking
        train_walk_forward(config)
