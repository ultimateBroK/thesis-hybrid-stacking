"""Tests for Stage 3 model experiment orchestration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from thesis.models.train import train_walk_forward
from thesis.shared.config import Config


@pytest.mark.unit
class TestDispatcher:
    """Stage 3 dispatcher tests."""

    @patch("thesis.models.train.save_model_experiment")
    @patch("thesis.models.train.run_model_experiment")
    @patch("thesis.models.train.build_walk_forward_windows")
    @patch("thesis.models.train.choose_model_features")
    @patch("thesis.models.train.load_model_dataset")
    def test_runs_fixed_feature_experiment(
        self,
        mock_load,
        mock_features,
        mock_windows,
        mock_run,
        mock_save,
    ) -> None:
        """train_walk_forward runs load -> choose -> windows -> run -> save."""
        config = Config()
        dataset = MagicMock()
        windows = [MagicMock()]
        experiment = MagicMock()

        mock_load.return_value = dataset
        mock_features.return_value = ["rsi_14"]
        mock_windows.return_value = windows
        mock_run.return_value = experiment

        train_walk_forward(config)

        mock_load.assert_called_once_with(config)
        mock_features.assert_called_once_with(dataset, config)
        mock_windows.assert_called_once_with(dataset, config)
        mock_run.assert_called_once_with(dataset, ["rsi_14"], windows, config)
        mock_save.assert_called_once_with(experiment, config)

    @pytest.mark.skip(reason="lgbm routing removed — always stacking now")
    def test_lgbm_architecture_routes_correctly(self) -> None:
        """Legacy routing test retained as skipped documentation."""
        pass

    def test_unsupported_architecture_raises(self) -> None:
        """Architecture no longer routes Stage 3."""
        config = Config()
        config.model.architecture = "unknown_arch"
        config.model.objective = "regression"
        with pytest.raises(ValueError, match="classification-only"):
            train_walk_forward(config)
