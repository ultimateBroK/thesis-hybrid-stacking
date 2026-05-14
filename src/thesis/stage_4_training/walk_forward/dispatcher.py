"""Route walk-forward training to the configured architecture."""

from __future__ import annotations

import logging

from thesis.shared.config import Config
from thesis.stage_4_training.walk_forward.lgbm import train_lgbm_walk_forward
from thesis.stage_4_training.walk_forward.stacking import train_stacking_walk_forward

logger = logging.getLogger("thesis")


def train_walk_forward(config: Config) -> None:
    """Dispatch to the right trainer based on config.model.architecture."""
    arch = config.model.architecture

    if arch == "stacking":
        train_stacking_walk_forward(config)
        return

    if arch == "lgbm":
        train_lgbm_walk_forward(
            config, expanded_features=config.model.lgbm_expanded_features
        )
        return

    raise ValueError(f"Unsupported architecture: {arch!r}. Use 'stacking' or 'lgbm'")
