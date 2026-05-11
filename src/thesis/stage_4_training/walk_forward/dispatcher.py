"""Walk-forward training dispatcher — routes supported model architecture variants."""

from __future__ import annotations

import logging

from thesis.shared.config import Config
from thesis.stage_4_training.walk_forward.lgbm import train_lgbm_walk_forward
from thesis.stage_4_training.walk_forward.stacking import train_stacking_walk_forward

logger = logging.getLogger("thesis.pipeline")


def train_walk_forward(config: Config) -> None:
    """Dispatch walk-forward training to the configured architecture.

    Supported runtime architectures are classical stacking and LightGBM-only.
    Legacy sequence-model and sequence-hybrid paths are no longer production
    runtime paths.
    """
    architecture = config.model.architecture

    if architecture == "stacking":
        logger.info("Using classical stacking walk-forward pipeline")
        train_stacking_walk_forward(config)
        return

    if architecture == "lgbm":
        logger.info("Using LightGBM walk-forward pipeline")
        train_lgbm_walk_forward(
            config, expanded_features=config.model.lgbm_expanded_features
        )
        return

    raise ValueError(
        f"Unsupported model.architecture: {architecture!r}. "
        "Must be one of: 'stacking', 'lgbm'"
    )
