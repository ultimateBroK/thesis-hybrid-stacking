"""Walk-forward training dispatcher — LightGBM only."""

from __future__ import annotations

import logging

from thesis.shared.config import Config
from thesis.stage_4_training.walk_forward.lgbm import train_lgbm_walk_forward

logger = logging.getLogger("thesis.pipeline")


def train_walk_forward(config: Config) -> None:
    """Run LightGBM walk-forward training.

    Args:
        config: Application configuration.

    Raises:
        ValueError: If ``model.architecture`` is not ``'lgbm'``.
    """
    if config.model.architecture != "lgbm":
        raise ValueError(
            "This thesis version only supports LightGBM architecture. "
            f"Got model.architecture={config.model.architecture!r}. "
            "Set architecture = 'lgbm' in config.toml [model] section."
        )

    logger.info("Starting LightGBM walk-forward training")
    train_lgbm_walk_forward(
        config, expanded_features=config.model.lgbm_expanded_features
    )
    logger.info("LightGBM walk-forward training complete")
