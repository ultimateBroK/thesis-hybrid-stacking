"""Dispatch Stage 4 — Hybrid Stacking only."""

from __future__ import annotations

import logging

from thesis.shared.config import Config
from thesis.stage_4_training.walk_forward.stacking import train_stacking_walk_forward

logger = logging.getLogger("thesis")


def train_walk_forward(config: Config) -> None:
    """Run Hybrid Stacking walk-forward training."""
    logger.info("Architecture: hybrid_stacking (fixed)")
    train_stacking_walk_forward(config)
