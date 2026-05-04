"""Pipeline orchestration — sequential stage runner.

Stages:
    1. Data preparation (tick → OHLCV)
    2. Feature engineering
    3. Triple-barrier labeling
    4. Walk-forward training (dispatches to stage_4_training)
    5. Backtest (on concatenated OOF predictions)
    6. Report generation
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from thesis._shared.config import Config
from thesis._shared.ui import console, stage_header, stage_skip
from thesis.stage_1_data import prepare_data
from thesis.stage_2_features import generate_features
from thesis.stage_3_labels import generate_labels
from thesis.stage_4_training._walk_forward import _run_walk_forward, _run_static_train
from thesis.stage_5_backtest import run_backtest
from thesis.stage_6_reporting import generate_report

logger = logging.getLogger("thesis.pipeline")


# ---------------------------------------------------------------------------
# Cache fingerprinting
# ---------------------------------------------------------------------------

# Config sections whose values affect each stage's output.
# Used by _cache_hash to fingerprint the inputs.
_STAGE_CONFIG_SECTIONS: dict[int, list[str]] = {
    1: ["data"],
    2: ["features"],
    3: ["labels"],
    4: ["model", "gru", "validation"],
    5: ["backtest", "labels"],
    6: [],
}


def _cache_hash(config: Config, stage_num: int) -> str:
    """Compute an 8-char SHA-256 fingerprint of config sections relevant to a stage.

    Args:
        config: Application configuration.
        stage_num: Pipeline stage number (1–6).

    Returns:
        Hex digest string, or empty string if no sections are mapped.
    """
    sections = _STAGE_CONFIG_SECTIONS.get(stage_num, [])
    if not sections:
        return ""

    payload: dict[str, Any] = {}
    for name in sections:
        section_cfg = getattr(config, name, None)
        if section_cfg is not None:
            payload[name] = asdict(section_cfg)

    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:8]


def _resolve_cache_path(
    base: str | Path | None,
    invalidation: str,
    config: Config,
    stage_num: int,
) -> Path | None:
    """Resolve the effective cache check path based on invalidation strategy.

    Args:
        base: Raw cache path (e.g. ``data/processed/features.parquet``).
        invalidation: One of ``"path"``, ``"hash"``, ``"none"``.
        config: Application configuration.
        stage_num: Pipeline stage number.

    Returns:
        The resolved ``Path`` to use for cache existence checks, or
        ``None`` when caching is disabled.
    """
    if base is None or invalidation == "none":
        return None

    p = Path(base)
    if invalidation == "hash":
        h = _cache_hash(config, stage_num)
        if h:
            return p.with_stem(f"{p.stem}_{h}")
        return p

    return p


# ---------------------------------------------------------------------------
# Stage runner with cache checking
# ---------------------------------------------------------------------------


def _run_stage(
    stage_num: int,
    config: Config,
    flag_name: str,
    cache_path: str | Path | None,
    work_fn: callable,
) -> None:
    """Execute a pipeline stage with cache checking.

    Checks the workflow flag and optional cache file; skips the stage
    if disabled or cached unless ``force_rerun`` is set.

    Args:
        stage_num: Stage number for console display.
        config: Application configuration.
        flag_name: Workflow boolean flag name on ``config.workflow``.
        cache_path: Path to the cached output file, or ``None`` for
            no cache check.
        work_fn: Callable ``(Config) -> None`` that performs the
            actual stage work.
    """
    flag = getattr(config.workflow, flag_name, False)
    if not flag:
        stage_skip(stage_num, "disabled")
        return

    effective = _resolve_cache_path(
        cache_path,
        config.workflow.cache_invalidation,
        config,
        stage_num,
    )

    if effective is not None:
        if not config.workflow.force_rerun and effective.exists():
            stage_skip(stage_num, f"cached ({effective.name})")
            return

    stage_header(stage_num)
    work_fn(config)

    # Create cache marker so subsequent runs with the same config skip.
    if effective is not None and not effective.exists():
        effective.touch()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(config: Config) -> None:
    """Execute the full thesis pipeline.

    Stages:
        1. Data preparation (tick → OHLCV)
        2. Feature engineering
        3. Triple-barrier labeling
        4. Walk-forward model training (GRU + LightGBM per window)
        5. Backtest (on concatenated OOF predictions)
        6. Report generation

    Args:
        config: Loaded application configuration.
    """
    # Stage 1: Prepare OHLCV from raw ticks
    _run_stage(1, config, "run_data_pipeline", config.paths.ohlcv, prepare_data)

    # Stage 2: Features
    _run_stage(
        2,
        config,
        "run_feature_engineering",
        config.paths.features,
        generate_features,
    )

    # Stage 3: Labels
    _run_stage(3, config, "run_label_generation", config.paths.labels, generate_labels)

    # Stage 4: Training (walk-forward or static)
    if config.validation.method == "sliding":
        stage_header(4)
        logger.info(
            "Using walk-forward sliding window validation (%s architecture)",
            config.model.architecture,
        )
        if config.workflow.run_model_training:
            _run_walk_forward(config)
        else:
            stage_skip(4, "disabled")
    else:
        logger.info("Using static train/val/test split")
        _run_stage(4, config, "run_model_training", None, _run_static_train)

    # Stage 5: Backtest
    if config.workflow.run_backtest:
        tp_l = config.labels.atr_tp_multiplier
        sl_l = config.labels.atr_sl_multiplier
        tp_b = config.backtest.atr_tp_multiplier
        sl_b = config.backtest.atr_stop_multiplier
        if tp_l != tp_b or sl_l != sl_b:
            logger.warning(
                "Label and backtest barrier multipliers differ: "
                "labels (%.1f/%.1f) vs backtest (%.1f/%.1f)",
                tp_l,
                sl_l,
                tp_b,
                sl_b,
            )

    _run_stage(
        5,
        config,
        "run_backtest",
        None,
        run_backtest,
    )

    # Stage 6: Report
    _run_stage(
        6,
        config,
        "run_reporting",
        None,
        generate_report,
    )

    console.print()
    console.rule("[bold green]Pipeline Complete[/]")
    console.print()
