"""Pipeline: data → dataset → models → reporting."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from thesis.data.prepare_dataset import prepare_dataset
from thesis.dataset.build_features import build_features
from thesis.dataset.build_labels import build_labels
from thesis.dataset.build_ml_dataset import build_ml_dataset
from thesis.demo.backtest_demo import run_backtest_demo
from thesis.models.train import train_walk_forward
from thesis.reporting.report import generate_report
from thesis.shared.config import Config
from thesis.shared.utils import console, stage_header, stage_skip

logger = logging.getLogger("thesis.pipeline")

# Config sections used for cache fingerprinting per stage.
_STAGE_CONFIG_SECTIONS: dict[int, list[str]] = {
    1: ["data"],
    2: ["features", "labels"],
    3: ["model", "validation"],
    4: [],
}


def _cache_hash(config: Config, stage_num: int) -> str:
    """8-char SHA-256 fingerprint of stage-relevant config sections."""
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
    """Return cache Path or None (disabled).

    ``invalidation`` controls strategy:
    - ``path``: use base path directly
    - ``hash``: embed config fingerprint into stem
    - ``none``: caching disabled, always recompute
    """
    if base is None or invalidation == "none":
        return None

    p = Path(base)
    if invalidation == "hash":
        h = _cache_hash(config, stage_num)
        return p.with_stem(f"{p.stem}_{h}") if h else p

    return p


def _run_stage(
    stage_num: int,
    config: Config,
    flag_name: str,
    cache_path: str | Path | None,
    work_fn: callable,
) -> None:
    """Execute stage if enabled, skip on cache hit unless force_rerun."""
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

    if effective is not None and not config.workflow.force_rerun and effective.exists():
        stage_skip(stage_num, f"cached ({effective.name})")
        return

    stage_header(stage_num)
    work_fn(config)

    # Touch cache marker so later runs skip.
    if effective is not None and not effective.exists():
        effective.touch()


def _run_dataset_stage(config: Config) -> None:
    """Minimal features → triple-barrier labels → ML dataset assembly."""
    build_features(config)
    build_labels(config)
    build_ml_dataset(config)


def _run_reporting_stage(config: Config) -> None:
    """Report generation + optional application demo."""
    if config.workflow.run_backtest_demo:
        bt_path = Path(config.paths.backtest_results)
        if not bt_path.exists() or config.workflow.force_rerun:
            logger.info("Running backtest demo")
            run_backtest_demo(config)
        else:
            logger.info("Backtest results cached: %s", bt_path)
    generate_report(config)


def run_pipeline(config: Config) -> None:
    """Run full pipeline: data → dataset → models → reporting."""
    # Stage 1: Market data preparation (raw ticks -> OHLCV+ bars).
    _run_stage(1, config, "run_data", config.paths.ohlcv, prepare_dataset)

    # Stage 2: ML dataset construction (features -> labels -> dataset).
    _run_stage(2, config, "run_dataset", config.paths.ml_dataset, _run_dataset_stage)

    # Stage 3: Walk-forward model training & evaluation.
    _run_stage(3, config, "run_models", config.paths.model, train_walk_forward)

    # Stage 4: Report generation + optional application demo.
    _run_stage(4, config, "run_reporting", None, _run_reporting_stage)

    console.print()
    console.rule("[bold green]Pipeline Complete[/]")
    console.print()
