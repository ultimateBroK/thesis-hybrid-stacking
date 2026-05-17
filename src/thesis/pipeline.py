"""Pipeline: data → features → labels → train → backtest → report."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from thesis.shared.config import Config
from thesis.shared.ui import console, stage_header, stage_skip
from thesis.stage_1_data import generate_data
from thesis.stage_2_features import generate_features
from thesis.stage_3_labels import generate_labels
from thesis.stage_4_training.walk_forward import train_walk_forward
from thesis.stage_5_backtest import run_backtest
from thesis.stage_6_reporting import generate_report

logger = logging.getLogger("thesis.pipeline")

# Config sections used for cache fingerprinting per stage.
_STAGE_CONFIG_SECTIONS: dict[int, list[str]] = {
    1: ["data"],
    2: ["features"],
    3: ["labels"],
    4: ["model", "validation"],
    5: ["backtest", "labels"],
    6: [],
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


def _run_backtest_with_barrier_guard(config: Config) -> None:
    """Backtest only when label and execution ATR barriers match."""
    label_tp = config.labels.atr_tp_multiplier
    label_sl = config.labels.atr_sl_multiplier
    backtest_tp = config.backtest.atr_tp_multiplier
    backtest_sl = config.backtest.atr_stop_multiplier
    if label_tp != backtest_tp or label_sl != backtest_sl:
        raise ValueError(
            f"Label/Backtest ATR mismatch: labels(tp={label_tp}, sl={label_sl}) "
            f"!= backtest(tp={backtest_tp}, sl={backtest_sl}). "
            "Training target and execution exits must measure same event."
        )
    run_backtest(config)


def run_pipeline(config: Config) -> None:
    """Run full pipeline: data → features → labels → train → backtest → report."""
    # Stage 1: OHLCV from raw ticks.
    _run_stage(1, config, "run_data_pipeline", config.paths.ohlcv, generate_data)

    # Stage 2: Technical indicators.
    _run_stage(
        2, config, "run_feature_engineering", config.paths.features, generate_features
    )

    # Stage 3: Triple-barrier directional labels.
    _run_stage(3, config, "run_label_generation", config.paths.labels, generate_labels)

    # Stage 4: Walk-forward Hybrid Stacking.
    if config.workflow.run_model_training:
        stage_header(4)
        train_walk_forward(config)
    else:
        stage_skip(4, "disabled")

    # Stage 5: Optional backtest simulation.
    _run_stage(5, config, "run_backtest", None, _run_backtest_with_barrier_guard)

    # Stage 6: Markdown + HTML artefacts.
    _run_stage(6, config, "run_reporting", None, generate_report)

    console.print()
    console.rule("[bold green]Pipeline Complete[/]")
    console.print()
