"""Hybrid-vs-static statistical comparison and model-comparison helpers."""

from __future__ import annotations

from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import tomllib
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from polars.exceptions import ColumnNotFoundError

from thesis.shared import baselines as baselines_mod
from thesis.shared.config import Config
from thesis.stage_6_reporting.benchmarks import model_label

logger = logging.getLogger("thesis.report")

MIN_WINDOWS_COMPARISON: int = 3
SIGNIFICANCE_ALPHA: float = 0.05


def parse_date(date_str: str) -> datetime | None:
    """Parse date string trying multiple formats."""
    if not date_str:
        return None

    tz_aware_formats = ("%Y-%m-%d %H:%M:%S%z", "%Y-%m-%dT%H:%M:%S%z")
    for fmt in tz_aware_formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    naive_formats = ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S")
    truncated = date_str[:19]
    for fmt in naive_formats:
        try:
            return datetime.strptime(truncated, fmt)
        except ValueError:
            continue
    return None


def pair_windows_by_date(
    current_windows: list[dict],
    sibling_windows: list[dict],
) -> list[tuple[float, float]]:
    """Pair windows by overlapping test date ranges."""
    paired: list[tuple[float, float]] = []

    for cw in current_windows:
        if "accuracy" not in cw or cw["accuracy"] is None:
            continue
        cd = cw.get("test_dates", {})
        c_start = parse_date(cd.get("start", ""))
        c_end = parse_date(cd.get("end", ""))
        if c_start is None or c_end is None:
            continue

        best_sw = None
        best_overlap = timedelta.min
        for sw in sibling_windows:
            if "accuracy" not in sw or sw["accuracy"] is None:
                continue
            sd = sw.get("test_dates", {})
            s_start = parse_date(sd.get("start", ""))
            s_end = parse_date(sd.get("end", ""))
            if s_start is None or s_end is None:
                continue

            overlap_start = max(c_start, s_start)
            overlap_end = min(c_end, s_end)
            overlap = overlap_end - overlap_start
            if overlap > best_overlap:
                best_overlap = overlap
                best_sw = sw

        if best_sw is not None and best_overlap > timedelta(0):
            paired.append((cw["accuracy"], best_sw["accuracy"]))

    return paired


def find_architecture_session(
    results_dir: Path, target_arch: str, exclude_session: str
) -> Path | None:
    """Find most recent session directory with a given architecture."""
    if not results_dir.exists():
        return None

    candidates: list[tuple[float, Path]] = []
    for session_dir in sorted(results_dir.iterdir()):
        if not session_dir.is_dir():
            continue
        session_str = str(session_dir)
        if session_str == str(exclude_session):
            continue

        snapshot = session_dir / "config" / "config_snapshot.toml"
        if not snapshot.exists():
            continue

        try:
            with open(snapshot, "rb") as f:
                data = tomllib.load(f)
            arch = data.get("model", {}).get("architecture", "")
            if arch == target_arch or (target_arch == "static" and arch == "lgbm"):
                candidates.append((session_dir.stat().st_mtime, session_dir))
        except (OSError, ValueError):
            logger.debug(
                "Skipping session %s during architecture search",
                session_dir.name,
                exc_info=True,
            )
            continue

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def build_model_comparison_rows(
    config: Config, pred_stats: dict | None
) -> list[dict[str, Any]]:
    """Build thesis-level model comparison rows with available metrics."""
    rows: list[dict[str, Any]] = []

    if pred_stats:
        per_class = pred_stats.get("per_class", {})
        reg_aux = pred_stats.get("regression_auxiliary", {})
        rows.append(
            {
                "model": model_label(config),
                "directional_accuracy": pred_stats.get("directional_accuracy"),
                "accuracy": pred_stats.get("accuracy"),
                "macro_f1": pred_stats.get("macro_f1"),
                "long_f1": per_class.get("Long", {}).get("f1"),
                "short_f1": per_class.get("Short", {}).get("f1"),
                "mae_return": reg_aux.get("mae"),
                "rmse_return": reg_aux.get("rmse"),
                "r2_return": reg_aux.get("r_squared"),
                "source": "current_session",
            }
        )

    preds_path = Path(config.paths.predictions)
    if preds_path.exists():
        try:
            df = pl.read_csv(preds_path)
            y_true = df["true_label"].to_numpy()
            close_path = Path(config.paths.ohlcv)
            y_returns = np.zeros(len(y_true), dtype=np.float64)
            if close_path.exists():
                ohlcv = pl.read_parquet(close_path, columns=["close"])
                close = ohlcv["close"].to_numpy()
                if len(close) > 1:
                    bar_returns = np.diff(close) / close[:-1]
                    n = min(len(y_true), len(bar_returns))
                    y_returns = bar_returns[-n:]
                    y_true = y_true[-n:]
            baselines = baselines_mod.run_all_baselines(
                y_true, y_returns, seed=config.workflow.random_seed
            )
            for baseline_key, label in (
                ("naive_direction", "Naive Direction"),
                ("majority_class", "Majority Baseline"),
                ("random", "Random Baseline"),
            ):
                if baseline_key not in baselines:
                    continue
                m = baselines[baseline_key]
                rows.append(
                    {
                        "model": label,
                        "directional_accuracy": m.get("directional_accuracy"),
                        "accuracy": m.get("accuracy"),
                        "macro_f1": m.get("macro_f1"),
                        "long_f1": None,
                        "short_f1": None,
                        "mae_return": None,
                        "rmse_return": None,
                        "r2_return": None,
                        "source": "derived_baseline",
                    }
                )
        except (ColumnNotFoundError, ValueError):
            logger.warning(
                "Failed to build baseline rows for model comparison", exc_info=True
            )

    session_dir = config.paths.session_dir
    existing = {str(r["model"]).lower() for r in rows}
    name_map = {
        "logreg": "Logistic Regression",
        "rf": "Random Forest",
        "lgbm": "LightGBM",
        "hybrid_stacking": "Hybrid Stacking",
    }
    comparison_json = (
        Path(session_dir) / "reports" / "model_comparison.json" if session_dir else None
    )
    if comparison_json and comparison_json.exists():
        try:
            model_comparison = json.loads(comparison_json.read_text())
        except (OSError, json.JSONDecodeError):
            logger.warning("Failed to read model comparison JSON", exc_info=True)
            model_comparison = {}
        for key, metrics in model_comparison.items():
            model_name = name_map.get(key, str(key).replace("_", " ").title())
            if model_name.lower() in existing:
                continue
            per_class = metrics.get("per_class", {})
            rows.append(
                {
                    "model": model_name,
                    "directional_accuracy": metrics.get("directional_accuracy"),
                    "accuracy": metrics.get("accuracy"),
                    "macro_f1": metrics.get("macro_f1"),
                    "long_f1": per_class.get("1", {}).get("f1"),
                    "short_f1": per_class.get("-1", {}).get("f1"),
                    "mae_return": None,
                    "rmse_return": None,
                    "r2_return": None,
                    "source": "walk_forward_model_comparison",
                }
            )
            existing.add(model_name.lower())

    for model_name in (
        "Logistic Regression",
        "Random Forest",
        "LightGBM",
        "Hybrid Stacking",
    ):
        if model_name.lower() in existing:
            continue
        rows.append(
            {
                "model": model_name,
                "directional_accuracy": None,
                "accuracy": None,
                "macro_f1": None,
                "long_f1": None,
                "short_f1": None,
                "mae_return": None,
                "rmse_return": None,
                "r2_return": None,
                "source": "pending_experiment",
            }
        )
    return rows


def write_model_comparison_artifacts(out_dir: Path, rows: list[dict[str, Any]]) -> Path:
    """Write model comparison table to CSV."""
    csv_path = out_dir / "model_comparison.csv"
    frame = pd.DataFrame(rows)
    frame.to_csv(csv_path, index=False)
    return csv_path
