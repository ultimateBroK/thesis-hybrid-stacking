"""Merge features + labels -> ml_dataset.parquet."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import polars as pl

from thesis.shared.config import Config
from thesis.shared.constants import (
    LABEL_META_COLS,
    build_feature_output_cols,
    get_static_feature_cols,
)
from thesis.shared.utils import console

logger = logging.getLogger("thesis.dataset.build_ml_dataset")


def build_ml_dataset(config: Config) -> None:
    """Join features.parquet + labels.parquet -> ml_dataset.parquet.

    Writes:
        - data/modeling/ml_dataset.parquet
        - reports/label_distribution.json
        - reports/feature_list.json
    """
    features_path = Path(config.paths.features)
    labels_path = Path(config.paths.labels)
    out_path = Path(config.paths.ml_dataset)

    if not features_path.exists():
        raise FileNotFoundError(f"Features not found: {features_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels not found: {labels_path}")

    logger.info("Loading features: %s", features_path)
    features = pl.read_parquet(features_path)
    logger.info("Loading labels: %s", labels_path)
    labels = pl.read_parquet(labels_path)

    logger.info("Features: %d rows, %d cols", len(features), len(features.columns))
    logger.info("Labels:   %d rows, %d cols", len(labels), len(labels.columns))

    # -- join --
    if "timestamp" in features.columns and "timestamp" in labels.columns:
        # prefer timestamp-based join for safety
        df = features.join(labels, on="timestamp", how="inner")
    elif len(features) == len(labels):
        # same row count — positional concat
        df = pl.concat(
            [
                features,
                labels.drop([c for c in labels.columns if c in features.columns]),
            ],
            how="horizontal",
        )
    else:
        raise ValueError(
            f"Cannot join: features={len(features)} rows, labels={len(labels)} rows, "
            "and no shared timestamp column."
        )

    logger.info("Joined: %d rows, %d cols", len(df), len(df.columns))

    # -- drop NaN label rows --
    if "label" in df.columns:
        n_before = len(df)
        df = df.filter(pl.col("label").is_not_null())
        df = df.filter(~pl.col("label").is_nan())
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            logger.info("Dropped %d rows with NaN labels", n_dropped)

    if df.is_empty():
        raise ValueError("ML dataset is empty after dropping NaN labels")

    # -- validate columns --
    expected = set(build_feature_output_cols(config) + LABEL_META_COLS)
    missing = expected - set(df.columns)
    if missing:
        logger.warning("Missing expected columns (non-fatal): %s", sorted(missing))

    # -- write outputs --
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    logger.info(
        "Saved ml_dataset: %s (%d rows, %d cols)", out_path, len(df), len(df.columns)
    )

    reports_dir = out_path.parent.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # label_distribution.json
    if "label" in df.columns:
        dist = df["label"].value_counts().sort("label")
        dist_dict = {
            str(row["label"]): int(row["count"]) for row in dist.iter_rows(named=True)
        }
        _write_json(reports_dir / "label_distribution.json", dist_dict)
        logger.info("Label distribution: %s", dist_dict)

    # feature_list.json — model-facing columns only
    model_cols = sorted(
        c for c in df.columns if c in set(get_static_feature_cols(config))
    )
    _write_json(
        reports_dir / "feature_list.json",
        {"features": model_cols, "count": len(model_cols)},
    )
    logger.info("Feature columns (%d): %s", len(model_cols), model_cols)

    # -- summary --
    console.rule("ML Dataset Summary")
    console.print(f"  Rows: {len(df)}")
    console.print(f"  Columns: {len(df.columns)}")
    console.print(f"  Feature columns: {len(model_cols)}")
    if "label" in df.columns:
        console.print(f"  Label distribution: {dist_dict}")


def _write_json(path: Path, data: dict) -> None:
    """Write dict as indented JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.debug("Wrote %s", path)
