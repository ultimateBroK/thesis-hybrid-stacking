"""Report data loading helpers."""

import logging
from pathlib import Path

import polars as pl

from thesis.config import Config

logger = logging.getLogger("thesis.report.stats")


def _load_parquet_stats(path: Path) -> dict | None:
    """
    Load basic stats from a parquet file without loading full data.

    Reads only row/column metadata to provide a lightweight summary of the file contents.
    Useful for quickly checking file size and date range without loading entire datasets.

    Parameters:
        path: Path to the parquet file to inspect.

    Returns:
        Dictionary with 'rows', 'columns', and optionally 'date_range' (tuple of min/max timestamps),
        or None if the file doesn't exist or cannot be read.
    """
    if not path.exists():
        return None
    try:
        df = pl.read_parquet(path, columns=None)
        stats: dict = {"rows": len(df), "columns": df.columns}
        if "timestamp" in df.columns:
            ts = df["timestamp"]
            stats["date_range"] = (
                str(ts.min()),
                str(ts.max()),
            )
        return stats
    except Exception:
        return None


def _load_label_distribution(labels_path: Path) -> dict | None:
    """
    Load label class distribution from labels parquet.

    Computes the count and percentage of each label class (-1=Short, 0=Hold, 1=Long)
    in the dataset. Used for reporting class balance in the thesis document.

    Parameters:
        labels_path: Path to the labels parquet file.

    Returns:
        Dictionary mapping class names ('Short', 'Hold', 'Long') to (count, percentage) tuples,
        plus a 'total' key with the total row count. Returns None if file doesn't exist or cannot be read.
    """
    if not labels_path.exists():
        return None
    try:
        df = pl.read_parquet(labels_path, columns=["label"])
        total = len(df)
        dist: dict[str, tuple[int, float]] = {}
        for label_val, name in [(-1, "Short"), (0, "Hold"), (1, "Long")]:
            count = (df["label"] == label_val).sum()
            dist[name] = (count, count / total * 100 if total > 0 else 0)
        dist["total"] = total
        return dist
    except Exception:
        return None


def _load_split_stats(config: Config) -> dict:
    """
    Load row counts and label distributions per split (train/val/test).

    Reads the train, validation, and test parquet files specified in the config,
    and collects basic statistics for each: row count, column count, date range,
    and per-class label distribution.

    Parameters:
        config: Application configuration with paths to train_data, val_data, and test_data.

    Returns:
        Dictionary keyed by split name ('train', 'val', 'test'), each containing
        'rows', 'columns', 'date_range', and optionally 'label_distribution'.
    """
    splits = {}
    for name, path_str in [
        ("train", config.paths.train_data),
        ("val", config.paths.val_data),
        ("test", config.paths.test_data),
    ]:
        path = Path(path_str)
        if not path.exists():
            continue
        try:
            df = pl.read_parquet(path)
            info: dict = {"rows": len(df), "columns": len(df.columns)}
            if "label" in df.columns:
                total = len(df)
                label_dist = {}
                for lv, ln in [(-1, "Short"), (0, "Hold"), (1, "Long")]:
                    c = int((df["label"] == lv).sum())
                    label_dist[ln] = (c, c / total * 100 if total else 0)
                info["label_distribution"] = label_dist
            if "timestamp" in df.columns:
                info["date_range"] = (
                    str(df["timestamp"].min()),
                    str(df["timestamp"].max()),
                )
            splits[name] = info
        except Exception:
            continue
    return splits


def _load_prediction_stats(preds_path: Path) -> dict | None:
    """
    Load model prediction performance statistics from predictions parquet.

    Computes overall accuracy, per-class precision/recall/F1, confusion matrix,
    and optionally high-confidence accuracy (predictions where max probability >= 70%%).
    This provides a comprehensive classification performance summary for the thesis report.

    Parameters:
        preds_path: Path to the predictions parquet file containing
            'true_label', 'pred_label', and optionally probability columns.

    Returns:
        Dictionary with 'total', 'accuracy', 'majority_baseline', 'per_class' metrics,
        'confusion_matrix', and optionally 'high_confidence' stats.
        Returns None if file doesn't exist or cannot be read.
    """
    if not preds_path.exists():
        return None
    try:
        cols = ["true_label", "pred_label"]
        proba_cols = [
            "pred_proba_class_minus1",
            "pred_proba_class_0",
            "pred_proba_class_1",
        ]
        # Try loading with probability columns
        try:
            df = pl.read_parquet(preds_path)
        except Exception:
            df = pl.read_parquet(preds_path, columns=cols)

        true = df["true_label"].to_numpy()
        pred = df["pred_label"].to_numpy()
        total = len(true)

        # Overall accuracy: fraction of predictions matching true labels
        accuracy = float((true == pred).mean())
        # Majority baseline: accuracy if we always predict the most common class
        majority_baseline = float(max((true == lv).sum() for lv in [-1, 0, 1]) / total)

        # Per-class metrics: precision, recall, F1 for each class
        per_class = {}
        for lv, ln in [(-1, "Short"), (0, "Hold"), (1, "Long")]:
            true_mask = true == lv
            pred_mask = pred == lv
            recall = float((pred[true_mask] == lv).mean()) if true_mask.sum() > 0 else 0
            precision = (
                float((true[pred_mask] == lv).mean()) if pred_mask.sum() > 0 else 0
            )
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )
            per_class[ln] = {
                "true_count": int(true_mask.sum()),
                "pred_count": int(pred_mask.sum()),
                "recall": recall,
                "precision": precision,
                "f1": f1,
            }

        # Confusion matrix: rows=true labels, cols=predicted labels, values=count
        cm = {}
        for true_lv, true_name in [(-1, "Short"), (0, "Hold"), (1, "Long")]:
            row = {}
            for pred_lv, pred_name in [(-1, "Short"), (0, "Hold"), (1, "Long")]:
                row[pred_name] = int(((true == true_lv) & (pred == pred_lv)).sum())
            cm[true_name] = row

        result: dict = {
            "total": total,
            "accuracy": accuracy,
            "majority_baseline": majority_baseline,
            "per_class": per_class,
            "confusion_matrix": cm,
        }

        # Confidence-filtered accuracy: evaluate only on high-confidence predictions
        # where max probability >= 70%% threshold
        has_proba = all(c in df.columns for c in proba_cols)
        if has_proba:
            proba = df.select(proba_cols).to_numpy()
            max_proba = proba.max(axis=1)
            threshold = 0.70
            hc_mask = max_proba >= threshold
            if hc_mask.sum() > 0:
                hc_acc = float((true[hc_mask] == pred[hc_mask]).mean())
                hc_total = int(hc_mask.sum())
                # Directional (non-hold) accuracy
                non_hold = pred[hc_mask] != 0
                if non_hold.sum() > 0:
                    dir_acc = float(
                        (true[hc_mask][non_hold] == pred[hc_mask][non_hold]).mean()
                    )
                else:
                    dir_acc = 0
                result["high_confidence"] = {
                    "threshold": threshold,
                    "count": hc_total,
                    "pct_of_total": hc_total / total * 100,
                    "accuracy": hc_acc,
                    "directional_accuracy": dir_acc,
                }

        return result
    except Exception:
        return None
