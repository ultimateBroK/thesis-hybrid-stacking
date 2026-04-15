"""Static matplotlib/seaborn model performance charts."""

import json
import logging
from pathlib import Path

import polars as pl

from thesis.config import Config

from .data import _COLORS, _output_dir

logger = logging.getLogger("thesis.visualize")


def _generate_model_charts(config: Config) -> None:
    """Generate model performance charts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    out = _output_dir(config, "model")

    # Load predictions
    preds_path = Path(config.paths.predictions)
    if not preds_path.exists():
        logger.warning("Predictions not found: %s", preds_path)
        return

    preds_df = pl.read_parquet(preds_path)
    y_true = preds_df["true_label"].to_numpy()
    y_pred = preds_df["pred_label"].to_numpy()

    # --- 1. Confusion Matrix ---
    fig, ax = plt.subplots(figsize=(8, 6))
    labels_order = [-1, 0, 1]
    display_labels = ["Short (-1)", "Flat (0)", "Long (1)"]
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=labels_order,
        display_labels=display_labels,
        cmap="Blues",
        ax=ax,
        normalize="true",
    )
    ax.set_title("Normalized Confusion Matrix (Test Set)")
    ax.grid(False)
    fig.savefig(out / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart: confusion_matrix.png")

    # --- 2. Prediction Confidence Distribution ---
    if "pred_proba_class_1" in preds_df.columns:
        long_conf = preds_df["pred_proba_class_1"].to_numpy()
        short_conf = preds_df["pred_proba_class_minus1"].to_numpy()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(
            long_conf[y_pred == 1],
            bins=50,
            alpha=0.6,
            color=_COLORS["long"],
            label="Long confidence",
        )
        ax.hist(
            short_conf[y_pred == -1],
            bins=50,
            alpha=0.6,
            color=_COLORS["short"],
            label="Short confidence",
        )
        ax.set_title("Prediction Confidence Distribution")
        ax.set_xlabel("Confidence (max softmax probability)")
        ax.set_ylabel("Count")
        ax.legend()
        fig.savefig(out / "confidence_distribution.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Chart: confidence_distribution.png")

    # --- 3. Feature Importance ---
    if config.paths.session_dir:
        fi_path = Path(config.paths.session_dir) / "reports" / "feature_importance.json"
    else:
        fi_path = Path("results/feature_importance.json")

    if fi_path.exists():
        with open(fi_path) as f:
            fi = json.load(f)

        # Separate GRU and static features
        top_n = 20
        items = list(fi.items())[:top_n]
        names = [n for n, _ in items]
        values = [v for _, v in items]
        colors = [
            _COLORS["secondary"] if n.startswith("gru_") else _COLORS["primary"]
            for n in names
        ]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(names, values, color=colors)
        ax.set_title(f"Feature Importance (Top {top_n})")
        ax.invert_yaxis()

        # Legend for feature types
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor=_COLORS["secondary"], label="GRU hidden state"),
            Patch(facecolor=_COLORS["primary"], label="Static feature"),
        ]
        ax.legend(handles=legend_elements, loc="lower right")

        fig.savefig(out / "feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Chart: feature_importance.png")
