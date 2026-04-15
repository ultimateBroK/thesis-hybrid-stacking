"""SHAP analysis and feature importance."""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from thesis.config import Config
from thesis.hybrid.lgbm import _wrap_np

logger = logging.getLogger("thesis.hybrid.interpret")


def _compute_shap(
    model: Any, X_test: np.ndarray, feature_cols: list[str], config: Config
) -> None:
    """Compute and save SHAP summary."""
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        n_samples = min(500, len(X_test))
        X_sample = _wrap_np(X_test[:n_samples], feature_cols)
        shap_values = explainer.shap_values(X_sample)

        # Multiclass models return 3-D (samples × features × classes).
        # Convert to a list of 2-D arrays so summary_plot handles each
        # class correctly instead of misinterpreting as interaction values.
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 8))
        rng = np.random.default_rng(config.workflow.random_seed)
        shap.summary_plot(
            shap_values, X_sample, feature_names=feature_cols, show=False, rng=rng
        )
        if config.paths.session_dir:
            out = Path(config.paths.session_dir) / "reports" / "shap_summary.png"
        else:
            out = Path("results/shap_summary.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("SHAP summary saved: %s", out)
    except Exception as e:
        logger.warning("SHAP computation failed: %s", e)


def _save_feature_importance(
    model: Any, feature_cols: list[str], config: Config
) -> None:
    """Save feature importance as JSON."""
    try:
        imp = model.feature_importances_
        pairs = sorted(zip(feature_cols, imp), key=lambda x: x[1], reverse=True)
        if config.paths.session_dir:
            out_path = (
                Path(config.paths.session_dir) / "reports" / "feature_importance.json"
            )
        else:
            out_path = Path("results/feature_importance.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({name: float(val) for name, val in pairs}, f, indent=2)
        logger.info(
            "Feature importance saved: %s (top 5: %s)",
            out_path,
            [p[0] for p in pairs[:5]],
        )
    except Exception as e:
        logger.warning("Feature importance save failed: %s", e)
