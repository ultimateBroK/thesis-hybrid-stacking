"""SHAP analysis and feature importance."""

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from thesis.config import Config
from thesis.hybrid.lgbm import _wrap_np

logger = logging.getLogger("thesis.hybrid.interpret")


def _compute_shap(
    model: Any, X_test: np.ndarray, feature_cols: list[str], config: Config
) -> None:
    """
    Compute SHAP values for a subset of the test set and write a SHAP summary plot to disk.
    
    This function computes SHAP values for up to 500 rows from `X_test`, renders a SHAP summary plot, and saves the plot as `shap_summary.png` under `<session_dir>/reports/` when `config.paths.session_dir` is set or `results/shap_summary.png` otherwise. On success it logs an informational message with the number of samples processed and elapsed time; on any failure it logs a warning and exits silently.
    
    Parameters:
        model: A tree-based predictive model compatible with SHAP's TreeExplainer.
        X_test (np.ndarray): Feature matrix from which up to 500 rows are sampled for SHAP computation.
        feature_cols (list[str]): Ordered list of feature names corresponding to columns in `X_test`.
        config: Configuration object. Uses `config.workflow.random_seed` to seed plot randomness and `config.paths.session_dir` to determine the output directory.
    """
    try:
        import shap

        n_samples = min(500, len(X_test))
        shap_start = time.perf_counter()

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]SHAP analysis"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            transient=False,
        )

        with progress:
            task = progress.add_task("steps", total=2)

            # Step 1: Compute SHAP values
            progress.update(task, description="[bold cyan]SHAP computing")
            explainer = shap.TreeExplainer(model)
            X_sample = _wrap_np(X_test[:n_samples], feature_cols)
            shap_values = explainer.shap_values(X_sample)
            progress.update(task, advance=1)

            # Step 2: Render plot
            progress.update(task, description="[bold cyan]SHAP rendering")
            if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                shap_values = [
                    shap_values[:, :, i] for i in range(shap_values.shape[2])
                ]

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
            progress.update(task, advance=1)

        shap_time = time.perf_counter() - shap_start
        logger.info("SHAP done: %d samples, %.1fs", n_samples, shap_time)
    except Exception as e:
        logger.warning("SHAP computation failed: %s", e)


def _save_feature_importance(
    model: Any, feature_cols: list[str], config: Config
) -> None:
    """
    Write the model's feature importances to a JSON file, sorted by descending importance.
    
    The output is a JSON object mapping feature names to their importance values (floats), pretty-printed with an indentation of 2. If `config.paths.session_dir` is set the file is written to `<session_dir>/reports/feature_importance.json`, otherwise it is written to `results/feature_importance.json`. Parent directories are created if needed. On success the top five feature names are logged; on failure a warning is emitted.
    """
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
            "Feature importance saved (top 5: %s)",
            [p[0] for p in pairs[:5]],
        )
    except Exception as e:
        logger.warning("Feature importance save failed: %s", e)
