"""Classification and regression metric functions."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

DEFAULT_CLASSES: list[int] = [-1, 1]
DEFAULT_CLASS_NAMES: dict[int, str] = {-1: "Short", 0: "Hold", 1: "Long"}


def accuracy(y_true: npt.NDArray, y_pred: npt.NDArray) -> float:
    """Fraction of correct predictions."""
    return float((y_true == y_pred).mean())


def balanced_accuracy(
    y_true: npt.NDArray, y_pred: npt.NDArray, classes: list[int] | None = None
) -> float:
    """Balanced accuracy (mean recall per class)."""
    if classes is None:
        classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    recalls: list[float] = []
    for c in classes:
        mask = y_true == c
        if mask.sum() > 0:
            recalls.append(float((y_pred[mask] == c).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def directional_accuracy(y_true: npt.NDArray, y_pred: npt.NDArray) -> float:
    """Directional accuracy for Short/Long labels."""
    return accuracy(y_true, y_pred)


def mda_no_hold(y_true: npt.NDArray, y_pred: npt.NDArray) -> float:
    """Directional accuracy retained for report compatibility."""
    return accuracy(y_true, y_pred)


def mda_including_hold(y_true: npt.NDArray, y_pred: npt.NDArray) -> float:
    """Directional accuracy retained for report compatibility."""
    return accuracy(y_true, y_pred)


def mda_binary(y_true: npt.NDArray, y_pred: npt.NDArray) -> float:
    """Binary directional accuracy (Long/Short only)."""
    return accuracy(y_true, y_pred)


def _prf_for_class(
    y_true: npt.NDArray, y_pred: npt.NDArray, cls: int
) -> tuple[float, float, float]:
    true_mask = y_true == cls
    pred_mask = y_pred == cls
    rec = float((y_pred[true_mask] == cls).mean()) if true_mask.sum() > 0 else 0.0
    prec = float((y_true[pred_mask] == cls).mean()) if pred_mask.sum() > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def macro_f1(
    y_true: npt.NDArray, y_pred: npt.NDArray, classes: list[int] | None = None
) -> float:
    """Macro-averaged F1 score."""
    if classes is None:
        classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    f1s = [_prf_for_class(y_true, y_pred, c)[2] for c in classes]
    return float(np.mean(f1s))


def weighted_f1(
    y_true: npt.NDArray, y_pred: npt.NDArray, classes: list[int] | None = None
) -> float:
    """Weighted F1 by class support."""
    if classes is None:
        classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    total_f1 = 0.0
    total_support = 0
    for c in classes:
        support = int((y_true == c).sum())
        _, _, f1 = _prf_for_class(y_true, y_pred, c)
        total_f1 += f1 * support
        total_support += support
    return total_f1 / total_support if total_support > 0 else 0.0


def precision_recall_f1_per_class(
    y_true: npt.NDArray,
    y_pred: npt.NDArray,
    classes: list[int] | None = None,
    class_names: dict[int, str] | None = None,
) -> dict[str, dict[str, float]]:
    """Per-class precision/recall/F1."""
    if classes is None:
        classes = DEFAULT_CLASSES
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES
    result: dict[str, dict[str, float]] = {}
    for c in classes:
        prec, rec, f1 = _prf_for_class(y_true, y_pred, c)
        result[class_names.get(c, str(c))] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }
    return result


def confusion_matrix(
    y_true: npt.NDArray,
    y_pred: npt.NDArray,
    classes: list[int] | None = None,
    class_names: dict[int, str] | None = None,
) -> dict[str, dict[str, int]]:
    """Confusion matrix dict."""
    if classes is None:
        classes = DEFAULT_CLASSES
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES
    cm: dict[str, dict[str, int]] = {}
    for tc in classes:
        row: dict[str, int] = {}
        for pc in classes:
            row[class_names.get(pc, str(pc))] = int(
                ((y_true == tc) & (y_pred == pc)).sum()
            )
        cm[class_names.get(tc, str(tc))] = row
    return cm


def direction_confusion_matrix(
    y_true: npt.NDArray,
    y_pred: npt.NDArray,
    classes: list[int] | None = None,
    class_names: dict[int, str] | None = None,
) -> dict[str, dict[str, int]]:
    """Directional confusion matrix for directional class labels."""
    if classes is None:
        classes = [-1, 1]
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES
    cm: dict[str, dict[str, int]] = {}
    for tc in classes:
        row: dict[str, int] = {}
        for pc in classes:
            row[class_names.get(pc, str(pc))] = int(
                ((y_true == tc) & (y_pred == pc)).sum()
            )
        cm[class_names.get(tc, str(tc))] = row
    return cm


def majority_baseline_accuracy(
    y_true: npt.NDArray, classes: list[int] | None = None
) -> float:
    """Baseline accuracy (most frequent class)."""
    if classes is None:
        classes = DEFAULT_CLASSES
    n = len(y_true)
    if n == 0:
        return 0.0
    return float(max((y_true == c).sum() for c in classes) / n)


def high_confidence_accuracy(
    y_true: npt.NDArray,
    y_pred: npt.NDArray,
    y_proba: npt.NDArray,
    threshold: float = 0.70,
) -> dict[str, float | int]:
    """Accuracy on high-confidence predictions."""
    max_proba = y_proba.max(axis=1)
    mask = max_proba >= threshold
    count = int(mask.sum())
    total = len(y_true)
    if count == 0:
        return {"accuracy": 0.0, "count": 0, "pct_of_total": 0.0}
    acc = float((y_true[mask] == y_pred[mask]).mean())
    return {"accuracy": acc, "count": count, "pct_of_total": count / total}


def mae(y_true: npt.NDArray, y_pred: npt.NDArray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: npt.NDArray, y_pred: npt.NDArray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r_squared(y_true: npt.NDArray, y_pred: npt.NDArray) -> float:
    """R-squared (coefficient of determination)."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def compute_proxy_return(
    y_proba: npt.NDArray, classes: list[int] | None = None
) -> npt.NDArray:
    """Proxy returns from class probabilities."""
    if classes is None:
        classes = DEFAULT_CLASSES
    labels = np.array(classes, dtype=np.float64)
    return y_proba @ labels


def compute_regression_auxiliary(
    y_true_returns: npt.NDArray, y_pred_returns: npt.NDArray
) -> dict[str, float]:
    """Compute regression auxiliary metrics."""
    return {
        "mae": mae(y_true_returns, y_pred_returns),
        "rmse": rmse(y_true_returns, y_pred_returns),
        "r_squared": r_squared(y_true_returns, y_pred_returns),
    }


def compute_all_classification_metrics(
    y_true: npt.NDArray,
    y_pred: npt.NDArray,
    y_proba: npt.NDArray | None = None,
    classes: list[int] | None = None,
    class_names: dict[int, str] | None = None,
    y_true_returns: npt.NDArray | None = None,
    y_pred_returns: npt.NDArray | None = None,
) -> dict:
    """Compute all classification metrics."""
    if classes is None:
        classes = DEFAULT_CLASSES
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    result: dict = {
        "total": len(y_true),
        "accuracy": accuracy(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred, classes),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
        "mda_no_hold": mda_no_hold(y_true, y_pred),
        "mda_including_hold": mda_including_hold(y_true, y_pred),
        "mda_binary": mda_binary(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred, classes),
        "weighted_f1": weighted_f1(y_true, y_pred, classes),
        "precision_recall_f1_per_class": precision_recall_f1_per_class(
            y_true, y_pred, classes, class_names
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, classes, class_names),
        "direction_confusion_matrix": direction_confusion_matrix(
            y_true, y_pred, classes, class_names
        ),
        "majority_baseline_accuracy": majority_baseline_accuracy(y_true, classes),
    }

    if y_proba is not None:
        result["high_confidence_accuracy"] = high_confidence_accuracy(
            y_true, y_pred, y_proba
        )

    if y_true_returns is not None:
        if y_pred_returns is not None:
            pred_returns = y_pred_returns
        elif y_proba is not None:
            pred_returns = compute_proxy_return(y_proba, classes)
        else:
            pred_returns = None

        if pred_returns is not None:
            result["regression_auxiliary"] = compute_regression_auxiliary(
                y_true_returns, pred_returns
            )

    return result
