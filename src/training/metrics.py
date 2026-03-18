"""
Classification metrics for binary medical image classification.
"""

import logging

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute classification metrics for binary classification.

    Args:
        y_true: ground truth labels (0 or 1)
        y_pred: predicted labels (0 or 1)
        y_prob: predicted probabilities for positive class (optional)

    Returns:
        dict of metric_name -> value
    """
    metrics: dict[str, float] = {}

    # Basic metrics
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0.0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0.0))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0.0))

    # AUC-ROC (requires probabilities and both classes present)
    if y_prob is not None:
        n_classes = len(np.unique(y_true))
        if n_classes < 2:
            logger.warning("AUC undefined (single class in y_true). Setting to 0.0.")
            metrics["auc"] = 0.0
        else:
            try:
                auc_val = float(roc_auc_score(y_true, y_prob))
                metrics["auc"] = 0.0 if np.isnan(auc_val) else auc_val
            except ValueError:
                logger.warning("AUC computation failed. Setting to 0.0.")
                metrics["auc"] = 0.0

    # Confusion matrix components with safety guard
    try:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        metrics["tn"] = float(tn)
        metrics["fp"] = float(fp)
        metrics["fn"] = float(fn)
        metrics["tp"] = float(tp)
        metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    except ValueError:
        logger.warning("Confusion matrix computation failed. Setting components to 0.")
        for key in ("tn", "fp", "fn", "tp", "sensitivity", "specificity"):
            metrics[key] = 0.0

    return metrics


def aggregate_metrics(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate metrics across multiple runs/folds.

    Args:
        metrics_list: list of metric dicts

    Returns:
        dict with mean and std for each metric
    """
    if not metrics_list:
        return {}

    aggregated: dict[str, float] = {}
    keys = metrics_list[0].keys()

    for key in keys:
        values = [
            float(m[key])
            for m in metrics_list
            if isinstance(m[key], (int, float, np.integer, np.floating))
        ]
        if values:
            aggregated[f"{key}_mean"] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))

    return aggregated
