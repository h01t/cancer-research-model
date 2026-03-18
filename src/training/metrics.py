import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def compute_metrics(y_true, y_pred, y_prob=None):
    """Compute classification metrics.

    Args:
        y_true: ground truth labels (binary)
        y_pred: predicted labels (binary)
        y_prob: predicted probabilities for positive class (optional)

    Returns:
        dict of metrics
    """
    metrics = {}

    # Basic metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

    # AUC if probabilities available
    if y_prob is not None:
        try:
            metrics["auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["auc"] = 0.0

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["tn"] = tn
    metrics["fp"] = fp
    metrics["fn"] = fn
    metrics["tp"] = tp
    metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return metrics


def aggregate_metrics(metrics_list):
    """Aggregate metrics across multiple runs/folds.

    Args:
        metrics_list: list of metric dicts

    Returns:
        dict with mean and std for each metric
    """
    if not metrics_list:
        return {}

    aggregated = {}
    keys = metrics_list[0].keys()

    for key in keys:
        values = [m[key] for m in metrics_list if isinstance(m[key], (int, float))]
        if values:
            aggregated[f"{key}_mean"] = np.mean(values)
            aggregated[f"{key}_std"] = np.std(values)

    return aggregated
