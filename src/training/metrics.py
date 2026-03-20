"""
Classification metrics for binary medical image classification.
"""

import logging
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
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
    """Compute classification metrics for binary classification."""
    metrics: dict[str, float] = {}

    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0.0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0.0))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0.0))

    if y_prob is not None:
        n_classes = len(np.unique(y_true))
        if n_classes < 2:
            logger.warning("AUC undefined (single class in y_true). Setting to 0.0.")
            metrics["auc"] = 0.0
            metrics["pr_auc"] = 0.0
        else:
            try:
                auc_val = float(roc_auc_score(y_true, y_prob))
                metrics["auc"] = 0.0 if np.isnan(auc_val) else auc_val
            except ValueError:
                logger.warning("AUC computation failed. Setting to 0.0.")
                metrics["auc"] = 0.0
            try:
                pr_auc = float(average_precision_score(y_true, y_prob))
                metrics["pr_auc"] = 0.0 if np.isnan(pr_auc) else pr_auc
            except ValueError:
                logger.warning("PR AUC computation failed. Setting to 0.0.")
                metrics["pr_auc"] = 0.0

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


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """Compute simple calibration metrics for binary probabilities."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if len(y_true) == 0:
        return {"brier_score": 0.0, "ece": 0.0}

    brier = float(brier_score_loss(y_true, y_prob))
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1], right=True)

    ece = 0.0
    for bin_idx in range(n_bins):
        mask = bin_ids == bin_idx
        if not np.any(mask):
            continue
        bin_conf = float(np.mean(y_prob[mask]))
        bin_acc = float(np.mean(y_true[mask]))
        ece += (np.sum(mask) / len(y_true)) * abs(bin_acc - bin_conf)

    return {"brier_score": brier, "ece": float(ece)}


def calibration_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Build reliability-table style calibration bins."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1], right=True)
    rows = []

    for bin_idx in range(n_bins):
        mask = bin_ids == bin_idx
        if not np.any(mask):
            rows.append(
                {
                    "bin_index": bin_idx,
                    "bin_start": float(bins[bin_idx]),
                    "bin_end": float(bins[bin_idx + 1]),
                    "count": 0,
                    "mean_confidence": 0.0,
                    "empirical_positive_rate": 0.0,
                }
            )
            continue
        rows.append(
            {
                "bin_index": bin_idx,
                "bin_start": float(bins[bin_idx]),
                "bin_end": float(bins[bin_idx + 1]),
                "count": int(np.sum(mask)),
                "mean_confidence": float(np.mean(y_prob[mask])),
                "empirical_positive_rate": float(np.mean(y_true[mask])),
            }
        )

    return pd.DataFrame(rows)


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_threshold: float = 0.0,
    max_threshold: float = 1.0,
    num_thresholds: int = 201,
) -> tuple[float, dict[str, float]]:
    """Select a binary decision threshold using Youden's J statistic."""
    thresholds = np.linspace(min_threshold, max_threshold, num_thresholds)
    best_threshold = 0.5
    best_metrics: dict[str, float] | None = None
    best_score = float("-inf")

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        metrics = compute_metrics(y_true, y_pred, y_prob)
        score = metrics["sensitivity"] + metrics["specificity"] - 1.0
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics

    assert best_metrics is not None
    best_metrics["youden_j"] = best_score
    return best_threshold, best_metrics


def threshold_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute a threshold sweep table."""
    thresholds = thresholds if thresholds is not None else np.linspace(0.0, 1.0, 21)
    rows = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        metrics = compute_metrics(y_true, y_pred, y_prob)
        rows.append(
            {
                "threshold": float(threshold),
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "accuracy": metrics["accuracy"],
            }
        )
    return pd.DataFrame(rows)


def threshold_for_target_sensitivity(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_sensitivity: float,
    thresholds: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    """Find the highest-specificity threshold that still meets target sensitivity."""
    thresholds = thresholds if thresholds is not None else np.linspace(0.0, 1.0, 401)
    best_threshold = 0.0
    best_metrics: dict[str, float] | None = None
    best_specificity = float("-inf")

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        metrics = compute_metrics(y_true, y_pred, y_prob)
        if metrics["sensitivity"] + 1e-8 < target_sensitivity:
            continue
        if metrics["specificity"] > best_specificity:
            best_specificity = metrics["specificity"]
            best_threshold = float(threshold)
            best_metrics = metrics

    if best_metrics is None:
        best_threshold, best_metrics = find_best_threshold(y_true, y_prob)
    return best_threshold, best_metrics


def bootstrap_metric_ci(
    values_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_bootstrap: int = 500,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for a metric on probabilities."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    samples: list[float] = []

    if len(y_true) == 0:
        return 0.0, 0.0

    for _ in range(num_bootstrap):
        indices = rng.integers(0, len(y_true), len(y_true))
        sample_true = y_true[indices]
        sample_prob = y_prob[indices]
        if len(np.unique(sample_true)) < 2:
            continue
        samples.append(float(values_fn(sample_true, sample_prob)))

    if not samples:
        point = float(values_fn(y_true, y_prob))
        return point, point

    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def aggregate_group_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    group_ids: list[str] | np.ndarray,
    reducer: str = "max",
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate image-level probabilities into group-level predictions."""
    data = pd.DataFrame(
        {
            "group_id": list(group_ids),
            "label": np.asarray(y_true),
            "prob": np.asarray(y_prob),
        }
    )

    if reducer == "mean":
        grouped = data.groupby("group_id", as_index=False).agg({"label": "max", "prob": "mean"})
    else:
        grouped = data.groupby("group_id", as_index=False).agg({"label": "max", "prob": "max"})

    return grouped["label"].to_numpy(dtype=int), grouped["prob"].to_numpy(dtype=float)


def subgroup_metrics_table(
    metadata: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    decision_threshold: float,
    columns: list[str],
) -> pd.DataFrame:
    """Compute metrics for selected metadata columns and subgroup values."""
    rows = []
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= decision_threshold).astype(int)

    for column in columns:
        if column not in metadata.columns:
            continue
        values = metadata[column].fillna("unknown").astype(str)
        unique_values = sorted(v for v in values.unique() if v)
        if len(unique_values) <= 1:
            continue
        for value in unique_values:
            mask = values == value
            if not np.any(mask):
                continue
            metrics = compute_metrics(y_true[mask], y_pred[mask], y_prob[mask])
            rows.append(
                {
                    "column": column,
                    "value": value,
                    "count": int(np.sum(mask)),
                    "auc": metrics.get("auc", 0.0),
                    "pr_auc": metrics.get("pr_auc", 0.0),
                    "sensitivity": metrics["sensitivity"],
                    "specificity": metrics["specificity"],
                    "accuracy": metrics["accuracy"],
                }
            )

    return pd.DataFrame(rows)


def aggregate_metrics(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate metrics across multiple runs/folds."""
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


def precision_recall_points(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    """Return a precision-recall curve as a dataframe."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    threshold_values = np.concatenate([thresholds, np.array([1.0])])
    return pd.DataFrame(
        {
            "threshold": threshold_values,
            "precision": precision,
            "recall": recall,
        }
    )
