"""
Tests for classification metrics, including edge cases.
"""

import numpy as np

from src.training.metrics import (
    aggregate_group_predictions,
    aggregate_metrics,
    calibration_table,
    compute_calibration_metrics,
    compute_metrics,
    find_best_threshold,
    subgroup_metrics_table,
    threshold_for_target_sensitivity,
)


class TestComputeMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.8])

        metrics = compute_metrics(y_true, y_pred, y_prob)
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["auc"] == 1.0
        assert metrics["sensitivity"] == 1.0
        assert metrics["specificity"] == 1.0

    def test_all_wrong_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.1, 0.2])

        metrics = compute_metrics(y_true, y_pred, y_prob)
        assert metrics["accuracy"] == 0.0
        assert metrics["sensitivity"] == 0.0
        assert metrics["specificity"] == 0.0

    def test_single_class_in_true(self):
        """Edge case: only one class present (AUC should be 0.0)."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 1, 0])
        y_prob = np.array([0.1, 0.2, 0.6, 0.3])

        metrics = compute_metrics(y_true, y_pred, y_prob)
        assert metrics["auc"] == 0.0  # Cannot compute AUC with single class

    def test_single_class_predictions(self):
        """Edge case: model predicts only one class."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])  # All benign predictions
        y_prob = np.array([0.3, 0.2, 0.4, 0.3])

        metrics = compute_metrics(y_true, y_pred, y_prob)
        assert metrics["accuracy"] == 0.5
        assert metrics["sensitivity"] == 0.0
        assert metrics["specificity"] == 1.0

    def test_no_probabilities(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])

        metrics = compute_metrics(y_true, y_pred, y_prob=None)
        assert "accuracy" in metrics
        assert "auc" not in metrics  # No probabilities => no AUC

    def test_confusion_matrix_components(self):
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])

        metrics = compute_metrics(y_true, y_pred)
        assert metrics["tn"] == 1.0
        assert metrics["fp"] == 1.0
        assert metrics["fn"] == 1.0
        assert metrics["tp"] == 2.0


class TestAggregateMetrics:
    def test_aggregate(self):
        metrics_list = [
            {"accuracy": 0.8, "auc": 0.85},
            {"accuracy": 0.9, "auc": 0.90},
        ]
        agg = aggregate_metrics(metrics_list)
        assert abs(agg["accuracy_mean"] - 0.85) < 1e-6
        assert abs(agg["auc_mean"] - 0.875) < 1e-6
        assert agg["accuracy_std"] > 0.0

    def test_aggregate_empty(self):
        result = aggregate_metrics([])
        assert result == {}


class TestThresholdSelection:
    def test_find_best_threshold_returns_balanced_cutoff(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.4, 0.6, 0.9])

        threshold, metrics = find_best_threshold(y_true, y_prob)

        assert 0.4 <= threshold <= 0.6
        assert metrics["youden_j"] >= 0.0
        assert metrics["sensitivity"] == 1.0
        assert metrics["specificity"] == 1.0

    def test_threshold_for_target_sensitivity(self):
        y_true = np.array([0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.6, 0.8, 0.9])

        threshold, metrics = threshold_for_target_sensitivity(y_true, y_prob, 0.66)

        assert 0.0 <= threshold <= 1.0
        assert metrics["sensitivity"] >= 0.66


class TestClinicalMetrics:
    def test_calibration_metrics(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.3, 0.7, 0.9])

        metrics = compute_calibration_metrics(y_true, y_prob, n_bins=4)

        assert "brier_score" in metrics
        assert "ece" in metrics
        assert metrics["brier_score"] >= 0.0

    def test_calibration_table(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.3, 0.7, 0.9])

        table = calibration_table(y_true, y_prob, n_bins=4)

        assert len(table) == 4
        assert "mean_confidence" in table.columns

    def test_group_aggregation(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.2, 0.7, 0.4, 0.9])
        groups = ["A", "A", "B", "B"]

        agg_true, agg_prob = aggregate_group_predictions(y_true, y_prob, groups)

        assert agg_true.tolist() == [0, 1]
        assert agg_prob.tolist() == [0.7, 0.9]

    def test_subgroup_metrics_table(self):
        import pandas as pd

        metadata = pd.DataFrame(
            {
                "view": ["CC", "CC", "MLO", "MLO"],
                "laterality": ["LEFT", "RIGHT", "LEFT", "RIGHT"],
            }
        )
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.4, 0.6, 0.9])

        table = subgroup_metrics_table(metadata, y_true, y_prob, 0.5, ["view", "laterality"])

        assert not table.empty
        assert {"column", "value", "auc", "specificity"}.issubset(table.columns)
