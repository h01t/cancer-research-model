# Candidate Baseline Report

- Backbone: `efficientnet-b0`
- Image size: `512`
- Optimizer: `adamw`
- Seed: `46`
- Decision threshold: `0.2150`

## Validation
- ROC AUC: `0.8728`
- PR AUC: `0.8530`
- Sensitivity: `0.9032`
- Specificity: `0.7368`
- Brier score: `0.1571`
- ECE: `0.0897`

## Test
- ROC AUC: `0.7518`
- PR AUC: `0.6657`
- Sensitivity: `0.7133`
- Specificity: `0.6284`
- Brier score: `0.2056`
- ECE: `0.1163`

## Artifacts
- `clinical_summary.yaml`
- `clinical_summary.csv`
- `grouped_metrics.csv`
- `fixed_sensitivity_metrics.csv`
- `threshold_table.csv`
- `calibration_bins.csv`
- `subgroup_metrics.csv`
- `top_false_positives.csv`
- `top_false_negatives.csv`