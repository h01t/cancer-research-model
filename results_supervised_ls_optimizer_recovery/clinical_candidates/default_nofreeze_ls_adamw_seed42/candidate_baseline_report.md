# Candidate Baseline Report

- Backbone: `efficientnet-b0`
- Image size: `512`
- Optimizer: `adamw`
- Seed: `42`
- Decision threshold: `0.6850`

## Validation
- ROC AUC: `0.8818`
- PR AUC: `0.8810`
- Sensitivity: `0.8202`
- Specificity: `0.8283`
- Brier score: `0.1527`
- ECE: `0.1043`

## Test
- ROC AUC: `0.7760`
- PR AUC: `0.6764`
- Sensitivity: `0.6434`
- Specificity: `0.7936`
- Brier score: `0.2088`
- ECE: `0.1528`

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