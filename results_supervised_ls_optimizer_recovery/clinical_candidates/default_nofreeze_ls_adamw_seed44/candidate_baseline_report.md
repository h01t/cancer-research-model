# Candidate Baseline Report

- Backbone: `efficientnet-b0`
- Image size: `512`
- Optimizer: `adamw`
- Seed: `44`
- Decision threshold: `0.5450`

## Validation
- ROC AUC: `0.8572`
- PR AUC: `0.8225`
- Sensitivity: `0.8791`
- Specificity: `0.7500`
- Brier score: `0.1727`
- ECE: `0.1348`

## Test
- ROC AUC: `0.7671`
- PR AUC: `0.6526`
- Sensitivity: `0.7343`
- Specificity: `0.6606`
- Brier score: `0.2484`
- ECE: `0.2021`

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