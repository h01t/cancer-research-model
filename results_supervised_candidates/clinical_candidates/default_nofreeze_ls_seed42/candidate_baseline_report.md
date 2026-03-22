# Candidate Baseline Report

- Backbone: `efficientnet-b0`
- Image size: `512`
- Optimizer: `adam`
- Seed: `42`
- Decision threshold: `0.3600`

## Validation
- ROC AUC: `0.8992`
- PR AUC: `0.8974`
- Sensitivity: `0.8876`
- Specificity: `0.7778`
- Brier score: `0.1289`
- ECE: `0.0408`

## Test
- ROC AUC: `0.7740`
- PR AUC: `0.6854`
- Sensitivity: `0.6923`
- Specificity: `0.7018`
- Brier score: `0.2003`
- ECE: `0.1193`

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