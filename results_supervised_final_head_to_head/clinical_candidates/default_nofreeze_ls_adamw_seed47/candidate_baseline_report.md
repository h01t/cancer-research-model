# Candidate Baseline Report

- Backbone: `efficientnet-b0`
- Image size: `512`
- Optimizer: `adamw`
- Seed: `47`
- Decision threshold: `0.1000`

## Validation
- ROC AUC: `0.8473`
- PR AUC: `0.8299`
- Sensitivity: `0.8916`
- Specificity: `0.6596`
- Brier score: `0.1955`
- ECE: `0.1651`

## Test
- ROC AUC: `0.7386`
- PR AUC: `0.6618`
- Sensitivity: `0.7343`
- Specificity: `0.6101`
- Brier score: `0.2268`
- ECE: `0.1766`

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