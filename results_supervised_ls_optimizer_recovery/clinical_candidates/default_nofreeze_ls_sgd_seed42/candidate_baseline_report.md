# Candidate Baseline Report

- Backbone: `efficientnet-b0`
- Image size: `512`
- Optimizer: `sgd`
- Seed: `42`
- Decision threshold: `0.3200`

## Validation
- ROC AUC: `0.8210`
- PR AUC: `0.7458`
- Sensitivity: `0.9326`
- Specificity: `0.6263`
- Brier score: `0.1754`
- ECE: `0.1097`

## Test
- ROC AUC: `0.7735`
- PR AUC: `0.6907`
- Sensitivity: `0.8531`
- Specificity: `0.5275`
- Brier score: `0.2102`
- ECE: `0.1148`

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