# Candidate Baseline Report

- Backbone: `efficientnet-b0`
- Image size: `512`
- Optimizer: `adam`
- Seed: `43`
- Decision threshold: `0.4400`

## Validation
- ROC AUC: `0.8365`
- PR AUC: `0.8154`
- Sensitivity: `0.7640`
- Specificity: `0.8118`
- Brier score: `0.1803`
- ECE: `0.1428`

## Test
- ROC AUC: `0.7197`
- PR AUC: `0.6254`
- Sensitivity: `0.5524`
- Specificity: `0.7156`
- Brier score: `0.2502`
- ECE: `0.1946`

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