# Candidate Baseline Report

- Backbone: `efficientnet-b0`
- Image size: `512`
- Optimizer: `adam`
- Seed: `43`
- Decision threshold: `0.5100`

## Validation
- ROC AUC: `0.8780`
- PR AUC: `0.8731`
- Sensitivity: `0.7528`
- Specificity: `0.8706`
- Brier score: `0.1499`
- ECE: `0.1006`

## Test
- ROC AUC: `0.7200`
- PR AUC: `0.6462`
- Sensitivity: `0.5944`
- Specificity: `0.6835`
- Brier score: `0.2193`
- ECE: `0.1073`

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