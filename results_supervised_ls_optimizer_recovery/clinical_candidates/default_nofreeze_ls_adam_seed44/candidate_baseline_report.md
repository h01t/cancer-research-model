# Candidate Baseline Report

- Backbone: `efficientnet-b0`
- Image size: `512`
- Optimizer: `adam`
- Seed: `44`
- Decision threshold: `0.4500`

## Validation
- ROC AUC: `0.8623`
- PR AUC: `0.8554`
- Sensitivity: `0.7802`
- Specificity: `0.7917`
- Brier score: `0.1519`
- ECE: `0.0565`

## Test
- ROC AUC: `0.7442`
- PR AUC: `0.6789`
- Sensitivity: `0.6713`
- Specificity: `0.6697`
- Brier score: `0.2172`
- ECE: `0.1232`

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