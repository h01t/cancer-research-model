# Candidate Baseline Report

- Backbone: `efficientnet-b0`
- Image size: `512`
- Optimizer: `sgd`
- Seed: `44`
- Decision threshold: `0.4750`

## Validation
- ROC AUC: `0.8503`
- PR AUC: `0.8313`
- Sensitivity: `0.8571`
- Specificity: `0.7188`
- Brier score: `0.1638`
- ECE: `0.0907`

## Test
- ROC AUC: `0.7799`
- PR AUC: `0.7036`
- Sensitivity: `0.7692`
- Specificity: `0.6835`
- Brier score: `0.2138`
- ECE: `0.1723`

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