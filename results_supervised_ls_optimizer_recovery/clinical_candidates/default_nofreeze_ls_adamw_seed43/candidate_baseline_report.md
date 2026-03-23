# Candidate Baseline Report

- Backbone: `efficientnet-b0`
- Image size: `512`
- Optimizer: `adamw`
- Seed: `43`
- Decision threshold: `0.4750`

## Validation
- ROC AUC: `0.8702`
- PR AUC: `0.8612`
- Sensitivity: `0.8090`
- Specificity: `0.8235`
- Brier score: `0.1535`
- ECE: `0.0952`

## Test
- ROC AUC: `0.7297`
- PR AUC: `0.6530`
- Sensitivity: `0.6364`
- Specificity: `0.6743`
- Brier score: `0.2072`
- ECE: `0.0815`

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