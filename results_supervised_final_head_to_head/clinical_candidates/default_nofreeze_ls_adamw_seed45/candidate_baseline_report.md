# Candidate Baseline Report

- Backbone: `efficientnet-b0`
- Image size: `512`
- Optimizer: `adamw`
- Seed: `45`
- Decision threshold: `0.2100`

## Validation
- ROC AUC: `0.8502`
- PR AUC: `0.8264`
- Sensitivity: `0.8261`
- Specificity: `0.7826`
- Brier score: `0.1690`
- ECE: `0.1276`

## Test
- ROC AUC: `0.7901`
- PR AUC: `0.7390`
- Sensitivity: `0.8042`
- Specificity: `0.6239`
- Brier score: `0.2159`
- ECE: `0.1787`

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