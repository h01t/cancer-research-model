# Candidate Baseline Report

- Backbone: `efficientnet-b0`
- Image size: `512`
- Optimizer: `adam`
- Seed: `42`
- Decision threshold: `0.4000`

## Validation
- ROC AUC: `0.8744`
- PR AUC: `0.8508`
- Sensitivity: `0.9101`
- Specificity: `0.7374`
- Brier score: `0.1590`
- ECE: `0.1296`

## Test
- ROC AUC: `0.7950`
- PR AUC: `0.7316`
- Sensitivity: `0.7972`
- Specificity: `0.6560`
- Brier score: `0.2433`
- ECE: `0.2177`

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