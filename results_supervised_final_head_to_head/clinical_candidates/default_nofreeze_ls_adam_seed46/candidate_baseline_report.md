# Candidate Baseline Report

- Backbone: `efficientnet-b0`
- Image size: `512`
- Optimizer: `adam`
- Seed: `46`
- Decision threshold: `0.3350`

## Validation
- ROC AUC: `0.7411`
- PR AUC: `0.7644`
- Sensitivity: `0.4946`
- Specificity: `0.9053`
- Brier score: `0.2564`
- ECE: `0.2274`

## Test
- ROC AUC: `0.7040`
- PR AUC: `0.6040`
- Sensitivity: `0.4476`
- Specificity: `0.8257`
- Brier score: `0.2335`
- ECE: `0.1617`

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