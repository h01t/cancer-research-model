# Candidate Baseline Report

- Backbone: `efficientnet-b0`
- Image size: `512`
- Optimizer: `adam`
- Seed: `42`
- Decision threshold: `0.3550`

## Validation
- ROC AUC: `0.8988`
- PR AUC: `0.8970`
- Sensitivity: `0.8876`
- Specificity: `0.7778`
- Brier score: `0.1290`
- ECE: `0.0487`

## Test
- ROC AUC: `0.7744`
- PR AUC: `0.6870`
- Sensitivity: `0.6993`
- Specificity: `0.7018`
- Brier score: `0.2001`
- ECE: `0.1158`

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