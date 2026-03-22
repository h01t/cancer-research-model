# Candidate Baseline Report

- Backbone: `efficientnet-b0`
- Image size: `512`
- Optimizer: `adam`
- Seed: `43`
- Decision threshold: `0.4000`

## Validation
- ROC AUC: `0.8451`
- PR AUC: `0.8214`
- Sensitivity: `0.8090`
- Specificity: `0.7882`
- Brier score: `0.1687`
- ECE: `0.0943`

## Test
- ROC AUC: `0.7173`
- PR AUC: `0.6659`
- Sensitivity: `0.6643`
- Specificity: `0.6009`
- Brier score: `0.2211`
- ECE: `0.1228`

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