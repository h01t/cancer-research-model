# Candidate Baseline Report

- Backbone: `efficientnet-b0`
- Image size: `512`
- Optimizer: `adam`
- Seed: `42`
- Decision threshold: `0.8100`

## Validation
- ROC AUC: `0.8347`
- PR AUC: `0.7929`
- Sensitivity: `0.8090`
- Specificity: `0.7172`
- Brier score: `0.2219`
- ECE: `0.2114`

## Test
- ROC AUC: `0.7587`
- PR AUC: `0.6601`
- Sensitivity: `0.6993`
- Specificity: `0.6835`
- Brier score: `0.2699`
- ECE: `0.2475`

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