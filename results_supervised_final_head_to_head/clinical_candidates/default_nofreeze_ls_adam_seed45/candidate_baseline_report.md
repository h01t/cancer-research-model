# Candidate Baseline Report

- Backbone: `efficientnet-b0`
- Image size: `512`
- Optimizer: `adam`
- Seed: `45`
- Decision threshold: `0.4500`

## Validation
- ROC AUC: `0.8221`
- PR AUC: `0.8032`
- Sensitivity: `0.7826`
- Specificity: `0.7174`
- Brier score: `0.1901`
- ECE: `0.1146`

## Test
- ROC AUC: `0.7024`
- PR AUC: `0.5859`
- Sensitivity: `0.7063`
- Specificity: `0.5963`
- Brier score: `0.2171`
- ECE: `0.0730`

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