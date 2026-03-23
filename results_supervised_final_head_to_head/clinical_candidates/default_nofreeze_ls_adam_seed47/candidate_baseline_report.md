# Candidate Baseline Report

- Backbone: `efficientnet-b0`
- Image size: `512`
- Optimizer: `adam`
- Seed: `47`
- Decision threshold: `0.5050`

## Validation
- ROC AUC: `0.8488`
- PR AUC: `0.8128`
- Sensitivity: `0.7590`
- Specificity: `0.8085`
- Brier score: `0.1637`
- ECE: `0.0778`

## Test
- ROC AUC: `0.7504`
- PR AUC: `0.6605`
- Sensitivity: `0.6503`
- Specificity: `0.6972`
- Brier score: `0.2002`
- ECE: `0.0694`

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