# Candidate Baseline Report

- Backbone: `efficientnet-b0`
- Image size: `512`
- Optimizer: `adam`
- Seed: `44`
- Decision threshold: `0.3850`

## Validation
- ROC AUC: `0.8539`
- PR AUC: `0.8498`
- Sensitivity: `0.8022`
- Specificity: `0.7708`
- Brier score: `0.1611`
- ECE: `0.0806`

## Test
- ROC AUC: `0.7431`
- PR AUC: `0.6950`
- Sensitivity: `0.6783`
- Specificity: `0.6651`
- Brier score: `0.1997`
- ECE: `0.0645`

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