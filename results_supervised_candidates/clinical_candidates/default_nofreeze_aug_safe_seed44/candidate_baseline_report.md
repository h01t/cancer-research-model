# Candidate Baseline Report

- Backbone: `efficientnet-b0`
- Image size: `512`
- Optimizer: `adam`
- Seed: `44`
- Decision threshold: `0.2450`

## Validation
- ROC AUC: `0.8460`
- PR AUC: `0.8520`
- Sensitivity: `0.7692`
- Specificity: `0.8333`
- Brier score: `0.1864`
- ECE: `0.1553`

## Test
- ROC AUC: `0.7874`
- PR AUC: `0.7443`
- Sensitivity: `0.6713`
- Specificity: `0.7385`
- Brier score: `0.2117`
- ECE: `0.1796`

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