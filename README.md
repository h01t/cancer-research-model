# Mammography Classification Research

> **Note**: This repository contains code for an academic master's research project and is provided for educational and review purposes.

Medical imaging research on CBIS-DDSM mammography classification. The project started as a supervised-vs-SSL comparison and has now converged on a supervised-first path: the current promoted baseline is `label_smoothing + adamw` on an EfficientNet-B0 512px pipeline, selected after baseline correction, supervised sweeps, and clinical-style follow-up evaluation.

## Current Status

- Active path: `default_nofreeze_ls_adamw` using [`configs/default_nofreeze_ls_adamw.yaml`](/Users/grmim/Dev/ssl/configs/default_nofreeze_ls_adamw.yaml)
- Current promoted task: 500-label supervised mammography classification with clinical-style evaluation and calibration follow-up
- Historical research paths: frozen supervised, FixMatch, and Mean Teacher remain in the repo for reference, but they are no longer the recommended training workflow
- Chapter-close and progress summaries live in [`reports/ssl_closure_summary.txt`](/Users/grmim/Dev/ssl/reports/ssl_closure_summary.txt), [`reports/final_results.md`](/Users/grmim/Dev/ssl/reports/final_results.md), and [`reports/current_progress_brief.md`](/Users/grmim/Dev/ssl/reports/current_progress_brief.md)

## Results

### Phase 1: Closing The Vanilla SSL Chapter

Grouped mean validation AUC on the initial comparison:

| Method | 100 labels | 250 labels | 500 labels |
|--------|------------|------------|------------|
| Supervised (frozen historical baseline) | 0.6033 | 0.7097 | 0.8084 |
| Supervised no-freeze (official baseline) | 0.7442 | 0.8370 | 0.8575 |
| FixMatch | 0.6559 | 0.6788 | - |
| Mean Teacher | 0.6325 | 0.6595 | 0.6861 |

The key lesson was that the original supervised baseline was artificially weak because of freeze behavior. Once corrected, supervised fine-tuning clearly outperformed the vanilla SSL methods explored in this repo.

### Phase 2: Promoted Supervised Baseline

After supervised-only sweeps over resolution, backbone, regularization, and optimizer, the current promoted baseline is:

- `EfficientNet-B0`
- `512x512`
- `label_smoothing=0.1`
- `AdamW`

Combined 6-seed clinical-style summary (`42-47`) for `default_nofreeze_ls_adamw`:

| Metric | Value |
|--------|-------|
| Mean validation ROC AUC | `0.8633` |
| Mean test ROC AUC | `0.7589` |
| Mean test PR AUC | `0.6748` |
| Mean test sensitivity | `0.7110` |
| Mean test specificity | `0.6651` |
| Mean specificity at 0.90 target sensitivity | `0.5833` |
| Mean exam-level ROC AUC | `0.7668` |

`default_nofreeze_ls_adam` remained slightly better calibrated, but `adamw` won on validation AUC, test AUC, PR AUC, sensitivity, and fixed-sensitivity specificity. That made it the best overall decision-support candidate tested so far.

## Current Recommended Workflow

1. Train with `scripts/train_supervised.py` and [`configs/default_nofreeze_ls_adamw.yaml`](/Users/grmim/Dev/ssl/configs/default_nofreeze_ls_adamw.yaml)
2. Evaluate candidate runs with `scripts/clinical_eval.py` or `run_clinical_candidate_eval.sh`
3. Inspect training with TensorBoard and the notebooks in [`notebooks/`](/Users/grmim/Dev/ssl/notebooks)
4. Use historical SSL scripts only as archived research references

## Project Structure

```text
.
├── src/
│   ├── data/                 # Dataset loading, splits, and transforms
│   ├── experiments/          # Shared runtime and dataset builders
│   ├── models/               # EfficientNet classifier
│   └── training/             # Supervised + historical SSL trainers
├── scripts/                  # Thin training and summary entrypoints
├── configs/                  # Primary + historical experiment configs
├── notebooks/                # Research notebooks for data/model analysis
├── reports/                  # Compact versioned research summaries
├── tests/                    # Synthetic-data test suite
└── tasks/                    # Lessons learned and next steps
```

## Quick Start

### 1. Environment Setup

```bash
git clone <repository-url>
cd ssl
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

### 2. Verify Installation

```bash
python -m pytest tests/ -q
```

### 3. Prepare Dataset

Download [CBIS-DDSM from Kaggle](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset) and extract into:

```text
data/
├── csv/
└── jpeg/
```

### 4. Train The Official Baseline

```bash
python scripts/train_supervised.py \
  --config configs/default_nofreeze_ls_adamw.yaml \
  --labeled_subset 500 \
  --output_dir results/default_nofreeze_ls_adamw_500_seed42 \
  --seed 42
```

### 5. Run Clinical-Style Evaluation On A Trained Run

```bash
python scripts/clinical_eval.py \
  --run_dir results/default_nofreeze_ls_adamw_500_seed42 \
  --output_dir results/default_nofreeze_ls_adamw_500_seed42/clinical_eval
```

### 6. Analyze

```bash
tensorboard --logdir results
```

Use the notebooks in [`notebooks/`](/Users/grmim/Dev/ssl/notebooks) for:
- data exploration
- training curves
- embedding inspection
- Grad-CAM / interpretation

## Configs

- `configs/default_nofreeze_ls_adamw.yaml`: current official supervised baseline
- `configs/default_nofreeze_ls_adam.yaml`: calibration-oriented comparison baseline
- `configs/default_nofreeze.yaml`: earlier no-freeze supervised baseline
- `configs/default.yaml`: historical frozen supervised baseline for comparison
- `configs/fixmatch.yaml`: archived FixMatch research config
- `configs/mean_teacher.yaml`: archived Mean Teacher research config
- `configs/test.yaml`: quick smoke-test config

## Current Research Direction

The SSL codepaths are intentionally preserved for reproducibility and comparison, but they are not the main training recommendation anymore. The active roadmap is:

- calibration for the promoted `adamw` baseline
- false-positive / false-negative review and subgroup analysis
- exam-level and multi-view modeling
- external validation on additional mammography datasets
- traceable, clinical-style evidence packaging for future decision-support work

## TensorBoard

TensorBoard is opt-in via config:

```yaml
tensorboard:
  enabled: true
  log_dir: null
  flush_secs: 30
```

Then launch:

```bash
tensorboard --logdir results
```

TensorBoard complements the YAML and CSV outputs; it does not replace them.
