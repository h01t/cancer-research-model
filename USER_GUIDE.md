# Mammography Classification User Guide

## 1. Overview

This project studies binary mammography classification on CBIS-DDSM. It began as a supervised-vs-semi-supervised comparison, but the corrected no-freeze supervised baseline is now the official best-performing setup and the main path forward.

### Current Research Position

- Official baseline: `configs/default_nofreeze.yaml`
- Historical comparison paths: FixMatch and Mean Teacher
- Final compact summaries: `reports/ssl_closure_summary.txt`, `reports/ssl_closure_summary.csv`, `reports/final_results.md`

### Final Grouped Mean Validation AUC

| Method | 100 | 250 | 500 |
|--------|-----|-----|-----|
| Supervised (frozen historical baseline) | 0.6033 | 0.7097 | 0.8084 |
| Supervised no-freeze (official baseline) | 0.7442 | 0.8370 | 0.8575 |
| FixMatch | 0.6559 | 0.6788 | - |
| Mean Teacher | 0.6325 | 0.6595 | 0.6861 |

Conclusion: close the vanilla SSL chapter for now and push the project forward through stronger supervised experiments.

## 2. Installation

```bash
git clone <repository-url>
cd ssl
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

Run tests:

```bash
python -m pytest tests/ -q
```

## 3. Dataset

Download [CBIS-DDSM from Kaggle](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset) and extract into:

```text
data/
├── csv/
└── jpeg/
```

The repo uses patient-aware train/validation splitting to avoid leakage across mammograms from the same patient.

## 4. Recommended Workflow

### 4.1 Train The Official Baseline

```bash
python scripts/train_supervised.py \
  --config configs/default_nofreeze.yaml \
  --labeled_subset 100 \
  --output_dir results/supervised_nofreeze_100
```

### 4.2 Run The Official Multi-Seed Sweep

```bash
./run_nofreeze_baseline_sweep.sh
```

### 4.3 Inspect Training

Enable TensorBoard in config:

```yaml
tensorboard:
  enabled: true
  log_dir: null
  flush_secs: 30
```

Launch:

```bash
tensorboard --logdir results
```

### 4.4 Analyze In Notebooks

Use:
- `notebooks/01_data_exploration.ipynb`
- `notebooks/02_training_curves.ipynb`
- `notebooks/03_embeddings.ipynb`
- `notebooks/04_model_interpretation.ipynb`

## 5. Config Guide

| File | Status | Purpose |
|------|--------|---------|
| `configs/default_nofreeze.yaml` | Active | Official supervised baseline |
| `configs/default.yaml` | Historical | Frozen supervised comparison baseline |
| `configs/test.yaml` | Active | Quick smoke test |
| `configs/fixmatch.yaml` | Historical | Archived FixMatch comparison |
| `configs/mean_teacher.yaml` | Historical | Archived Mean Teacher comparison |
| `configs/fixmatch_static.yaml` | Historical | Rescue attribution probe |
| `configs/fixmatch_legacy_aug.yaml` | Historical | Rescue augmentation probe |

## 6. Historical SSL Workflows

These remain in the repo for reproducibility, but they are not the recommended path.

### 6.1 FixMatch

```bash
python scripts/train_fixmatch.py \
  --config configs/fixmatch.yaml \
  --labeled 100 \
  --output_dir results/fixmatch_100
```

### 6.2 Mean Teacher

```bash
python scripts/train_mean_teacher.py \
  --config configs/mean_teacher.yaml \
  --labeled 100 \
  --output_dir results/mean_teacher_100
```

### 6.3 Historical Summary Scripts

- `run_fixmatch_rescue.sh`
- `run_followup_overnight.sh`
- `scripts/summarize_rescue.py`
- `scripts/summarize_followup.py`

## 7. Result Summaries

The repo now keeps compact versioned summaries instead of large tracked result trees.

- `reports/final_results.md`: short human-readable conclusion
- `reports/ssl_closure_summary.txt`: full compact text summary
- `reports/ssl_closure_summary.csv`: tabular export

## 8. Next Research Directions

Recommended next steps:

- supervised augmentation and regularization tuning
- pretraining or self-supervised initialization
- higher-resolution or lesion-aware cropping experiments
- calibration and thresholding refinement
