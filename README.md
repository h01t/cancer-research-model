# Mammography Classification Research

Medical imaging research on CBIS-DDSM mammography classification. The project started as a supervised-vs-SSL comparison and currently moves forward with a corrected no-freeze supervised baseline, which outperformed the vanilla FixMatch and Mean Teacher setups tested here.

## Current Status

- Active path: `supervised_nofreeze` using `configs/default_nofreeze.yaml`
- Historical research paths: FixMatch and Mean Teacher remain in the repo for reference, but they are not the recommended training workflow
- Final chapter-close summaries live in `reports/ssl_closure_summary.txt`, `reports/ssl_closure_summary.csv`, and `reports/final_results.md`

## Results

Grouped mean validation AUC:

| Method | 100 labels | 250 labels | 500 labels |
|--------|------------|------------|------------|
| Supervised (frozen historical baseline) | 0.6033 | 0.7097 | 0.8084 |
| Supervised no-freeze (official baseline) | 0.7442 | 0.8370 | 0.8575 |
| FixMatch | 0.6559 | 0.6788 | - |
| Mean Teacher | 0.6325 | 0.6595 | 0.6861 |

The key lesson is that the original supervised baseline was artificially weak because of the freeze behavior. Once corrected, supervised fine-tuning clearly outperformed the vanilla SSL methods explored in this repo. The next phase is stronger supervised experimentation, not more FixMatch/Mean Teacher tuning.

## Current Recommended Workflow

1. Train with `scripts/train_supervised.py` and `configs/default_nofreeze.yaml`
2. Use `run_nofreeze_baseline_sweep.sh` for multi-seed baseline runs
3. Inspect training with TensorBoard and the notebooks in `notebooks/`
4. Treat FixMatch/Mean Teacher scripts as archived research references

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
  --config configs/default_nofreeze.yaml \
  --labeled_subset 100 \
  --output_dir results/supervised_nofreeze_100
```

### 5. Run The Official Multi-Seed Baseline Sweep

```bash
./run_nofreeze_baseline_sweep.sh
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

- `configs/default_nofreeze.yaml`: official supervised baseline
- `configs/default.yaml`: historical frozen supervised baseline for comparison
- `configs/fixmatch.yaml`: archived FixMatch research config
- `configs/mean_teacher.yaml`: archived Mean Teacher research config
- `configs/test.yaml`: quick smoke-test config

## Historical SSL Notes

The SSL codepaths are intentionally preserved for reproducibility and comparison, but they are not the main training recommendation anymore. The repo’s current research direction is:

- stronger supervised tuning
- pretraining / representation learning
- resolution and cropping experiments
- calibration and thresholding refinement

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
