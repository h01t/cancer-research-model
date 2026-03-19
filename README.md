# SSL Mammography: Semi-Supervised Learning for Breast Cancer Detection

Semi-supervised learning (FixMatch) on the CBIS-DDSM mammography dataset to improve classification performance with limited labeled data.

## Project Overview

- **Dataset**: CBIS-DDSM -- mammography images with benign/malignant labels
- **Task**: Binary classification (benign vs malignant) using full mammograms at 512px
- **SSL Method**: FixMatch with confidence-thresholded pseudo-labeling + EMA teacher
- **Model**: EfficientNet-B0 (torchvision, ImageNet pretrained)
- **Evaluation**: AUC-ROC primary metric, with accuracy, F1, sensitivity, specificity
- **Platform**: Develops on Apple Silicon (MPS), trains on CUDA workstation

## Key Features

- FixMatch semi-supervised learning with EMA teacher model
- Conservative paper-aligned SSL config (tau=0.95, lambda_u=1.0)
- Mixed precision training (AMP) for CUDA and MPS
- LR warmup, gradient clipping, class-weighted loss, backbone freeze/unfreeze
- Config-driven augmentation and training pipeline
- Ablation study across labeled data sizes (100, 250, 500, full)
- Patient-aware train/val splits (no data leakage between patients)
- Complete SSL state checkpoints (EMA, ramp schedules, early stopping)
- Per-experiment result isolation with training history CSV
- W&B integration for experiment tracking (optional)
- 52 unit tests with synthetic data (no real dataset required)

## Project Structure

```
.
├── src/                      # Source code
│   ├── data/
│   │   ├── dataset.py        # CBIS-DDSM dataset loader + patient-aware splitting
│   │   ├── ssl_dataset.py    # SSL dataset wrappers (FixMatch)
│   │   └── transforms.py     # Augmentation pipeline (weak, strong, test)
│   ├── models/
│   │   └── efficientnet.py   # EfficientNet-B0 classifier
│   └── training/
│       ├── trainer.py        # Base trainer (AMP, warmup, grad clip, checkpoints)
│       ├── fixmatch_trainer.py  # FixMatch SSL trainer
│       ├── ema.py            # Exponential Moving Average model
│       └── metrics.py        # Classification metrics
├── scripts/                  # Training scripts
│   ├── train_supervised.py   # Supervised baseline
│   └── train_fixmatch.py     # FixMatch SSL training
├── configs/                  # YAML configuration files
│   ├── default.yaml          # Supervised baseline config (512px, batch 8)
│   ├── fixmatch.yaml         # FixMatch config (paper-aligned, matches supervised base)
│   └── test.yaml             # Quick validation config (224px, 2 epochs)
├── tests/                    # Unit tests (52 tests, synthetic data)
├── tasks/                    # Project TODO and lessons learned
├── run_ablation.sh           # Full ablation study script (7 experiments)
├── pyproject.toml            # Python packaging and tool config
├── requirements.txt          # Dependencies
└── USER_GUIDE.md             # Comprehensive usage guide
```

## Quick Start

### 1. Environment Setup

```bash
git clone <repository-url>
cd ssl
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Development tools (tests, linting)
pip install -e ".[dev]"
```

### 2. Verify Installation

```bash
# Run tests (no dataset needed)
python -m pytest tests/ -v
# Expected: 52 passed
```

### 3. Download Dataset

Download [CBIS-DDSM from Kaggle](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset) and extract into `data/`:

```
data/
├── csv/               # CSV annotation files
└── jpeg/              # JPEG images
```

### 4. Train

```bash
# Supervised baseline (100 labeled samples)
python scripts/train_supervised.py \
    --config configs/default.yaml \
    --labeled_subset 100 \
    --output_dir results/supervised_100

# FixMatch SSL (100 labeled samples)
python scripts/train_fixmatch.py \
    --config configs/fixmatch.yaml \
    --labeled 100 \
    --output_dir results/fixmatch_100

# Full ablation study (7 experiments: supervised + FixMatch at 100/250/500/full)
./run_ablation.sh
```

### 5. Device Selection

```bash
# Auto-detect (CUDA > MPS > CPU)
python scripts/train_supervised.py --config configs/default.yaml

# Explicit device
python scripts/train_supervised.py --config configs/default.yaml --device cuda
python scripts/train_supervised.py --config configs/default.yaml --device mps
```

## Configuration

Two configs are provided. They share identical base hyperparameters so ablation results isolate the SSL effect cleanly.

**`configs/default.yaml`** -- supervised baseline:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `dataset.image_size` | 512 | Input resolution |
| `training.batch_size` | 8 | Fits 8GB VRAM at 512px |
| `training.learning_rate` | 0.001 | Adam optimizer |
| `training.num_epochs` | 100 | Max epochs |
| `training.warmup_epochs` | 5 | Linear LR warmup |
| `training.early_stopping_patience` | 20 | Patience before stopping |
| `model.freeze_backbone` | true | Freeze backbone for first 5 epochs |
| `training.class_weighted_loss` | true | Inverse-frequency class weights |

**`configs/fixmatch.yaml`** -- FixMatch SSL (differences from supervised only):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `training.num_epochs` | 150 | More epochs for pseudo-label stabilization |
| `training.early_stopping_patience` | 25 | More patience for SSL convergence |
| `ssl.confidence_threshold` | 0.95 | Paper default -- only high-confidence pseudo-labels |
| `ssl.lambda_u` | 1.0 | Paper default -- full unsupervised loss weight |
| `ssl.use_ema` | true | EMA teacher for stable pseudo-labels |
| `ssl.unlabeled_batch_ratio` | 2 | Unlabeled/labeled batch ratio (fits 8GB VRAM) |

## SSH Workflow (Local Dev -> Remote Training)

```bash
# Sync code to workstation
rsync -avz --exclude='.venv' --exclude='data' --exclude='results' \
    ./ user@workstation:~/ssl/

# Run overnight ablation
ssh user@workstation
cd ~/ssl && source .venv/bin/activate
nohup ./run_ablation.sh 2>&1 &

# Pull results back
rsync -avz user@workstation:~/ssl/results/ ./results/
```

## Troubleshooting

- **Out of memory**: Reduce `training.batch_size` or `dataset.image_size` in config
- **SSL certificate errors**: Set `model.pretrained: false` or download weights manually
- **MPS issues**: Set `training.use_amp: false` if AMP causes issues on Apple Silicon
- **Slow data loading**: Increase `training.num_workers` (up to CPU core count)

## License

MIT License -- see LICENSE file.

For detailed instructions, see [USER_GUIDE.md](USER_GUIDE.md).
