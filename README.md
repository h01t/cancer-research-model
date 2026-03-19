# SSL Mammography: Semi-Supervised Learning for Breast Cancer Detection

Semi-supervised learning (FixMatch) on the CBIS-DDSM mammography dataset to improve classification performance with limited labeled data.

## Project Overview

- **Dataset**: CBIS-DDSM — mammography images with benign/malignant labels
- **Task**: Binary classification (benign vs malignant) using full mammograms at configurable resolution (default 512px)
- **SSL Method**: FixMatch with confidence-thresholded pseudo-labeling + EMA + distribution alignment
- **Model**: EfficientNet-B0 (torchvision, ImageNet pretrained)
- **Evaluation**: AUC-ROC primary metric, with accuracy, F1, sensitivity, specificity
- **Platform**: Develops on Apple Silicon (MPS), deploys on CUDA workstation via SSH

## Key Features

- FixMatch semi-supervised learning with EMA teacher model and distribution alignment
- Mixed precision training (AMP) for CUDA and MPS
- LR warmup, gradient clipping, class-weighted loss
- Config-driven augmentation pipeline
- Ablation study across labeled data sizes (100, 250, 500 samples)
- Per-experiment checkpoints and result isolation
- W&B integration for experiment tracking (optional)
- 52 unit tests with synthetic data (no real dataset required)
- Patient-aware train/val/test splits (prevent data leakage)
- Complete SSL state checkpoints (EMA, distribution alignment, ramp schedules)
- Dual-device support: M4 Pro (dev) + CUDA workstation (training)

## Project Structure

```
.
├── src/                      # Source code
│   ├── data/
│   │   ├── dataset.py        # CBIS-DDSM dataset loader
│   │   ├── ssl_dataset.py    # SSL dataset wrappers (FixMatch)
│   │   └── transforms.py     # Augmentation pipeline
│   ├── models/
│   │   └── efficientnet.py   # EfficientNet-B0 classifier
│   └── training/
│       ├── trainer.py        # Base trainer (AMP, warmup, grad clip)
│       ├── fixmatch_trainer.py  # FixMatch SSL trainer
│       ├── ema.py            # Exponential Moving Average
│       └── metrics.py        # Classification metrics
├── scripts/                  # Training and evaluation scripts
│   ├── train_supervised.py   # Supervised baseline
│   ├── train_fixmatch.py     # FixMatch SSL training
│   └── debug/                # Dataset exploration scripts
├── configs/                  # YAML configuration files
│   ├── default.yaml          # Full experiment config
│   └── test.yaml             # Quick validation config
├── tests/                    # Unit tests (52 tests, synthetic data)
├── tasks/                    # Project TODO and lessons learned
├── notebooks/                # Jupyter notebooks for analysis
├── pyproject.toml            # Python packaging and tool config
├── requirements.txt          # Dependencies
└── USER_GUIDE.md             # Comprehensive usage guide
```

## Quick Start

### 1. Environment Setup

```bash
# Clone and create environment
git clone <repository-url>
cd ssl
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For W&B experiment tracking (optional)
pip install wandb

# For development (tests, linting)
pip install -e ".[dev]"
```

### 2. Verify Installation

```bash
# Run tests (no dataset needed)
python -m pytest tests/ -v

# Or use the quick start script
chmod +x quick_start.sh
./quick_start.sh
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

# Full ablation study
./run_ablation.sh
```

### 5. Device Selection

```bash
# Auto-detect (CUDA > MPS > CPU)
python scripts/train_supervised.py --config configs/default.yaml

# Explicit device
python scripts/train_supervised.py --config configs/default.yaml --device mps
python scripts/train_supervised.py --config configs/default.yaml --device cuda
```

## Configuration

Supervised baseline parameters are in `configs/default.yaml`. SSL (FixMatch) parameters are in `configs/fixmatch.yaml`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `training.use_amp` | `true` | Mixed precision training |
| `training.warmup_epochs` | `5` | Linear LR warmup |
| `training.class_weighted_loss` | `true` | Inverse-frequency class weights |
| `ssl.confidence_threshold` | `0.95` | Pseudo-label confidence threshold |
| `ssl.use_ema` | `true` | EMA model for stable pseudo-labels |
| `ssl.distribution_alignment` | `true` | ReMixMatch-style class distribution alignment |
| `ssl.threshold_ramp` | `[0.7, 0.9, 40]` | Confidence threshold ramp [start, end, epochs] |
| `ssl.lambda_u_ramp` | `[0.1, 0.5, 20]` | Unsupervised loss weight ramp [start, end, epochs] |
| `ssl.backbone_unfreeze_epoch` | `5` | Epoch to unfreeze pretrained backbone |
| `wandb.enabled` | `false` | W&B experiment tracking |

## SSH Workflow (Local Dev → Remote Training)

```bash
# Sync code to workstation
rsync -avz --exclude='.venv' --exclude='data' --exclude='results' \
    ./ user@workstation:~/ssl/

# Run training remotely
ssh user@workstation "cd ~/ssl && python scripts/train_fixmatch.py --config configs/fixmatch.yaml --labeled 100"

# Pull results back
rsync -avz user@workstation:~/ssl/results/ ./results/
```

## Troubleshooting

- **Memory issues**: Reduce `training.batch_size` or `dataset.image_size` in config
- **SSL certificate errors**: Set `model.pretrained: false` or download weights manually
- **MPS issues**: Set `training.use_amp: false` if AMP causes issues on Apple Silicon
- **Patient-aware splits**: Ensure train/val/test splits respect patient IDs to prevent data leakage

## License

MIT License — see LICENSE file.

For detailed instructions, see [USER_GUIDE.md](USER_GUIDE.md).
