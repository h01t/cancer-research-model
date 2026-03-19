# SSL Mammography: Comprehensive User Guide

## 1. Overview

This guide covers reproducing the semi-supervised learning (SSL) experiments on the CBIS-DDSM mammography dataset using the FixMatch algorithm. The project demonstrates how SSL can improve classification performance when labeled medical imaging data is scarce.

### Research Context
- **Problem**: Limited labeled medical imaging data due to annotation costs and privacy concerns
- **Solution**: FixMatch algorithm combining consistency regularization and pseudo-labeling with EMA
- **Dataset**: CBIS-DDSM (Curated Breast Imaging Subset DDSM) with binary malignancy labels
- **Evaluation**: Performance comparison across varying labeled data sizes (ablation study)

### Key Research Questions
1. How much does SSL improve over supervised baselines with limited labels?
2. What is the relationship between labeled data quantity and SSL benefit?
3. How reliable are pseudo-labels for medical image classification?

---

## 2. System Requirements

### 2.1 Hardware

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| **GPU** | GTX 1080 (8GB) | RTX 3090 (24GB) | CUDA backend for training |
| **Dev GPU** | Apple M1 (8GB) | M4 Pro (48GB) | MPS backend for development |
| **CPU** | 4 cores | 8+ cores | Data loading and augmentation |
| **RAM** | 16 GB | 32+ GB | Dataset loading and model training |
| **Storage** | 15 GB free | 50 GB free | Dataset (~6GB) + checkpoints + results |

### 2.2 Software

- **Python**: 3.10+ (tested on 3.13)
- **PyTorch**: 2.0+ with CUDA 11.8+ or MPS backend
- **OS**: macOS 12+ (Apple Silicon), Ubuntu 20.04+, Windows WSL2

### 2.3 Supported Devices

| Device | Backend | AMP Support | Notes |
|--------|---------|-------------|-------|
| NVIDIA GPU | CUDA | float16 + GradScaler | Full support, recommended for training |
| Apple Silicon | MPS | bfloat16, no GradScaler | Good for development and small runs |
| CPU | CPU | Disabled | Functional but slow, for testing only |

Device is auto-detected at runtime (CUDA > MPS > CPU). Override with `--device`:
```bash
python scripts/train_supervised.py --device cuda
python scripts/train_supervised.py --device mps
```

### 2.4 Performance Estimates

| Experiment | GPU Memory | Time (CUDA 8GB) | Time (MPS) |
|------------|------------|------------------|------------|
| Quick test (224px, 2 epochs) | 2-4 GB | 2-5 min | 5-10 min |
| Supervised (512px, 100 labels, 100 epochs) | ~6 GB | 10-15 min | 30-60 min |
| FixMatch (512px, 100 labels, 150 epochs) | ~7 GB | 45-90 min | 2-4 hours |
| Full ablation (7 experiments) | ~7 GB | 6-8 hours | Not recommended |

---

## 3. Installation

### 3.1 Environment Setup

```bash
git clone <repository-url>
cd ssl
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
# .venv\Scripts\activate     # Windows

# Install core dependencies
pip install -r requirements.txt

# For GPU: install PyTorch with CUDA separately if needed
# See https://pytorch.org/get-started/locally/
```

### 3.2 Optional Dependencies

```bash
# Weights & Biases experiment tracking
pip install wandb
wandb login

# Development tools (tests, linting)
pip install -e ".[dev]"
```

### 3.3 Verify Installation

```bash
# Run test suite (no dataset required)
python -m pytest tests/ -v
```

Expected: **52 tests pass**. All tests use synthetic data -- no CBIS-DDSM dataset needed.

### 3.4 Verify Device

```python
import torch
from src.training.trainer import get_device

device = get_device()
print(f"PyTorch: {torch.__version__}")
print(f"Device: {device}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"MPS: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
```

---

## 4. Dataset Preparation

### 4.1 Download

The CBIS-DDSM dataset is available on [Kaggle](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset).

```bash
# Option A: Kaggle CLI
pip install kaggle
kaggle datasets download -d awsaf49/cbis-ddsm-breast-cancer-image-dataset
unzip cbis-ddsm-breast-cancer-image-dataset.zip -d data/

# Option B: Manual download from Kaggle web UI, extract to data/
```

### 4.2 Expected Structure

```
data/
├── csv/
│   ├── dicom_info.csv
│   ├── mass_case_description_train_set.csv
│   ├── mass_case_description_test_set.csv
│   ├── calc_case_description_train_set.csv
│   ├── calc_case_description_test_set.csv
│   └── meta.csv
└── jpeg/
    ├── 1.3.6.1.4.1.9590.100.1.2.../
    │   ├── 1-263.jpg
    │   └── 2-241.jpg
    └── [9,999+ similar folders]
```

### 4.3 Dataset Statistics

| Split | Type | Total | Malignant | Benign | Patients |
|-------|------|-------|-----------|--------|----------|
| Train | Mass | 1,231 | 637 | 594 | ~691 |
| Test | Mass | 361 | 143 | 218 | ~202 |

The "mass" abnormality type is used by default. Labels are binary: `MALIGNANT` (1) vs `BENIGN`/`BENIGN_WITHOUT_CALLBACK` (0).

**Patient-aware splitting**: The train set is split into train/val by patient ID (not image index), ensuring no patient appears in both sets. This prevents data leakage from multi-view mammograms of the same patient. The test set uses CBIS-DDSM's official predefined split.

### 4.4 Validate Dataset Loading

```bash
python -c "
from src.data.dataset import CBISDDSMDataset
from src.data.transforms import get_transforms

transform = get_transforms('test', image_size=224)
dataset = CBISDDSMDataset(split='train', abnormality_type='mass',
                          labeled_subset_size=10, transform=transform, data_dir='data')
print(f'Loaded: {len(dataset)} samples')
print(f'Classes: {dataset.get_class_counts()}')
img, label = dataset[0]
print(f'Shape: {img.shape}, Label: {label}')
"
```

---

## 5. Configuration

### 5.1 Configuration Files

| File | Purpose |
|------|---------|
| `configs/default.yaml` | Supervised baseline (512px, batch 8, 100 epochs) |
| `configs/fixmatch.yaml` | FixMatch SSL -- paper-aligned, identical base params to supervised |
| `configs/test.yaml` | Quick validation (224px, 2 epochs, AMP off, no pretrained weights) |

**Design principle**: `default.yaml` and `fixmatch.yaml` share identical base hyperparameters (LR, weight decay, dropout, augmentation, backbone freeze schedule). The only differences in `fixmatch.yaml` are the SSL-specific additions (more epochs/patience, pseudo-labels, EMA). This isolates the SSL effect for clean ablation comparison.

### 5.2 Key Parameters

#### Dataset
```yaml
dataset:
  data_dir: "data"
  abnormality_type: "mass"   # mass, calc, both
  image_size: 512
  labeled_subset_size: null   # null = all; integer for ablation
```

#### Model
```yaml
model:
  name: "efficientnet-b0"
  num_classes: 2
  pretrained: true
  dropout_rate: 0.2
  freeze_backbone: true        # freeze backbone for initial epochs
  freeze_backbone_epochs: 5    # unfreeze after 5 epochs
```

#### Training
```yaml
training:
  batch_size: 8                # fits 8GB VRAM at 512px
  num_epochs: 100              # 150 for FixMatch
  learning_rate: 0.001
  weight_decay: 0.0001
  optimizer: "adam"            # adam, adamw, sgd
  scheduler: "cosine"         # cosine, step, plateau, null
  warmup_epochs: 5            # linear LR warmup
  early_stopping_patience: 20 # 25 for FixMatch
  use_amp: true               # mixed precision training
  gradient_clipping: true
  max_grad_norm: 1.0
  class_weighted_loss: true   # inverse-frequency weighting
  num_workers: 20
```

#### SSL (FixMatch)
```yaml
ssl:
  confidence_threshold: 0.95  # paper default -- only high-confidence pseudo-labels
  lambda_u: 1.0               # paper default -- full unsupervised loss weight
  unlabeled_batch_ratio: 2    # unlabeled/labeled batch ratio (fits 8GB VRAM)
  use_ema: true               # EMA model for stable pseudo-labels
  ema_decay: 0.999
  distribution_alignment: false  # disabled for conservative baseline
  randaugment_n: 2
  randaugment_m: 10
```

#### Augmentation
```yaml
augmentation:
  weak:
    random_horizontal_flip: true
    random_vertical_flip: true
    random_rotation: 5         # degrees
    color_jitter: 0.1          # brightness/contrast/saturation
```

#### Weights & Biases (optional)
```yaml
wandb:
  enabled: false
  project: "ssl-mammography"
  run_name: null               # auto-generated if null
```

### 5.3 Parameter Sensitivity

| Parameter | Range | Effect | Default |
|-----------|-------|--------|---------|
| `confidence_threshold` | 0.90-0.99 | Higher = fewer but more reliable pseudo-labels | 0.95 |
| `lambda_u` | 0.5-2.0 | Unsupervised loss weight | 1.0 |
| `unlabeled_batch_ratio` | 2-7 | Unlabeled samples per labeled batch | 2 |
| `learning_rate` | 1e-4 to 1e-2 | Convergence speed | 0.001 |
| `image_size` | 224-1024 | Detail vs memory/speed | 512 |
| `ema_decay` | 0.99-0.9999 | EMA smoothing (higher = slower update) | 0.999 |

---

## 6. Running Experiments

### 6.1 Quick Validation

```bash
python scripts/train_supervised.py \
  --config configs/test.yaml \
  --labeled_subset 10 \
  --output_dir results/test \
  --max_epochs 2
```

### 6.2 Supervised Baseline

```bash
# Full training data (upper bound)
python scripts/train_supervised.py \
  --config configs/default.yaml \
  --output_dir results/supervised_full

# Limited labels (ablation)
python scripts/train_supervised.py \
  --config configs/default.yaml \
  --labeled_subset 100 \
  --output_dir results/supervised_100
```

### 6.3 FixMatch SSL

```bash
python scripts/train_fixmatch.py \
  --config configs/fixmatch.yaml \
  --labeled 100 \
  --output_dir results/fixmatch_100
```

### 6.4 Full Ablation Study

The ablation script runs all 7 experiments sequentially -- designed for overnight runs on a CUDA workstation:

```bash
# Default: results go to results/
chmod +x run_ablation.sh
./run_ablation.sh

# Custom output directory
./run_ablation.sh -o results_v2

# Run in background (survives SSH disconnect)
nohup ./run_ablation.sh -o results 2>&1 &

# Monitor progress
tail -f results/ablation.log
```

**Experiments run by the ablation script:**

| # | Type | Labels | Output Dir |
|---|------|--------|------------|
| 1 | Supervised | 100 | `results/supervised_100` |
| 2 | Supervised | 250 | `results/supervised_250` |
| 3 | Supervised | 500 | `results/supervised_500` |
| 4 | Supervised | ALL | `results/supervised_full` |
| 5 | FixMatch | 100 | `results/fixmatch_100` |
| 6 | FixMatch | 250 | `results/fixmatch_250` |
| 7 | FixMatch | 500 | `results/fixmatch_500` |

The script logs timestamps, continues past individual failures, and prints a summary table at the end.

### 6.5 Training Output

Each experiment saves to its `--output_dir`:
```
results/fixmatch_100/
├── config.yaml              # Experiment config snapshot
├── best_model.pth           # Best model (by val AUC)
├── checkpoint_epoch_10.pth  # Periodic checkpoints
├── test_metrics.yaml        # Final test set metrics
└── training_history.csv     # Per-epoch metrics
```

**Supervised history columns**: `train_loss`, `val_loss`, `train_acc`, `val_acc`, `train_auc`, `val_auc`, `learning_rate`

**FixMatch history columns**: all of the above plus `sup_loss`, `unsup_loss`, `mask_ratio`, `lambda_u`

### 6.6 W&B Experiment Tracking

Enable in config:
```yaml
wandb:
  enabled: true
  project: "ssl-mammography"
  run_name: "fixmatch-100-v1"
```

Or stay with CSV logging (default). Both work independently.

---

## 7. SSH Workflow (Local Dev + Remote Training)

This project is designed for developing on Apple Silicon and training on a CUDA workstation.

### 7.1 Sync Code to Workstation

```bash
rsync -avz --exclude='.venv' --exclude='data' --exclude='results' \
  --exclude='__pycache__' --exclude='.ruff_cache' \
  ./ user@workstation:~/ssl/
```

### 7.2 Remote Setup (first time)

```bash
ssh user@workstation
cd ~/ssl
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Download dataset on workstation (or rsync data/ separately)
```

### 7.3 Run Training Remotely

```bash
# Interactive
ssh user@workstation "cd ~/ssl && source .venv/bin/activate && \
  python scripts/train_fixmatch.py --config configs/fixmatch.yaml --labeled 100"

# Background overnight run
ssh user@workstation
cd ~/ssl && source .venv/bin/activate
nohup ./run_ablation.sh 2>&1 &
```

### 7.4 Pull Results Back

```bash
rsync -avz user@workstation:~/ssl/results/ ./results/
```

---

## 8. Interpreting Results

### 8.1 Key Metrics

#### Primary: AUC-ROC
- **Range**: 0.0 (worst) to 1.0 (best)
- **Meaning**: Probability the model ranks a random positive higher than a random negative
- **Target**: > 0.85 for clinical utility, > 0.90 for strong performance

#### Secondary Metrics
| Metric | Interpretation | Target |
|--------|---------------|--------|
| **Sensitivity** (TP/(TP+FN)) | Ability to detect cancer | > 0.85 (critical) |
| **Specificity** (TN/(TN+FP)) | Ability to rule out non-cancer | > 0.80 |
| **F1-Score** | Precision-recall balance | > 0.80 |
| **Accuracy** | Overall correctness | > 0.80 |

### 8.2 Comparing Results

```python
import yaml

def load_metrics(exp_dir):
    with open(f"{exp_dir}/test_metrics.yaml") as f:
        return yaml.safe_load(f)

ssl = load_metrics('results/fixmatch_100')
sup = load_metrics('results/supervised_100')

print(f"AUC improvement: {ssl['auc'] - sup['auc']:.3f}")
print(f"Sensitivity improvement: {ssl['sensitivity'] - sup['sensitivity']:.3f}")
```

### 8.3 SSL-Specific Metrics

FixMatch training history includes:
- **mask_ratio**: Fraction of unlabeled samples above confidence threshold. With tau=0.95, expect 0.0 for early epochs (model not confident enough), gradually rising as the model improves. Healthy range: 0.1-0.5 after warmup.
- **sup_loss**: Supervised loss on labeled data
- **unsup_loss**: Consistency loss on pseudo-labeled data

**Warning signs**:
- mask_ratio near 0.0 throughout training: model never becomes confident enough. Consider lowering threshold.
- mask_ratio shoots to 0.5+ in early epochs: threshold too low, pseudo-labels are unreliable noise.
- Train AUC near 1.0 but val AUC stuck below 0.6: overfitting to labeled set, pseudo-labels reinforcing errors.

### 8.4 Overfitting Indicators

| Train-Val AUC Gap | Severity | Action |
|-------------------|----------|--------|
| < 0.10 | Healthy | None |
| 0.10 - 0.20 | Mild | Monitor, may resolve with more epochs |
| 0.20 - 0.30 | Significant | Consider more regularization |
| > 0.30 | Severe | Reduce model capacity, add dropout, reduce LR |

---

## 9. Visualization

### 9.1 Quick Training Curve Plot

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results/fixmatch_100/training_history.csv')

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(df['train_loss'], label='Train')
axes[0].plot(df['val_loss'], label='Val')
axes[0].set_ylabel('Loss'); axes[0].legend()

axes[1].plot(df['val_auc'], label='Val AUC')
axes[1].set_ylabel('AUC'); axes[1].legend()

axes[2].plot(df['mask_ratio'], label='Mask Ratio')
axes[2].set_ylabel('Ratio'); axes[2].legend()

plt.tight_layout()
plt.savefig('training_curves.png')
```

### 9.2 Ablation Comparison Table

```python
import yaml
from pathlib import Path

experiments = [
    ('Supervised 100', 'results/supervised_100'),
    ('Supervised 250', 'results/supervised_250'),
    ('Supervised 500', 'results/supervised_500'),
    ('Supervised Full', 'results/supervised_full'),
    ('FixMatch 100', 'results/fixmatch_100'),
    ('FixMatch 250', 'results/fixmatch_250'),
    ('FixMatch 500', 'results/fixmatch_500'),
]

print(f"{'Experiment':<20} {'AUC':>6} {'Acc':>6} {'F1':>6} {'Sens':>6} {'Spec':>6}")
print('-' * 62)
for name, path in experiments:
    metrics_file = Path(path) / 'test_metrics.yaml'
    if metrics_file.exists():
        m = yaml.safe_load(metrics_file.read_text())
        print(f"{name:<20} {m['auc']:>6.3f} {m['accuracy']:>6.3f} {m['f1']:>6.3f} "
              f"{m['sensitivity']:>6.3f} {m['specificity']:>6.3f}")
```

---

## 10. Troubleshooting

### SSL Certificate Errors (Pretrained Weights)

```bash
# Option A: Disable verification
PYTHONHTTPSVERIFY=0 python scripts/train_supervised.py ...

# Option B: Skip pretrained weights
# Set model.pretrained: false in config
```

### Out of Memory

1. Reduce `training.batch_size` (4 for 512px, 8 for 224px)
2. Reduce `dataset.image_size` (224 instead of 512)
3. AMP is enabled by default (`training.use_amp: true`) -- this already halves memory
4. Reduce `ssl.unlabeled_batch_ratio` (1 instead of 2)

### NaN Loss

Gradient clipping is enabled by default (`training.gradient_clipping: true`, `max_grad_norm: 1.0`). If you still get NaN:
1. Lower learning rate: `training.learning_rate: 0.0001`
2. Check your data for corrupted images

### Slow Data Loading

1. Increase `training.num_workers` (up to CPU core count)
2. `pin_memory` is automatically enabled only for CUDA (incompatible with MPS)

### MPS-Specific Issues

If AMP causes issues on Apple Silicon:
```yaml
training:
  use_amp: false
```

### Debugging Checklist

```bash
# 1. Tests pass?
python -m pytest tests/ -v

# 2. Device detected?
python -c "from src.training.trainer import get_device; print(get_device())"

# 3. Dataset loadable?
python -c "from src.data.dataset import CBISDDSMDataset; d = CBISDDSMDataset(data_dir='data'); print(len(d))"

# 4. Quick training works?
python scripts/train_supervised.py --config configs/test.yaml --output_dir /tmp/test
```

---

## 11. Architecture Reference

### 11.1 Data Flow

**Supervised pipeline:**
```
CBISDDSMDataset(transform=None)  ->  PIL Image
    |
    v
TransformSubset(weak_transform)  ->  Tensor  ->  DataLoader  ->  BaseTrainer
TransformSubset(test_transform)  ->  Tensor  ->  DataLoader  ->  (validation)
```

**FixMatch pipeline:**
```
CBISDDSMDataset(transform=None)  ->  PIL Image
    |
    +-> FixMatchLabeledDataset(weak_transform)      -> (tensor, label)
    +-> FixMatchUnlabeledDataset(weak_t, strong_t)  -> (tensor_weak, tensor_strong)
    +-> TransformSubset(test_transform)              -> (tensor, label) [validation]
    |
    v
DataLoaders  ->  FixMatchTrainer
```

Key principle: the dataset returns **raw PIL images**. All augmentation is applied by wrapper datasets, ensuring each pipeline gets exactly the transforms it needs.

### 11.2 Patient-Aware Splitting

The `patient_aware_split()` function in `src/data/dataset.py` ensures no patient appears in both train and val sets. In medical imaging, the same patient may have multiple images (left/right breast, CC/MLO views). Splitting by image index would leak patient-specific features into validation, inflating metrics.

The split is:
1. Group images by `patient_id` from the case-description CSV
2. Stratify patients by majority label (benign vs malignant)
3. Assign ~15% of patients to validation, rest to train
4. Deterministic with seed=42

Both `train_supervised.py` and `train_fixmatch.py` use the same split function and seed, ensuring identical validation sets for fair comparison.

### 11.3 Training Features

| Feature | Where | Config Key |
|---------|-------|------------|
| Mixed precision (AMP) | `BaseTrainer` | `training.use_amp` |
| Linear LR warmup | `BaseTrainer._apply_warmup()` | `training.warmup_epochs` |
| Gradient clipping | `BaseTrainer.train_epoch()` | `training.gradient_clipping`, `max_grad_norm` |
| Class-weighted loss | `BaseTrainer.__init__()` | `training.class_weighted_loss` |
| Backbone freeze/unfreeze | `EfficientNetClassifier` | `model.freeze_backbone`, `freeze_backbone_epochs` |
| EMA teacher model | `FixMatchTrainer` + `EMAModel` | `ssl.use_ema`, `ssl.ema_decay` |
| EMA refresh on unfreeze | `FixMatchTrainer.train()` | Automatic when backbone unfreezes |
| Pseudo-label logging | `FixMatchTrainer.train_epoch_ssl()` | Always on (mask_ratio, sup/unsup loss) |
| Checkpoint resume | `FixMatchTrainer.train(resume_from=...)` | Full SSL state preserved |
| W&B integration | `BaseTrainer._init_wandb()` | `wandb.enabled` |
| Per-experiment checkpoints | `BaseTrainer.save_checkpoint()` | Saves to `--output_dir` |

### 11.4 Extending with New SSL Methods

1. Subclass `BaseTrainer`:
```python
class MeanTeacherTrainer(BaseTrainer):
    def __init__(self, model, config, device=None, output_dir=None, class_weights=None):
        super().__init__(model, config, device, output_dir, class_weights)
        self.ema = EMAModel(model, decay=config['ssl']['ema_decay'])

    def train_epoch_ssl(self, labeled_loader, unlabeled_loader):
        # Your training logic here
        pass
```

2. Create a training script in `scripts/` following the pattern of `train_fixmatch.py`.

3. Add tests in `tests/` using `SyntheticDataset` from `conftest.py`.

---

## 12. References

- **FixMatch**: Sohn et al., *FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence*, NeurIPS 2020
- **CBIS-DDSM**: Lee et al., *A curated mammography data set for use in computer-aided detection and diagnosis research*, Scientific Data 2017
- **EfficientNet**: Tan & Le, *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*, ICML 2019

---

## Appendix: Command Reference

```bash
# ---- Setup ----
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install wandb                  # optional
pip install -e ".[dev]"            # optional

# ---- Tests ----
python -m pytest tests/ -v         # 52 tests, no dataset needed

# ---- Training ----
python scripts/train_supervised.py --config configs/default.yaml --labeled_subset 100 --output_dir results/sup_100
python scripts/train_fixmatch.py   --config configs/fixmatch.yaml --labeled 100 --output_dir results/fm_100
./run_ablation.sh                  # full ablation study (7 experiments)

# ---- Options ----
--max_epochs 50                    # override epoch count
--device cuda                      # force device
--output_dir results/my_experiment # per-experiment output

# ---- SSH Workflow ----
rsync -avz --exclude='.venv' --exclude='data' ./ user@workstation:~/ssl/
ssh user@workstation "cd ~/ssl && nohup ./run_ablation.sh 2>&1 &"
rsync -avz user@workstation:~/ssl/results/ ./results/
```

### File Locations

| File | Purpose |
|------|---------|
| `configs/default.yaml` | Supervised baseline config |
| `configs/fixmatch.yaml` | FixMatch SSL config (paper-aligned) |
| `configs/test.yaml` | Quick validation config |
| `scripts/train_supervised.py` | Supervised baseline training |
| `scripts/train_fixmatch.py` | FixMatch SSL training |
| `run_ablation.sh` | Full ablation study (7 experiments) |
| `src/training/trainer.py` | Base trainer (AMP, warmup, etc.) |
| `src/training/fixmatch_trainer.py` | FixMatch implementation |
| `src/training/ema.py` | EMA model utility |
| `src/data/dataset.py` | CBIS-DDSM dataset loader + patient-aware split |
| `src/data/ssl_dataset.py` | SSL dataset wrappers |
| `src/data/transforms.py` | Augmentation pipeline |
| `src/models/efficientnet.py` | EfficientNet-B0 classifier |
| `tasks/todo.md` | Project TODO and version history |
| `tasks/lessons.md` | Patterns and anti-patterns discovered |
