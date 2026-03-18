# SSL Mammography: Semi-Supervised Learning for Breast Cancer Detection

This project implements semi-supervised learning (SSL) methods on the CBIS-DDSM mammography dataset to improve classification performance with limited labeled data.

## Project Overview

- **Dataset**: CBIS-DDSM (Curated Breast Imaging Subset DDSM) - mammography images with benign/malignant labels
- **Task**: Binary classification (benign vs. malignant) using full mammograms resized to 512×512
- **SSL Method**: FixMatch algorithm with confidence-thresholded pseudo-labeling
- **Model**: EfficientNet-B0 backbone
- **Evaluation**: AUC-ROC primary metric, with accuracy, F1-score, sensitivity, specificity

## Key Features

- Implementation of FixMatch semi-supervised learning algorithm
- Ablation study with varying labeled data subsets (e.g., 100, 250, 500 samples)
- Comparison against supervised baseline and fully-supervised upper bound
- Visualization of embeddings using t-SNE/UMAP
- Modular code structure with unit tests
- Configurable training pipelines with YAML configuration files

## Project Structure

```
.
├── src/                   # Source code modules
│   ├── data/             # Dataset loading and preprocessing
│   ├── models/           # Model definitions
│   └── training/         # Training loops, metrics, trainers
├── notebooks/            # Jupyter notebooks for exploration and visualization
├── scripts/              # Training and evaluation scripts
├── configs/              # Configuration files (YAML)
├── tests/                # Unit tests
├── data/                 # Dataset (not included in repo)
└── tasks/                # Project management files
```

## Installation

1. Clone repository:
   ```bash
   git clone <repository-url>
   cd ssl
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   For GPU support, install PyTorch with CUDA separately (see [PyTorch website](https://pytorch.org/)).

3. Download CBIS-DDSM dataset from [Kaggle](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset) and extract into `data/` directory. The expected structure is:
   ```
   data/
   ├── csv/               # CSV annotation files
   └── jpeg/              # JPEG images (organized by SeriesInstanceUID)
   ```

## Quick Start

After installation, you can validate your environment and run a quick test:

```bash
# Make helper scripts executable
chmod +x quick_start.sh run_ablation.sh

# Run quick validation (tests + small training)
./quick_start.sh
```

For detailed instructions, see the comprehensive [USER_GUIDE.md](USER_GUIDE.md).

## Dataset Notes

The dataset contains full mammograms, cropped regions, and ROI masks. This project uses only full mammogram images. Labels are binary: `MALIGNANT` (1) vs. `BENIGN`/`BENIGN_WITHOUT_CALLBACK` (0). The dataset is split into training and test sets as provided.

## Usage

### Configuration

Modify `configs/default.yaml` to adjust hyperparameters, model settings, and augmentation strategies.

### Supervised Baseline Training

Train a supervised model with a limited labeled subset:

```bash
python scripts/train_supervised.py --config configs/default.yaml --labeled_subset 100 --output_dir results/supervised_100
```

### FixMatch SSL Training

Train with FixMatch semi-supervised learning:

```bash
python scripts/train_fixmatch.py --config configs/default.yaml --labeled 100 --output_dir results/fixmatch_100
```

### Ablation Study

Run multiple experiments with different labeled subset sizes:

**Using helper script (recommended):**
```bash
./run_ablation.sh --config configs/default.yaml --output results
```

**Manual loop:**
```bash
for n in 100 250 500; do
  python scripts/train_supervised.py --config configs/default.yaml --labeled_subset $n --output_dir results/supervised_$n
  python scripts/train_fixmatch.py --config configs/default.yaml --labeled $n --output_dir results/fixmatch_$n
done
```

### Evaluation

Metrics are automatically computed on the test set and saved as YAML files. Training history (loss, accuracy, AUC) is saved as CSV.

### Visualization

Use Jupyter notebooks in `notebooks/` to:

- **`01_data_exploration.ipynb`**: Explore dataset statistics and sample images
- **`02_training_curves.ipynb`**: Visualize training curves, compare supervised vs SSL
- **`03_embeddings.ipynb`**: Generate t-SNE/UMAP embeddings of learned features
- **`04_model_interpretation.ipynb`**: Plot confidence distributions, pseudo-label quality, and model interpretation

## Results

Expected outcomes:
- SSL (FixMatch) should outperform supervised baseline when labeled data is scarce
- Performance improves as more labeled samples are available
- AUC-ROC is the primary metric for medical classification

## Troubleshooting

- **SSL certificate errors**: If downloading pretrained weights fails due to SSL issues, set `pretrained: false` in config or set environment variable `PYTHONHTTPSVERIFY=0`.
- **Memory issues**: Reduce batch size or image size in config.
- **Dataset loading warnings**: Pandas SettingWithCopy warnings can be ignored.

## License

MIT License - see LICENSE file for details.