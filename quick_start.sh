#!/bin/bash
# Quick-start script for SSL Mammography project
# This script validates the environment and runs a quick test training.

set -e  # Exit on error

echo "========================================="
echo "SSL Mammography - Quick Start"
echo "========================================="

# Check Python version
echo "Checking Python version..."
python --version | grep -E "3\.(8|9|10)\." || {
    echo "ERROR: Python 3.8, 3.9, or 3.10 required."
    exit 1
}

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run unit tests
echo "Running unit tests..."
python -m pytest tests/ -v

# Check dataset structure
echo "Checking dataset structure..."
if [ ! -d "data/csv" ] || [ ! -d "data/jpeg" ]; then
    echo "WARNING: Dataset not found. Please download CBIS-DDSM from Kaggle:"
    echo "  https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset"
    echo "Extract into 'data/' directory with subdirectories 'csv/' and 'jpeg/'."
    echo "Skipping training test..."
else
    # Run a quick supervised training (2 epochs, small subset)
    echo "Dataset found. Running quick supervised training..."
    python scripts/train_supervised.py \
        --config configs/test.yaml \
        --labeled_subset 10 \
        --output_dir test_output \
        --max_epochs 2
    
    echo "Quick training completed. Check 'test_output/' for logs and metrics."
fi

echo "========================================="
echo "Quick start completed successfully!"
echo "Next steps:"
echo "1. Review USER_GUIDE.md for detailed instructions"
echo "2. Run ablation study: ./run_ablation.sh"
echo "3. Explore notebooks in notebooks/"
echo "========================================="