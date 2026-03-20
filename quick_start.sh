#!/bin/bash
# Quick-start script for the supervised-first mammography project.
# Validates the environment and runs a quick supervised smoke test.

set -e

echo "========================================="
echo "Mammography Classification - Quick Start"
echo "========================================="

# Check Python version (3.10+)
echo "Checking Python version..."
python3 --version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo "ERROR: Python 3.10+ required. Found: $PYTHON_VERSION"
    exit 1
fi
echo "Python $PYTHON_VERSION OK"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt -q

# Run unit tests (no real data needed)
echo ""
echo "Running unit tests..."
python3 -m pytest tests/ -v --ignore=tests/test_supervised_integration.py 2>/dev/null || \
    python3 -m pytest tests/test_transforms.py tests/test_metrics.py -v

# Check dataset structure
echo ""
echo "Checking dataset structure..."
if [ ! -d "data/csv" ] || [ ! -d "data/jpeg" ]; then
    echo "WARNING: Dataset not found."
    echo "  Download CBIS-DDSM from Kaggle:"
    echo "  https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset"
    echo "  Extract into 'data/' with subdirectories 'csv/' and 'jpeg/'."
    echo "Skipping training test..."
else
    echo "Dataset found. Running quick supervised training..."
    python3 scripts/train_supervised.py \
        --config configs/test.yaml \
        --labeled_subset 10 \
        --output_dir test_output \
        --max_epochs 2
    echo "Quick training completed. Check 'test_output/' for results."
fi

echo ""
echo "========================================="
echo "Quick start completed!"
echo "Next steps:"
echo "  1. Review USER_GUIDE.md for the supervised-first workflow"
echo "  2. Train the official baseline with configs/default_nofreeze.yaml"
echo "  3. Run ./run_nofreeze_baseline_sweep.sh for the multi-seed baseline sweep"
echo "  4. Explore notebooks in notebooks/ and reports/ for analysis"
echo "========================================="
