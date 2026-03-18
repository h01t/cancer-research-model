#!/bin/bash
# Ablation study script for SSL Mammography project
# Runs supervised and FixMatch SSL experiments for varying labeled subset sizes.

set -e  # Exit on error

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Run ablation study across labeled subset sizes."
    echo ""
    echo "Options:"
    echo "  -c, --config CONFIG_FILE   Configuration file (default: configs/default.yaml)"
    echo "  -o, --output OUTPUT_DIR    Base output directory (default: results)"
    echo "  -h, --help                 Show this help message"
    exit 0
}

# Default values
CONFIG="configs/default.yaml"
OUTPUT_DIR="results"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

echo "========================================="
echo "SSL Mammography - Ablation Study"
echo "========================================="
echo "Configuration: $CONFIG"
echo "Output base: $OUTPUT_DIR"
echo "Labeled subset sizes: 100 250 500"
echo ""

# Check if config file exists
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file '$CONFIG' not found."
    exit 1
fi

# Create output directory if needed
mkdir -p "$OUTPUT_DIR"

# Array of labeled subset sizes
SUBSETS=(100 250 500)

for subset in "${SUBSETS[@]}"; do
    echo "=== Running experiments with $subset labeled samples ==="
    
    # Supervised baseline
    echo "  Supervised baseline..."
    python3 scripts/train_supervised.py \
        --config "$CONFIG" \
        --labeled_subset "$subset" \
        --output_dir "${OUTPUT_DIR}/supervised_${subset}"
    
    # FixMatch SSL
    echo "  FixMatch SSL..."
    python3 scripts/train_fixmatch.py \
        --config "$CONFIG" \
        --labeled "$subset" \
        --output_dir "${OUTPUT_DIR}/fixmatch_${subset}"
    
    echo "  Completed subset $subset"
    echo ""
done

echo "========================================="
echo "Ablation study completed!"
echo "Results saved in: $OUTPUT_DIR/"
echo "To analyze results:"
echo "  - Check individual experiment directories for metrics and logs"
echo "  - Use notebooks/02_training_curves.ipynb for visualization"
echo "  - Compare AUC-ROC across subsets"
echo "========================================="