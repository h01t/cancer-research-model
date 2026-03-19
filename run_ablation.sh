#!/bin/bash
# Ablation study script for SSL Mammography project
# Runs supervised and FixMatch SSL experiments for varying labeled subset sizes.
# Designed for overnight runs on CUDA workstation — logs everything, continues past failures.

set -u  # Treat unset variables as errors
# DO NOT use set -e; we want to continue past individual experiment failures

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]
Run ablation study across labeled subset sizes (7 experiments total).

Experiments:
  1. Supervised baseline with 100 labeled samples
  2. Supervised baseline with 250 labeled samples
  3. Supervised baseline with 500 labeled samples
  4. Supervised baseline with ALL labeled samples (fully supervised upper bound)
  5. FixMatch SSL with 100 labeled samples
  6. FixMatch SSL with 250 labeled samples
  7. FixMatch SSL with 500 labeled samples

Options:
  -c, --config CONFIG_FILE   Configuration file for supervised (default: configs/default.yaml)
  -f, --fixmatch CONFIG_FILE Configuration file for FixMatch (default: configs/fixmatch.yaml)
  -o, --output OUTPUT_DIR    Base output directory (default: results)
  -h, --help                 Show this help message

Output:
  Each experiment creates its own subdirectory under OUTPUT_DIR (e.g., results/supervised_100).
  Full logs are written to OUTPUT_DIR/ablation.log via tee.
  At the end, a summary table is printed showing AUC, accuracy, sensitivity, specificity.

Example (overnight run):
  nohup ./run_ablation.sh -o results_20250319 2>&1 | tee results_20250319/ablation.log &
EOF
    exit 0
}

# Default values
SUPERVISED_CONFIG="configs/default.yaml"
FIXMATCH_CONFIG="configs/fixmatch.yaml"
OUTPUT_DIR="results"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            SUPERVISED_CONFIG="$2"
            shift 2
            ;;
        -f|--fixmatch)
            FIXMATCH_CONFIG="$2"
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
            echo "ERROR: Unknown option: $1"
            usage
            ;;
    esac
done

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Log file path
LOG_FILE="$OUTPUT_DIR/ablation.log"

# Function to log with timestamp
log() {
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $*" | tee -a "$LOG_FILE"
}

# Function to run an experiment and capture its exit status
run_experiment() {
    local name=$1
    local cmd=$2
    local start_time end_time elapsed
    
    log "STARTING: $name"
    log "Command: $cmd"
    start_time=$(date +%s)
    
    # Execute the command, capture both stdout and stderr, append to log
    eval "$cmd" 2>&1 | tee -a "$LOG_FILE"
    local exit_status=${PIPESTATUS[0]}
    
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    
    if [ $exit_status -eq 0 ]; then
        log "SUCCESS: $name completed in ${elapsed}s"
    else
        log "FAILED: $name exited with code $exit_status after ${elapsed}s"
    fi
    
    return $exit_status
}

# Print header
log "========================================="
log "SSL Mammography - Comprehensive Ablation Study"
log "========================================="
log "Supervised config: $SUPERVISED_CONFIG"
log "FixMatch config:   $FIXMATCH_CONFIG"
log "Output base:       $OUTPUT_DIR"
log "Log file:          $LOG_FILE"
log ""

# Check if config files exist
if [ ! -f "$SUPERVISED_CONFIG" ]; then
    log "ERROR: Supervised config file '$SUPERVISED_CONFIG' not found."
    exit 1
fi
if [ ! -f "$FIXMATCH_CONFIG" ]; then
    log "ERROR: FixMatch config file '$FIXMATCH_CONFIG' not found."
    exit 1
fi

# Arrays of experiments
SUPERVISED_SUBSETS=(100 250 500 "null")
FIXMATCH_SUBSETS=(100 250 500)

# Track overall status
OVERALL_SUCCESS=true
FAILED_EXPERIMENTS=()

# Run supervised experiments
log "=== SUPERVISED BASELINES ==="
for subset in "${SUPERVISED_SUBSETS[@]}"; do
    if [ "$subset" = "null" ]; then
        dir_name="supervised_full"
        subset_arg=""
    else
        dir_name="supervised_${subset}"
        subset_arg="--labeled_subset ${subset}"
    fi
    
    cmd="python3 scripts/train_supervised.py \
        --config \"$SUPERVISED_CONFIG\" \
        $subset_arg \
        --output_dir \"${OUTPUT_DIR}/${dir_name}\""
    
    if ! run_experiment "Supervised ($subset)" "$cmd"; then
        OVERALL_SUCCESS=false
        FAILED_EXPERIMENTS+=("Supervised ($subset)")
    fi
    
    log ""
done

# Run FixMatch experiments
log "=== FIXMATCH SSL EXPERIMENTS ==="
for subset in "${FIXMATCH_SUBSETS[@]}"; do
    cmd="python3 scripts/train_fixmatch.py \
        --config \"$FIXMATCH_CONFIG\" \
        --labeled ${subset} \
        --output_dir \"${OUTPUT_DIR}/fixmatch_${subset}\""
    
    if ! run_experiment "FixMatch ($subset)" "$cmd"; then
        OVERALL_SUCCESS=false
        FAILED_EXPERIMENTS+=("FixMatch ($subset)")
    fi
    
    log ""
done

# Generate summary table
log "=== RESULTS SUMMARY ==="
log "Experiment                    | AUC      | Accuracy | Sensitivity | Specificity"
log "------------------------------|----------|----------|-------------|-------------"

for subset in "${SUPERVISED_SUBSETS[@]}"; do
    if [ "$subset" = "null" ]; then
        dir_name="supervised_full"
    else
        dir_name="supervised_${subset}"
    fi
    
    metrics_file="${OUTPUT_DIR}/${dir_name}/test_metrics.yaml"
    if [ -f "$metrics_file" ]; then
        # Extract metrics using yq (if available) or grep
        auc=$(grep -E "^auc:" "$metrics_file" | awk '{print $2}' 2>/dev/null || echo "N/A")
        acc=$(grep -E "^accuracy:" "$metrics_file" | awk '{print $2}' 2>/dev/null || echo "N/A")
        sens=$(grep -E "^sensitivity:" "$metrics_file" | awk '{print $2}' 2>/dev/null || echo "N/A")
        spec=$(grep -E "^specificity:" "$metrics_file" | awk '{print $2}' 2>/dev/null || echo "N/A")
        
        # Format the line
        printf "Supervised (%4s)           | %8s | %8s | %11s | %11s\n" \
            "$subset" "$auc" "$acc" "$sens" "$spec" | tee -a "$LOG_FILE"
    else
        log "Supervised ($subset)           | MISSING  | MISSING  | MISSING     | MISSING"
    fi
done

for subset in "${FIXMATCH_SUBSETS[@]}"; do
    metrics_file="${OUTPUT_DIR}/fixmatch_${subset}/test_metrics.yaml"
    if [ -f "$metrics_file" ]; then
        auc=$(grep -E "^auc:" "$metrics_file" | awk '{print $2}' 2>/dev/null || echo "N/A")
        acc=$(grep -E "^accuracy:" "$metrics_file" | awk '{print $2}' 2>/dev/null || echo "N/A")
        sens=$(grep -E "^sensitivity:" "$metrics_file" | awk '{print $2}' 2>/dev/null || echo "N/A")
        spec=$(grep -E "^specificity:" "$metrics_file" | awk '{print $2}' 2>/dev/null || echo "N/A")
        
        printf "FixMatch (%4s)             | %8s | %8s | %11s | %11s\n" \
            "$subset" "$auc" "$acc" "$sens" "$spec" | tee -a "$LOG_FILE"
    else
        log "FixMatch ($subset)             | MISSING  | MISSING  | MISSING     | MISSING"
    fi
done

log ""

# Final status
if [ "$OVERALL_SUCCESS" = true ]; then
    log "✅ All experiments completed successfully!"
    log "Results are ready for analysis in: $OUTPUT_DIR/"
    log "Next steps:"
    log "  - Review individual experiment directories for detailed metrics and logs"
    log "  - Use notebooks/02_training_curves.ipynb for visualization"
    log "  - Compare AUC-ROC across subsets to quantify SSL benefit"
else
    log "⚠️  Some experiments failed:"
    for exp in "${FAILED_EXPERIMENTS[@]}"; do
        log "    - $exp"
    done
    log ""
    log "Check $LOG_FILE for detailed error messages."
    log "Successful experiments are available in: $OUTPUT_DIR/"
fi

log "========================================="
log "Ablation study finished at $(date '+%Y-%m-%d %H:%M:%S')"
log "========================================="