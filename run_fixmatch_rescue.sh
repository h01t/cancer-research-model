#!/bin/bash
# Targeted FixMatch rescue sweep with seed-controlled supervised baselines.

set -u

SUPERVISED_CONFIG="configs/default.yaml"
FIXMATCH_CONFIG="configs/fixmatch.yaml"
OUTPUT_DIR="results_rescue"
SEEDS=(42 43 44)
SUBSETS=(100 250)

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Runs the targeted rescue matrix:
  - Supervised: subsets 100, 250 x seeds 42, 43, 44
  - FixMatch:   subsets 100, 250 x seeds 42, 43, 44

Options:
  -c, --config CONFIG_FILE    Supervised config (default: configs/default.yaml)
  -f, --fixmatch CONFIG_FILE  FixMatch config (default: configs/fixmatch.yaml)
  -o, --output OUTPUT_DIR     Output directory (default: results_rescue)
  -h, --help                  Show this help message
EOF
    exit 0
}

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

mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/rescue.log"

log() {
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $*" | tee -a "$LOG_FILE"
}

run_experiment() {
    local name=$1
    local cmd=$2
    local start_time end_time elapsed

    log "STARTING: $name"
    log "Command: $cmd"
    start_time=$(date +%s)

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

log "========================================="
log "SSL Mammography - Targeted FixMatch Rescue"
log "========================================="
log "Supervised config: $SUPERVISED_CONFIG"
log "FixMatch config:   $FIXMATCH_CONFIG"
log "Output base:       $OUTPUT_DIR"
log ""

for subset in "${SUBSETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_experiment \
            "Supervised (${subset}, seed=${seed})" \
            "python3 scripts/train_supervised.py \
                --config \"$SUPERVISED_CONFIG\" \
                --labeled_subset ${subset} \
                --seed ${seed} \
                --output_dir \"${OUTPUT_DIR}/supervised_${subset}_seed${seed}\""
        log ""

        run_experiment \
            "FixMatch (${subset}, seed=${seed})" \
            "python3 scripts/train_fixmatch.py \
                --config \"$FIXMATCH_CONFIG\" \
                --labeled ${subset} \
                --seed ${seed} \
                --output_dir \"${OUTPUT_DIR}/fixmatch_${subset}_seed${seed}\""
        log ""
    done
done

log "=== RESCUE SUMMARY ==="
python3 scripts/summarize_rescue.py --output_dir "$OUTPUT_DIR" --seeds "${SEEDS[@]}" | tee -a "$LOG_FILE"
log "========================================="
