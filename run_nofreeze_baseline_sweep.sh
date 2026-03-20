#!/bin/bash
# Official supervised no-freeze baseline sweep.

set -u

CONFIG_FILE="configs/default_nofreeze.yaml"
RESCUE_DIR="results_rescue"
FOLLOWUP_DIR="results_followup"
OUTPUT_DIR="results_nofreeze_baseline"
DEVICE=""
MAX_EPOCHS=""
RETRY_FAILED=0
SKIP_COMPLETED=1
DRY_RUN=0

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Runs the official supervised no-freeze baseline sweep:
  - 100 labels: seed 44
  - 250 labels: seed 44
  - 500 labels: seeds 42, 43, 44

Options:
  -c, --config FILE       Config file (default: configs/default_nofreeze.yaml)
  --rescue_dir DIR        Rescue results directory (default: results_rescue)
  --followup_dir DIR      Follow-up results directory (default: results_followup)
  -o, --output DIR        Output directory for sweep runs (default: results_nofreeze_baseline)
  --device DEVICE         Optional device override
  --max_epochs N          Optional max-epochs override
  --retry_failed N        Retry failed runs N times (default: 0)
  --skip_completed        Skip runs with existing test_metrics.yaml (default)
  --no-skip_completed     Re-run even if test_metrics.yaml already exists
  --dry_run               Log commands without executing them
  -h, --help              Show this help message
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --rescue_dir)
            RESCUE_DIR="$2"
            shift 2
            ;;
        --followup_dir)
            FOLLOWUP_DIR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --max_epochs)
            MAX_EPOCHS="$2"
            shift 2
            ;;
        --retry_failed)
            RETRY_FAILED="$2"
            shift 2
            ;;
        --skip_completed)
            SKIP_COMPLETED=1
            shift
            ;;
        --no-skip_completed)
            SKIP_COMPLETED=0
            shift
            ;;
        --dry_run)
            DRY_RUN=1
            shift
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
LOG_FILE="$OUTPUT_DIR/nofreeze_baseline.log"
FAILED_EXPERIMENTS=()

log() {
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $*" | tee -a "$LOG_FILE"
}

append_optional_args() {
    local cmd="$1"
    if [[ -n "$DEVICE" ]]; then
        cmd="$cmd --device \"$DEVICE\""
    fi
    if [[ -n "$MAX_EPOCHS" ]]; then
        cmd="$cmd --max_epochs \"$MAX_EPOCHS\""
    fi
    echo "$cmd"
}

run_experiment() {
    local name="$1"
    local exp_dir="$2"
    local cmd="$3"
    local test_metrics="$exp_dir/test_metrics.yaml"
    local attempt=0
    local exit_status=0

    if [[ "$SKIP_COMPLETED" -eq 1 && -f "$test_metrics" ]]; then
        log "SKIP: $name already completed at $exp_dir"
        return 0
    fi

    mkdir -p "$exp_dir"
    cmd=$(append_optional_args "$cmd")

    while true; do
        log "STARTING: $name (attempt $((attempt + 1)))"
        log "Command: $cmd"

        if [[ "$DRY_RUN" -eq 1 ]]; then
            log "DRY RUN: not executing $name"
            return 0
        fi

        eval "$cmd" 2>&1 | tee -a "$LOG_FILE"
        exit_status=${PIPESTATUS[0]}

        if [[ "$exit_status" -eq 0 ]]; then
            log "SUCCESS: $name"
            return 0
        fi

        attempt=$((attempt + 1))
        if [[ "$attempt" -gt "$RETRY_FAILED" ]]; then
            log "FAILED: $name exited with code $exit_status"
            FAILED_EXPERIMENTS+=("$name")
            return "$exit_status"
        fi

        log "RETRYING: $name after failure code $exit_status"
    done
}

build_experiments() {
    local experiments=(
        "supervised_nofreeze_100_seed44|$OUTPUT_DIR/supervised_nofreeze_100_seed44|python3 scripts/train_supervised.py --config \"$CONFIG_FILE\" --labeled_subset 100 --seed 44 --output_dir \"$OUTPUT_DIR/supervised_nofreeze_100_seed44\""
        "supervised_nofreeze_250_seed44|$OUTPUT_DIR/supervised_nofreeze_250_seed44|python3 scripts/train_supervised.py --config \"$CONFIG_FILE\" --labeled_subset 250 --seed 44 --output_dir \"$OUTPUT_DIR/supervised_nofreeze_250_seed44\""
        "supervised_nofreeze_500_seed42|$OUTPUT_DIR/supervised_nofreeze_500_seed42|python3 scripts/train_supervised.py --config \"$CONFIG_FILE\" --labeled_subset 500 --seed 42 --output_dir \"$OUTPUT_DIR/supervised_nofreeze_500_seed42\""
        "supervised_nofreeze_500_seed43|$OUTPUT_DIR/supervised_nofreeze_500_seed43|python3 scripts/train_supervised.py --config \"$CONFIG_FILE\" --labeled_subset 500 --seed 43 --output_dir \"$OUTPUT_DIR/supervised_nofreeze_500_seed43\""
        "supervised_nofreeze_500_seed44|$OUTPUT_DIR/supervised_nofreeze_500_seed44|python3 scripts/train_supervised.py --config \"$CONFIG_FILE\" --labeled_subset 500 --seed 44 --output_dir \"$OUTPUT_DIR/supervised_nofreeze_500_seed44\""
    )
    printf "%s\n" "${experiments[@]}"
}

if [[ ! -f "$CONFIG_FILE" ]]; then
    log "ERROR: Config file '$CONFIG_FILE' not found."
    exit 1
fi

log "=============================================="
log "SSL Mammography - Official No-Freeze Baselines"
log "=============================================="
log "Config:            $CONFIG_FILE"
log "Rescue directory:  $RESCUE_DIR"
log "Follow-up dir:     $FOLLOWUP_DIR"
log "Output directory:  $OUTPUT_DIR"
log "Dry run:           $DRY_RUN"
log "Skip completed:    $SKIP_COMPLETED"
log "Retry failed:      $RETRY_FAILED"

EXPERIMENTS="$(build_experiments)"
while IFS= read -r entry; do
    [[ -z "$entry" ]] && continue
    IFS="|" read -r name exp_dir cmd <<< "$entry"
    run_experiment "$name" "$exp_dir" "$cmd"
done <<EOF
$EXPERIMENTS
EOF

log "=== SSL CLOSURE SUMMARY ==="
python3 scripts/summarize_ssl_closure.py \
    --rescue_dir "$RESCUE_DIR" \
    --followup_dir "$FOLLOWUP_DIR" \
    --output_dir "$OUTPUT_DIR" | tee -a "$LOG_FILE"

if [[ "${#FAILED_EXPERIMENTS[@]}" -gt 0 ]]; then
    log "Failed experiments:"
    for name in "${FAILED_EXPERIMENTS[@]}"; do
        log "  - $name"
    done
else
    log "All official no-freeze baseline experiments completed or were skipped."
fi
