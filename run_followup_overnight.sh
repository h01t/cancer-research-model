#!/bin/bash
# Gate-aware follow-up overnight runner for FixMatch confirmation or Mean Teacher pivot.

set -u

RESCUE_DIR="results_rescue"
OUTPUT_DIR="results_followup"
PROFILE="confirm_fixmatch"
DEVICE=""
MAX_EPOCHS=""
RETRY_FAILED=0
SKIP_COMPLETED=1
DRY_RUN=0

SUPERVISED_CONFIG="configs/default.yaml"
FIXMATCH_CONFIG="configs/fixmatch.yaml"
FIXMATCH_STATIC_CONFIG="configs/fixmatch_static.yaml"
FIXMATCH_LEGACY_AUG_CONFIG="configs/fixmatch_legacy_aug.yaml"
MEAN_TEACHER_CONFIG="configs/mean_teacher.yaml"

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --rescue_dir DIR        Completed rescue results directory (default: results_rescue)
  -o, --output DIR        Output directory for follow-up runs (default: results_followup)
  --device DEVICE         Optional device override passed to training scripts
  --profile PROFILE       Follow-up profile (default: confirm_fixmatch)
  --max_epochs N          Optional max-epochs override for all runs
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
        --rescue_dir)
            RESCUE_DIR="$2"
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
        --profile)
            PROFILE="$2"
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
LOG_FILE="$OUTPUT_DIR/followup.log"
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

get_rescue_decision() {
    python3 scripts/summarize_rescue.py --output_dir "$RESCUE_DIR" --json
}

build_pass_branch() {
    local experiments=(
        "supervised_500_seed42|$OUTPUT_DIR/supervised_500_seed42|python3 scripts/train_supervised.py --config \"$SUPERVISED_CONFIG\" --labeled_subset 500 --seed 42 --output_dir \"$OUTPUT_DIR/supervised_500_seed42\""
        "fixmatch_500_seed42|$OUTPUT_DIR/fixmatch_500_seed42|python3 scripts/train_fixmatch.py --config \"$FIXMATCH_CONFIG\" --labeled 500 --seed 42 --output_dir \"$OUTPUT_DIR/fixmatch_500_seed42\""
        "supervised_500_seed43|$OUTPUT_DIR/supervised_500_seed43|python3 scripts/train_supervised.py --config \"$SUPERVISED_CONFIG\" --labeled_subset 500 --seed 43 --output_dir \"$OUTPUT_DIR/supervised_500_seed43\""
        "fixmatch_500_seed43|$OUTPUT_DIR/fixmatch_500_seed43|python3 scripts/train_fixmatch.py --config \"$FIXMATCH_CONFIG\" --labeled 500 --seed 43 --output_dir \"$OUTPUT_DIR/fixmatch_500_seed43\""
        "supervised_500_seed44|$OUTPUT_DIR/supervised_500_seed44|python3 scripts/train_supervised.py --config \"$SUPERVISED_CONFIG\" --labeled_subset 500 --seed 44 --output_dir \"$OUTPUT_DIR/supervised_500_seed44\""
        "fixmatch_500_seed44|$OUTPUT_DIR/fixmatch_500_seed44|python3 scripts/train_fixmatch.py --config \"$FIXMATCH_CONFIG\" --labeled 500 --seed 44 --output_dir \"$OUTPUT_DIR/fixmatch_500_seed44\""
        "fixmatch_static_100_seed42|$OUTPUT_DIR/fixmatch_static_100_seed42|python3 scripts/train_fixmatch.py --config \"$FIXMATCH_STATIC_CONFIG\" --labeled 100 --seed 42 --output_dir \"$OUTPUT_DIR/fixmatch_static_100_seed42\""
        "fixmatch_static_100_seed43|$OUTPUT_DIR/fixmatch_static_100_seed43|python3 scripts/train_fixmatch.py --config \"$FIXMATCH_STATIC_CONFIG\" --labeled 100 --seed 43 --output_dir \"$OUTPUT_DIR/fixmatch_static_100_seed43\""
        "fixmatch_legacy_aug_100_seed42|$OUTPUT_DIR/fixmatch_legacy_aug_100_seed42|python3 scripts/train_fixmatch.py --config \"$FIXMATCH_LEGACY_AUG_CONFIG\" --labeled 100 --seed 42 --output_dir \"$OUTPUT_DIR/fixmatch_legacy_aug_100_seed42\""
        "fixmatch_legacy_aug_100_seed43|$OUTPUT_DIR/fixmatch_legacy_aug_100_seed43|python3 scripts/train_fixmatch.py --config \"$FIXMATCH_LEGACY_AUG_CONFIG\" --labeled 100 --seed 43 --output_dir \"$OUTPUT_DIR/fixmatch_legacy_aug_100_seed43\""
        "supervised_full_seed42|$OUTPUT_DIR/supervised_full_seed42|python3 scripts/train_supervised.py --config \"$SUPERVISED_CONFIG\" --seed 42 --output_dir \"$OUTPUT_DIR/supervised_full_seed42\""
        "supervised_full_seed43|$OUTPUT_DIR/supervised_full_seed43|python3 scripts/train_supervised.py --config \"$SUPERVISED_CONFIG\" --seed 43 --output_dir \"$OUTPUT_DIR/supervised_full_seed43\""
    )
    printf "%s\n" "${experiments[@]}"
}

build_fail_branch() {
    local experiments=(
        "mean_teacher_100_seed42|$OUTPUT_DIR/mean_teacher_100_seed42|python3 scripts/train_mean_teacher.py --config \"$MEAN_TEACHER_CONFIG\" --labeled 100 --seed 42 --output_dir \"$OUTPUT_DIR/mean_teacher_100_seed42\""
        "mean_teacher_100_seed43|$OUTPUT_DIR/mean_teacher_100_seed43|python3 scripts/train_mean_teacher.py --config \"$MEAN_TEACHER_CONFIG\" --labeled 100 --seed 43 --output_dir \"$OUTPUT_DIR/mean_teacher_100_seed43\""
        "mean_teacher_100_seed44|$OUTPUT_DIR/mean_teacher_100_seed44|python3 scripts/train_mean_teacher.py --config \"$MEAN_TEACHER_CONFIG\" --labeled 100 --seed 44 --output_dir \"$OUTPUT_DIR/mean_teacher_100_seed44\""
        "mean_teacher_250_seed42|$OUTPUT_DIR/mean_teacher_250_seed42|python3 scripts/train_mean_teacher.py --config \"$MEAN_TEACHER_CONFIG\" --labeled 250 --seed 42 --output_dir \"$OUTPUT_DIR/mean_teacher_250_seed42\""
        "mean_teacher_250_seed43|$OUTPUT_DIR/mean_teacher_250_seed43|python3 scripts/train_mean_teacher.py --config \"$MEAN_TEACHER_CONFIG\" --labeled 250 --seed 43 --output_dir \"$OUTPUT_DIR/mean_teacher_250_seed43\""
        "mean_teacher_250_seed44|$OUTPUT_DIR/mean_teacher_250_seed44|python3 scripts/train_mean_teacher.py --config \"$MEAN_TEACHER_CONFIG\" --labeled 250 --seed 44 --output_dir \"$OUTPUT_DIR/mean_teacher_250_seed44\""
        "supervised_500_seed42|$OUTPUT_DIR/supervised_500_seed42|python3 scripts/train_supervised.py --config \"$SUPERVISED_CONFIG\" --labeled_subset 500 --seed 42 --output_dir \"$OUTPUT_DIR/supervised_500_seed42\""
        "mean_teacher_500_seed42|$OUTPUT_DIR/mean_teacher_500_seed42|python3 scripts/train_mean_teacher.py --config \"$MEAN_TEACHER_CONFIG\" --labeled 500 --seed 42 --output_dir \"$OUTPUT_DIR/mean_teacher_500_seed42\""
        "supervised_500_seed43|$OUTPUT_DIR/supervised_500_seed43|python3 scripts/train_supervised.py --config \"$SUPERVISED_CONFIG\" --labeled_subset 500 --seed 43 --output_dir \"$OUTPUT_DIR/supervised_500_seed43\""
        "mean_teacher_500_seed43|$OUTPUT_DIR/mean_teacher_500_seed43|python3 scripts/train_mean_teacher.py --config \"$MEAN_TEACHER_CONFIG\" --labeled 500 --seed 43 --output_dir \"$OUTPUT_DIR/mean_teacher_500_seed43\""
    )
    printf "%s\n" "${experiments[@]}"
}

if [[ "$PROFILE" != "confirm_fixmatch" ]]; then
    log "ERROR: Unsupported profile '$PROFILE'"
    exit 1
fi

if [[ ! -d "$RESCUE_DIR" ]]; then
    log "ERROR: Rescue directory '$RESCUE_DIR' not found."
    exit 1
fi

log "========================================="
log "SSL Mammography - Follow-Up Overnight Run"
log "========================================="
log "Rescue directory: $RESCUE_DIR"
log "Output directory: $OUTPUT_DIR"
log "Profile:          $PROFILE"
log "Dry run:          $DRY_RUN"
log "Skip completed:   $SKIP_COMPLETED"
log "Retry failed:     $RETRY_FAILED"

RESCUE_SUMMARY_JSON=$(get_rescue_decision)
printf "%s\n" "$RESCUE_SUMMARY_JSON" > "$OUTPUT_DIR/rescue_summary.json"
RESCUE_DECISION=$(printf "%s\n" "$RESCUE_SUMMARY_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["promotion_gate"]["decision"])')
RESCUE_LABEL=$(printf "%s\n" "$RESCUE_SUMMARY_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["promotion_gate"]["decision_label"])')
log "Rescue decision: $RESCUE_LABEL"

if [[ "$RESCUE_DECISION" == "keep_fixmatch" ]]; then
    log "Branch selected: FixMatch confirmation"
    EXPERIMENTS="$(build_pass_branch)"
else
    log "Branch selected: Mean Teacher fallback"
    EXPERIMENTS="$(build_fail_branch)"
fi

while IFS= read -r entry; do
    [[ -z "$entry" ]] && continue
    IFS="|" read -r name exp_dir cmd <<< "$entry"
    run_experiment "$name" "$exp_dir" "$cmd"
done <<EOF
$EXPERIMENTS
EOF

log "=== FOLLOW-UP SUMMARY ==="
python3 scripts/summarize_followup.py --output_dir "$OUTPUT_DIR" --rescue_dir "$RESCUE_DIR" | tee -a "$LOG_FILE"

if [[ "${#FAILED_EXPERIMENTS[@]}" -gt 0 ]]; then
    log "Failed experiments:"
    for name in "${FAILED_EXPERIMENTS[@]}"; do
        log "  - $name"
    done
else
    log "All scheduled follow-up experiments completed or were skipped."
fi
