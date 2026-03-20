#!/bin/bash
# Run a supervised-only sweep across active modeling directions.

set -u

OUTPUT_DIR="results_supervised_sweep"
LABELED_SUBSET="500"
DEVICE=""
PROFILE="all"
SKIP_COMPLETED=1
MAX_EPOCHS=""
SEEDS=(42 43 44)

usage() {
  cat <<EOF
Usage: ./run_supervised_sweep.sh [options]

Options:
  -o, --output DIR         Output directory (default: results_supervised_sweep)
  --labeled N             Labeled subset size (default: 500)
  --device DEVICE         Device override passed to train_supervised.py
  --profile NAME          Sweep profile: all, resolution, backbone, optimizer, regularization
  --max_epochs N          Optional max-epochs override for quick probes
  --skip_completed        Skip runs with test_metrics.yaml already present (default)
  --no-skip_completed     Re-run completed experiments
  -h, --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --labeled)
      LABELED_SUBSET="$2"
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
    --skip_completed)
      SKIP_COMPLETED=1
      shift
      ;;
    --no-skip_completed)
      SKIP_COMPLETED=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/supervised_sweep.log"

declare -a CONFIGS=()
case "$PROFILE" in
  all)
    CONFIGS=(
      "default_nofreeze_res512"
      "default_nofreeze_res768"
      "default_nofreeze_res1024"
      "default_nofreeze_b2"
      "default_nofreeze_b3"
      "default_nofreeze_adamw"
      "default_nofreeze_sgd"
      "default_nofreeze_ls"
      "default_nofreeze_aug_safe"
    )
    ;;
  resolution)
    CONFIGS=("default_nofreeze_res512" "default_nofreeze_res768" "default_nofreeze_res1024")
    ;;
  backbone)
    CONFIGS=("default_nofreeze_res512" "default_nofreeze_b2" "default_nofreeze_b3")
    ;;
  optimizer)
    CONFIGS=("default_nofreeze_res512" "default_nofreeze_adamw" "default_nofreeze_sgd")
    ;;
  regularization)
    CONFIGS=("default_nofreeze_res512" "default_nofreeze_ls" "default_nofreeze_aug_safe")
    ;;
  *)
    echo "Unknown profile: $PROFILE"
    exit 1
    ;;
esac

run_one() {
  local config_name="$1"
  local seed="$2"
  local run_name="${config_name}_seed${seed}"
  local run_dir="$OUTPUT_DIR/$run_name"

  if [[ "$SKIP_COMPLETED" -eq 1 && -f "$run_dir/test_metrics.yaml" ]]; then
    echo "[skip] $run_name" | tee -a "$LOG_FILE"
    return 0
  fi

  mkdir -p "$run_dir"
  echo "[run] $run_name" | tee -a "$LOG_FILE"

  local cmd=(
    python3 scripts/train_supervised.py
    --config "configs/${config_name}.yaml"
    --labeled_subset "$LABELED_SUBSET"
    --seed "$seed"
    --output_dir "$run_dir"
  )

  if [[ -n "$DEVICE" ]]; then
    cmd+=(--device "$DEVICE")
  fi
  if [[ -n "$MAX_EPOCHS" ]]; then
    cmd+=(--max_epochs "$MAX_EPOCHS")
  fi

  "${cmd[@]}" >>"$LOG_FILE" 2>&1
}

for config_name in "${CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_one "$config_name" "$seed"
  done
done

python3 scripts/summarize_supervised.py --results_dir "$OUTPUT_DIR" --output_dir "$OUTPUT_DIR" >>"$LOG_FILE" 2>&1
echo "Sweep complete. Summary written to $OUTPUT_DIR/supervised_summary.txt"
