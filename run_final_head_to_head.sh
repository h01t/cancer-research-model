#!/bin/bash
# Run the final Adam vs AdamW head-to-head for the label-smoothing candidate,
# then launch clinical-style follow-up on the resulting runs.

set -u

OUTPUT_DIR="results_supervised_final_head_to_head"
LABELED_SUBSET="500"
DEVICE=""
SKIP_COMPLETED=1
MAX_EPOCHS=""
BOOTSTRAP_SAMPLES="200"
SEEDS=(45 46 47)
CONFIG_ENTRIES=(
  "default_nofreeze_ls_adam|configs/default_nofreeze_ls_adam.yaml"
  "default_nofreeze_ls_adamw|configs/default_nofreeze_ls_adamw.yaml"
)

usage() {
  cat <<EOF
Usage: ./run_final_head_to_head.sh [options]

Options:
  -o, --output DIR         Output directory (default: results_supervised_final_head_to_head)
  --labeled N              Labeled subset size (default: 500)
  --seeds A,B,C            Comma-separated seed list (default: 45,46,47)
  --device DEVICE          Device override passed to train/eval scripts
  --bootstrap_samples N    Bootstrap samples for clinical evaluation (default: 200)
  --max_epochs N           Optional max-epochs override for quick probes
  --skip_completed         Skip runs with test_metrics.yaml/clinical_summary.yaml present (default)
  --no-skip_completed      Re-run completed experiments
  -h, --help               Show this help
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
    --seeds)
      IFS=',' read -r -a SEEDS <<< "$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --bootstrap_samples)
      BOOTSTRAP_SAMPLES="$2"
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
LOG_FILE="$OUTPUT_DIR/final_head_to_head.log"

run_one() {
  local run_label="$1"
  local config_path="$2"
  local seed="$3"
  local run_name="${run_label}_seed${seed}"
  local run_dir="$OUTPUT_DIR/$run_name"

  if [[ "$SKIP_COMPLETED" -eq 1 && -f "$run_dir/test_metrics.yaml" ]]; then
    echo "[skip-train] $run_name" | tee -a "$LOG_FILE"
    return 0
  fi

  mkdir -p "$run_dir"
  echo "[train] $run_name" | tee -a "$LOG_FILE"

  local cmd=(
    python3 scripts/train_supervised.py
    --config "$config_path"
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

for entry in "${CONFIG_ENTRIES[@]}"; do
  IFS='|' read -r run_label config_path <<< "$entry"
  for seed in "${SEEDS[@]}"; do
    run_one "$run_label" "$config_path" "$seed"
  done
done

python3 scripts/summarize_supervised.py --results_dir "$OUTPUT_DIR" --output_dir "$OUTPUT_DIR" >>"$LOG_FILE" 2>&1

SEEDS_CSV=""
for seed in "${SEEDS[@]}"; do
  if [[ -n "$SEEDS_CSV" ]]; then
    SEEDS_CSV+=","
  fi
  SEEDS_CSV+="$seed"
done

CLINICAL_CMD=(
  ./run_clinical_candidate_eval.sh
  --results_dir "$OUTPUT_DIR"
  -o "$OUTPUT_DIR/clinical_candidates"
  --config default_nofreeze_ls_adam
  --config default_nofreeze_ls_adamw
  --seeds "$SEEDS_CSV"
  --bootstrap_samples "$BOOTSTRAP_SAMPLES"
)
if [[ -n "$DEVICE" ]]; then
  CLINICAL_CMD+=(--device "$DEVICE")
fi
if [[ "$SKIP_COMPLETED" -eq 0 ]]; then
  CLINICAL_CMD+=(--no-skip_completed)
fi

echo "[clinical] default_nofreeze_ls_adam vs default_nofreeze_ls_adamw" | tee -a "$LOG_FILE"
"${CLINICAL_CMD[@]}" >>"$LOG_FILE" 2>&1

echo "Final head-to-head complete."
echo "Training summary: $OUTPUT_DIR/supervised_summary.txt"
echo "Clinical summary: $OUTPUT_DIR/clinical_candidates/clinical_candidates_summary.txt"
