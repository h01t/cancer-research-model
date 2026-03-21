#!/bin/bash
# Run clinical_eval.py across candidate supervised runs and summarize them.

set -u

RESULTS_DIR="results_supervised_sweep"
OUTPUT_DIR="results_supervised_sweep/clinical_candidates"
BOOTSTRAP_SAMPLES="200"
DEVICE=""
CONFIGS=(
  "default_nofreeze_res512"
  "default_nofreeze_aug_safe"
  "default_nofreeze_ls"
  "default_nofreeze_aug_safe_ls"
)
SEEDS=(42 43 44)

usage() {
  cat <<EOF
Usage: ./run_clinical_candidate_eval.sh [options]

Options:
  --results_dir DIR        Directory containing supervised run folders
                           (default: results_supervised_sweep)
  -o, --output DIR         Output directory for clinical bundles
                           (default: results_supervised_sweep/clinical_candidates)
  --bootstrap_samples N    Number of bootstrap samples for clinical_eval.py
  --device DEVICE          Optional device override
  -h, --help               Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --results_dir)
      RESULTS_DIR="$2"
      shift 2
      ;;
    -o|--output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --bootstrap_samples)
      BOOTSTRAP_SAMPLES="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
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
LOG_FILE="$OUTPUT_DIR/clinical_eval.log"

for config_name in "${CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_name="${config_name}_seed${seed}"
    run_dir="$RESULTS_DIR/$run_name"
    out_dir="$OUTPUT_DIR/$run_name"

    if [[ ! -d "$run_dir" ]]; then
      echo "[missing] $run_name" | tee -a "$LOG_FILE"
      continue
    fi

    echo "[run] $run_name" | tee -a "$LOG_FILE"
    cmd=(
      python3 scripts/clinical_eval.py
      --run_dir "$run_dir"
      --output_dir "$out_dir"
      --bootstrap_samples "$BOOTSTRAP_SAMPLES"
    )
    if [[ -n "$DEVICE" ]]; then
      cmd+=(--device "$DEVICE")
    fi
    "${cmd[@]}" >>"$LOG_FILE" 2>&1
  done
done

python3 scripts/summarize_clinical_candidates.py --results_dir "$OUTPUT_DIR" --output_dir "$OUTPUT_DIR" >>"$LOG_FILE" 2>&1
echo "Clinical candidate evaluation complete. Summary written to $OUTPUT_DIR/clinical_candidates_summary.txt"
