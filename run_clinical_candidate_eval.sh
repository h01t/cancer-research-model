#!/bin/bash
# Run clinical_eval.py across candidate supervised runs and summarize them.

set -u

RESULTS_DIR="results_supervised_candidates"
OUTPUT_DIR="results_supervised_candidates/clinical_candidates"
BOOTSTRAP_SAMPLES="200"
DEVICE=""
SKIP_COMPLETED=1
CUSTOM_CONFIGS=0
DRY_RUN=0
CONFIGS=(
  "default_nofreeze_res512"
  "default_nofreeze_aug_safe"
  "default_nofreeze_ls"
)
SEEDS=(42 43 44)

usage() {
  cat <<EOF
Usage: ./run_clinical_candidate_eval.sh [options]

Options:
  --results_dir DIR        Directory containing supervised run folders
                           (default: results_supervised_candidates)
  -o, --output DIR         Output directory for clinical bundles
                           (default: results_supervised_candidates/clinical_candidates)
  --bootstrap_samples N    Number of bootstrap samples for clinical_eval.py
  --device DEVICE          Optional device override
  --config NAME            Restrict evaluation to a specific config name
                           (repeatable; defaults to the promoted top 3 candidates)
  --seeds A,B,C            Comma-separated seed list (default: 42,43,44)
  --dry_run                Show which runs would be evaluated and exit
  --skip_completed         Skip runs with clinical_summary.yaml already present (default)
  --no-skip_completed      Re-run completed clinical bundles
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
    --seeds)
      IFS=',' read -r -a SEEDS <<< "$2"
      shift 2
      ;;
    --config)
      if [[ "$CUSTOM_CONFIGS" -eq 0 ]]; then
        CONFIGS=()
        CUSTOM_CONFIGS=1
      fi
      CONFIGS+=("$2")
      shift 2
      ;;
    --dry_run)
      DRY_RUN=1
      shift
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

discover_configs() {
  find "$RESULTS_DIR" -maxdepth 1 -mindepth 1 -type d -name '*_seed*' \
    | sed 's#.*/##' \
    | sed -E 's/_seed[0-9]+$//' \
    | sort -u
}

if [[ "$CUSTOM_CONFIGS" -eq 0 ]]; then
  DEFAULT_MATCHES=0
  for config_name in "${CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      if [[ -d "$RESULTS_DIR/${config_name}_seed${seed}" ]]; then
        DEFAULT_MATCHES=1
        break 2
      fi
    done
  done

  if [[ "$DEFAULT_MATCHES" -eq 0 ]]; then
    DISCOVERED_CONFIGS=()
    while IFS= read -r discovered; do
      if [[ -n "$discovered" ]]; then
        DISCOVERED_CONFIGS+=("$discovered")
      fi
    done < <(discover_configs)
    if [[ "${#DISCOVERED_CONFIGS[@]}" -gt 0 ]]; then
      CONFIGS=("${DISCOVERED_CONFIGS[@]}")
    fi
  fi
fi

mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/clinical_eval.log"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Clinical evaluation configs:"
  for config_name in "${CONFIGS[@]}"; do
    echo "  $config_name"
    for seed in "${SEEDS[@]}"; do
      run_name="${config_name}_seed${seed}"
      run_dir="$RESULTS_DIR/$run_name"
      if [[ -d "$run_dir" ]]; then
        echo "    [present] $run_name"
      else
        echo "    [missing] $run_name"
      fi
    done
  done
  exit 0
fi

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
    if [[ "$SKIP_COMPLETED" -eq 1 && -f "$out_dir/clinical_summary.yaml" ]]; then
      echo "[skip] $run_name" | tee -a "$LOG_FILE"
      continue
    fi

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
