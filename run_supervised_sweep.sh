#!/bin/bash
# Run supervised sweeps across active modeling directions.

set -u

OUTPUT_DIR="results_supervised_sweep"
LABELED_SUBSET="500"
DEVICE=""
PROFILE="candidate"
SKIP_COMPLETED=1
MAX_EPOCHS=""
SEEDS=(42 43 44)
OPTIMIZER_BASE_CONFIG=""

usage() {
  cat <<EOF
Usage: ./run_supervised_sweep.sh [options]

Options:
  -o, --output DIR            Output directory (default: results_supervised_sweep)
  --labeled N                Labeled subset size (default: 500)
  --device DEVICE            Device override passed to train_supervised.py
  --profile NAME             Sweep profile:
                             candidate (default), optimizer_recovery,
                             resolution, backbone, optimizer, regularization, all
  --optimizer_base_config X  Base config name for optimizer_recovery
                             (for example: default_nofreeze_aug_safe_ls)
  --max_epochs N             Optional max-epochs override for quick probes
  --skip_completed           Skip runs with test_metrics.yaml already present (default)
  --no-skip_completed        Re-run completed experiments
  -h, --help                 Show this help
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
    --optimizer_base_config)
      OPTIMIZER_BASE_CONFIG="$2"
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

declare -a RUN_ENTRIES=()
TEMP_CONFIG_DIR=""

build_optimizer_override_config() {
  local base_config_path="$1"
  local output_config_path="$2"
  local optimizer_name="$3"

  python3 - "$base_config_path" "$output_config_path" "$optimizer_name" <<'PY'
import sys
import yaml

base_path, output_path, optimizer_name = sys.argv[1:4]
with open(base_path, "r") as f:
    config = yaml.safe_load(f)

training = config["training"]
if optimizer_name == "adam":
    pass
elif optimizer_name == "adamw":
    training["optimizer"] = "adamw"
    training["learning_rate"] = 0.0003
    training["weight_decay"] = 0.0005
elif optimizer_name == "sgd":
    training["optimizer"] = "sgd"
    training["learning_rate"] = 0.01
    training["weight_decay"] = 0.0001
    training["momentum"] = 0.9
else:
    raise ValueError(f"Unsupported optimizer override: {optimizer_name}")

with open(output_path, "w") as f:
    yaml.safe_dump(config, f, sort_keys=False)
PY
}

cleanup() {
  if [[ -n "$TEMP_CONFIG_DIR" && -d "$TEMP_CONFIG_DIR" ]]; then
    rm -rf "$TEMP_CONFIG_DIR"
  fi
}

trap cleanup EXIT

case "$PROFILE" in
  candidate)
    RUN_ENTRIES=(
      "default_nofreeze_res512|configs/default_nofreeze_res512.yaml"
      "default_nofreeze_aug_safe|configs/default_nofreeze_aug_safe.yaml"
      "default_nofreeze_ls|configs/default_nofreeze_ls.yaml"
      "default_nofreeze_aug_safe_ls|configs/default_nofreeze_aug_safe_ls.yaml"
    )
    ;;
  optimizer_recovery)
    if [[ -z "$OPTIMIZER_BASE_CONFIG" ]]; then
      echo "optimizer_recovery requires --optimizer_base_config"
      exit 1
    fi
    TEMP_CONFIG_DIR=$(mktemp -d)
    BASE_CONFIG_PATH="configs/${OPTIMIZER_BASE_CONFIG}.yaml"
    if [[ ! -f "$BASE_CONFIG_PATH" ]]; then
      echo "Base config not found: $BASE_CONFIG_PATH"
      exit 1
    fi
    build_optimizer_override_config "$BASE_CONFIG_PATH" "$TEMP_CONFIG_DIR/${OPTIMIZER_BASE_CONFIG}_adam.yaml" "adam"
    build_optimizer_override_config "$BASE_CONFIG_PATH" "$TEMP_CONFIG_DIR/${OPTIMIZER_BASE_CONFIG}_adamw.yaml" "adamw"
    build_optimizer_override_config "$BASE_CONFIG_PATH" "$TEMP_CONFIG_DIR/${OPTIMIZER_BASE_CONFIG}_sgd.yaml" "sgd"
    RUN_ENTRIES=(
      "${OPTIMIZER_BASE_CONFIG}_adam|$TEMP_CONFIG_DIR/${OPTIMIZER_BASE_CONFIG}_adam.yaml"
      "${OPTIMIZER_BASE_CONFIG}_adamw|$TEMP_CONFIG_DIR/${OPTIMIZER_BASE_CONFIG}_adamw.yaml"
      "${OPTIMIZER_BASE_CONFIG}_sgd|$TEMP_CONFIG_DIR/${OPTIMIZER_BASE_CONFIG}_sgd.yaml"
    )
    ;;
  resolution)
    RUN_ENTRIES=(
      "default_nofreeze_res512|configs/default_nofreeze_res512.yaml"
      "default_nofreeze_res768|configs/default_nofreeze_res768.yaml"
      "default_nofreeze_res1024|configs/default_nofreeze_res1024.yaml"
    )
    ;;
  backbone)
    RUN_ENTRIES=(
      "default_nofreeze_res512|configs/default_nofreeze_res512.yaml"
      "default_nofreeze_b2|configs/default_nofreeze_b2.yaml"
      "default_nofreeze_b3|configs/default_nofreeze_b3.yaml"
    )
    ;;
  optimizer)
    RUN_ENTRIES=(
      "default_nofreeze_res512|configs/default_nofreeze_res512.yaml"
      "default_nofreeze_adamw|configs/default_nofreeze_adamw.yaml"
      "default_nofreeze_sgd|configs/default_nofreeze_sgd.yaml"
    )
    ;;
  regularization)
    RUN_ENTRIES=(
      "default_nofreeze_res512|configs/default_nofreeze_res512.yaml"
      "default_nofreeze_ls|configs/default_nofreeze_ls.yaml"
      "default_nofreeze_aug_safe|configs/default_nofreeze_aug_safe.yaml"
      "default_nofreeze_aug_safe_ls|configs/default_nofreeze_aug_safe_ls.yaml"
    )
    ;;
  all)
    RUN_ENTRIES=(
      "default_nofreeze_res512|configs/default_nofreeze_res512.yaml"
      "default_nofreeze_res768|configs/default_nofreeze_res768.yaml"
      "default_nofreeze_res1024|configs/default_nofreeze_res1024.yaml"
      "default_nofreeze_b2|configs/default_nofreeze_b2.yaml"
      "default_nofreeze_b3|configs/default_nofreeze_b3.yaml"
      "default_nofreeze_adamw|configs/default_nofreeze_adamw.yaml"
      "default_nofreeze_sgd|configs/default_nofreeze_sgd.yaml"
      "default_nofreeze_ls|configs/default_nofreeze_ls.yaml"
      "default_nofreeze_aug_safe|configs/default_nofreeze_aug_safe.yaml"
      "default_nofreeze_aug_safe_ls|configs/default_nofreeze_aug_safe_ls.yaml"
    )
    ;;
  *)
    echo "Unknown profile: $PROFILE"
    exit 1
    ;;
esac

run_one() {
  local run_label="$1"
  local config_path="$2"
  local seed="$3"
  local run_name="${run_label}_seed${seed}"
  local run_dir="$OUTPUT_DIR/$run_name"

  if [[ "$SKIP_COMPLETED" -eq 1 && -f "$run_dir/test_metrics.yaml" ]]; then
    echo "[skip] $run_name" | tee -a "$LOG_FILE"
    return 0
  fi

  mkdir -p "$run_dir"
  echo "[run] $run_name" | tee -a "$LOG_FILE"

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

for entry in "${RUN_ENTRIES[@]}"; do
  IFS='|' read -r run_label config_path <<< "$entry"
  for seed in "${SEEDS[@]}"; do
    run_one "$run_label" "$config_path" "$seed"
  done
done

python3 scripts/summarize_supervised.py --results_dir "$OUTPUT_DIR" --output_dir "$OUTPUT_DIR" >>"$LOG_FILE" 2>&1
echo "Sweep complete. Summary written to $OUTPUT_DIR/supervised_summary.txt"
