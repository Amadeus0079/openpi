#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/openpi-ascend-env.sh"
cd "$OPENPI_ROOT"

if [[ -z "${LIBERO_DATASET_REPO_ID:-}" ]]; then
  echo "Set LIBERO_DATASET_REPO_ID to the local LIBERO LeRobot dataset before training." >&2
  exit 2
fi
if [[ ! -f "$PYTORCH_WEIGHT_PATH/model.safetensors" && ! -d "$PYTORCH_WEIGHT_PATH" ]]; then
  echo "Converted PyTorch weights missing; run scripts/ascend/prepare_pi05_weights.sh first." >&2
  exit 2
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
EXP_NAME="${EXP_NAME:-pi05_libero_npu}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"

torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
  scripts/train_pytorch.py pi05_libero \
  --exp_name "$EXP_NAME" \
  --save_interval "$SAVE_INTERVAL" \
  --batch_size "$BATCH_SIZE"
