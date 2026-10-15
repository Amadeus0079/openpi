#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/openpi-ascend-env.sh"
cd "$OPENPI_ROOT"

if [[ ! -f "$PALIGEMMA_TOKENIZER_PATH" ]]; then
  echo "Tokenizer missing: $PALIGEMMA_TOKENIZER_PATH" >&2
  exit 1
fi

JAX_CHECKPOINT="$OPENPI_MODEL_ROOT/openpi-assets/checkpoints/pi05_base"
if [[ ! -f "$JAX_CHECKPOINT/params/commit_success.txt" ]]; then
  echo "Original Pi0.5 checkpoint is incomplete: $JAX_CHECKPOINT/params" >&2
  exit 1
fi

# Conversion deliberately stays on CPU: this container currently has no NPU
# device node attached, and conversion does not benefit from Ascend execution.
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python examples/convert_jax_model_to_pytorch.py \
  --checkpoint_dir "$JAX_CHECKPOINT" \
  --config_name pi05_libero \
  --output_path "$PYTORCH_WEIGHT_PATH"
