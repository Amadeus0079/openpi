#!/usr/bin/env bash
set -euo pipefail
OPENPI_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$OPENPI_ROOT/scripts/ascend/openpi-ascend-env.sh"
cd "$OPENPI_ROOT"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export JAX_PLATFORMS=cpu
export TOKENIZERS_PARALLELISM=false
exec torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/train_pytorch_fsdp.py pi05_yam \
    --exp-name "${EXP_NAME:-yam_pour_water_fsdp_5k}" "$@"
