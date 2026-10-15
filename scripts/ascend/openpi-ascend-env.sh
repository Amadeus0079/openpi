#!/usr/bin/env bash
# OpenPI Pi0.5 training environment for this CANN 9.0 container.
# Source this file from bash or zsh: source scripts/ascend/openpi-ascend-env.sh

export OPENPI_WORKSPACE="/inspire/sj-ssd3/project/robot-reasoning/public/zichen/workplace"
export OPENPI_ROOT="$OPENPI_WORKSPACE/openpi"
export OPENPI_VENV="$OPENPI_ROOT/.venv"
export OPENPI_MODEL_ROOT="/inspire/sj-ssd3/project/robot-reasoning/public/models/pi05_openpi"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$OPENPI_WORKSPACE/../.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$UV_CACHE_DIR/python}"
export UV_LINK_MODE=copy
export UV_CONFIG_FILE="$OPENPI_ROOT/scripts/ascend/uv.toml"

# CANN supplies HCCL and the Ascend runtime needed by torch_npu.
source "$OPENPI_WORKSPACE/../scripts/ascend_env.sh"
if [[ ! -f "$OPENPI_VENV/bin/activate" ]]; then
    echo "OpenPI .venv missing; run scripts/ascend/setup_uv.sh first." >&2
    return 1
fi
source "$OPENPI_VENV/bin/activate"

export OPENPI_DATA_HOME="$OPENPI_MODEL_ROOT"
export HF_HOME="$OPENPI_MODEL_ROOT/hf_cache"
export PALIGEMMA_TOKENIZER_PATH="$OPENPI_MODEL_ROOT/paligemma_tokenizer/tokenizer.model"
export PYTORCH_WEIGHT_PATH="$OPENPI_MODEL_ROOT/pi05_base_pytorch"

# Keep the local project source first.
export PYTHONPATH="$OPENPI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
unset TORCH_DEVICE_BACKEND_AUTOLOAD
