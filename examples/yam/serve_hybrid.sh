#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
dataset="${1:-yam_only}"
port="${2:-8001}"
case "$dataset" in yam_only|yam_source_top10|yam_global_top80) ;; *) echo "Unknown dataset: $dataset" >&2; exit 2;; esac
checkpoint="/nas/data/yam-ckpts/hybrid_pi05/$dataset/10000"
if [[ ! -f "$checkpoint/inference-manifest.json" ]]; then
    echo "Verified checkpoint not available: $checkpoint" >&2
    echo "Publish via OBS, then run scripts/fetch_hybrid_checkpoints.py." >&2
    exit 2
fi
cd "$root"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
exec "$root/.venv/bin/python" scripts/serve_policy.py --host 127.0.0.1 \
    --pytorch-device cuda:0 --port "$port" policy:checkpoint \
    --policy.config pi05_yam_hybrid --policy.dir "$checkpoint"
