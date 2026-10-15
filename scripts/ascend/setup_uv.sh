#!/usr/bin/env bash
set -eo pipefail

OPENPI_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$OPENPI_ROOT"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$OPENPI_ROOT/../../.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$UV_CACHE_DIR/python}"
# Transformers is patched below; copies prevent edits to uv's shared cache.
export UV_LINK_MODE=copy
export UV_CONFIG_FILE="$OPENPI_ROOT/scripts/ascend/uv.toml"
uv sync --managed-python --locked --no-dev

.venv/bin/python - <<'PY'
from importlib.metadata import distribution, version
from pathlib import Path
import shutil

assert version("transformers") == "4.53.2"
source = Path("src/openpi/models_pytorch/transformers_replace")
target = Path(distribution("transformers").locate_file("transformers"))
for path in source.rglob("*"):
    if path.is_file() and "__pycache__" not in path.parts:
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # uv may have created hardlinks during an earlier installation.
        # Replace the destination inode before applying the patch.
        if destination.exists():
            destination.unlink()
        shutil.copy2(path, destination)
print(f"Applied OpenPI Transformers patches to {target}")
PY

echo "OpenPI ready: source scripts/ascend/openpi-ascend-env.sh"
