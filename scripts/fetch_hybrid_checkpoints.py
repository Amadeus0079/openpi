"""Download inference-only checkpoint publications via OBS and verify SHA256."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

PREFIX = "obs://sai.liyl/zichen/openpi-hybrid-checkpoints/hybrid-three-20261018/"
NAMES = ("yam_only", "yam_source_top10", "yam_global_top80")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(root, prefix=PREFIX, names=NAMES):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    logs = root / ".transfer-logs"
    logs.mkdir(exist_ok=True)
    for name in names:
        stage = root / ("." + name + "-10000.download")
        stage.mkdir(exist_ok=True)
        object_root = prefix.rstrip("/") + f"/{name}/10000/"

        def download(key, destination, object_root=object_root):
            destination.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "obsutil",
                    "cp",
                    object_root + key,
                    str(destination),
                    "-f",
                    "-p=4",
                    "-o=" + str(logs),
                    "-cpd=" + str(logs / "parts"),
                ],
                check=True,
            )

        download("inference-manifest.json", stage / "inference-manifest.json")
        manifest = json.loads((stage / "inference-manifest.json").read_text())
        if manifest["dataset"] != name or manifest["step"] != 10000:
            raise ValueError("Publication name/step mismatch")
        final = root / name / "10000"
        if final.exists():
            if (final / "inference-manifest.json").read_bytes() != (stage / "inference-manifest.json").read_bytes():
                raise FileExistsError(f"Refusing to replace existing checkpoint: {final}")
            directory = final
        else:
            directory = stage
        for key, record in manifest["files"].items():
            target = directory / key
            if not target.resolve().is_relative_to(directory.resolve()):
                raise ValueError("Unsafe publication path")
            valid = target.is_file() and target.stat().st_size == record["size"] and sha(target) == record["sha256"]
            if not valid:
                if directory == final:
                    raise ValueError(f"Existing checkpoint is corrupted: {target}")
                download(key, target)
                if target.stat().st_size != record["size"] or sha(target) != record["sha256"]:
                    raise ValueError(f"Checksum mismatch: {key}")
        if directory != final:
            final.parent.mkdir(exist_ok=True)
            stage.rename(final)
        print(json.dumps({"dataset": name, "path": str(final), "verified": True}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/nas/data/yam-ckpts/hybrid_pi05")
    parser.add_argument("--prefix", default=PREFIX)
    parser.add_argument("--dataset", choices=NAMES, action="append")
    args = parser.parse_args()
    fetch(args.root, args.prefix, args.dataset or NAMES)
