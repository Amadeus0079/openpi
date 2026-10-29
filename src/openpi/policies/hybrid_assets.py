"""Verify an inference publication before loading its portable checkpoint."""

import dataclasses
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(root, model_config):
    root = Path(root).resolve()
    manifest = json.loads((root / "inference-manifest.json").read_text())
    step = manifest["step"]
    if type(step) is not int or step <= 0:
        raise ValueError("Invalid published checkpoint step")
    required = {
        "model.safetensors",
        "run.json",
        "complete.json",
        "assets/yam_hybrid/deployment.json",
        "assets/yam_hybrid/norm_stats.json",
        "assets/yam_hybrid/config.json",
        "assets/yam_hybrid/deployment/tokenizer.model",
    }
    if not required.issubset(manifest["files"]):
        raise ValueError("Inference manifest omits required files")
    for name, record in manifest["files"].items():
        path = root / name
        if not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing/unsafe checkpoint file: {name}")
        if path.stat().st_size != record["size"] or sha(path) != record["sha256"]:
            raise ValueError(f"Checkpoint checksum mismatch: {name}")
    assets = root / "assets/yam_hybrid"
    contract = json.loads((assets / "deployment.json").read_text())
    if (
        contract["action_version"],
        contract["state_dim"],
        contract["action_dim"],
        contract["horizon"],
        contract["fps"],
        contract["ee_definition"],
    ) != ("camera_anchor_ee_v1", 22, 14, 30, 30, "gripper_mount"):
        raise ValueError("Unsupported hybrid deployment contract")
    if contract["dataset_identity"] != manifest["dataset_identity"]:
        raise ValueError("Manifest dataset identity mismatch")
    if sha(assets / "norm_stats.json") != contract["norm_stats_sha256"]:
        raise ValueError("Normalization is not bound to the deployment contract")
    run = json.loads((root / "run.json").read_text())
    if run["step"] != step:
        raise ValueError("Run step differs from the published checkpoint")
    complete = json.loads((root / "complete.json").read_text())
    if complete["step"] != step:
        raise ValueError("Complete marker step differs from the published checkpoint")
    if run["hybrid"]["deployment"] != contract:
        raise ValueError("Run deployment contract mismatch")
    expected = json.loads((assets / "config.json").read_text())["model"]
    current = json.loads(json.dumps(dataclasses.asdict(model_config), default=str))
    if current != expected:
        raise ValueError("Requested model configuration differs from the trained model")
    os.environ["PALIGEMMA_TOKENIZER_PATH"] = str(assets / "deployment/tokenizer.model")
    return {
        **contract,
        "checkpoint": str(root),
        "action_horizon": contract["horizon"],
        "model_sha256": manifest["files"]["model.safetensors"]["sha256"],
    }
