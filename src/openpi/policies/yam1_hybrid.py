"""Explicit YAM-1 v3 serving contract using the deployed hybrid model and IO."""

import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch

from openpi import transforms
from openpi.models_pytorch import pi0_pytorch
from openpi.policies.policy import Policy
from openpi.shared import normalize
from openpi.training.config import get_config
from openpi.training.pytorch_weights import load_fp32_weights

TARGET_ID = "e5ee832e9d5566f8d7a06a17db9e03090422b2136cc8030bcc0d0757e69f45dc"
AXES = [[0.0, 0.0, -1.0, 0.0], [0.0, -1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root, config=None):
    root = Path(root).resolve()
    manifest = json.loads((root / "inference-manifest.json").read_text())
    if manifest.get("schema") != "yam1-hybrid-inference-v1":
        raise ValueError("Not a YAM-1 publication")
    if not isinstance(manifest.get("group"), str) or type(manifest.get("step")) is not int or manifest["step"] <= 0:
        raise ValueError("Wrong experiment/step")
    if manifest["target_identity"] != TARGET_ID or manifest["native_T_policy_ee"] != AXES:
        raise ValueError("Wrong target or axes")
    for name, record in manifest["files"].items():
        path = root / name
        if not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError("Missing/unsafe file " + name)
        if path.stat().st_size != record["size"] or sha(path) != record["sha256"]:
            raise ValueError("Checksum mismatch " + name)
    assets = root / "assets/yam_hybrid"
    run = json.loads((root / "run.json").read_text())
    complete = json.loads((root / "complete.json").read_text())
    if run["step"] != complete["step"] or run["step"] != manifest["step"]:
        raise ValueError("Step mismatch")
    if run["alignment"]["name"] != manifest["group"] or run["alignment"]["architecture"] != manifest["architecture"]:
        raise ValueError("Architecture mismatch")
    training = run["hybrid"]["deployment"]
    if training["inputs"]["target"]["identity"] != TARGET_ID:
        raise ValueError("Training target mismatch")
    if sha(assets / "norm_stats.json") != training["norm_stats_sha256"]:
        raise ValueError("Frozen normalization mismatch")
    if sha(assets / "deployment/tokenizer.model") != training["tokenizer_sha256"]:
        raise ValueError("Tokenizer mismatch")
    geometry = assets / "deployment/target/geometry"
    axis = json.loads((geometry / "axis-contract.json").read_text())
    if axis["status"] != "user_approved" or axis["native_T_policy_ee"] != AXES:
        raise ValueError("Unapproved tool axes")
    published = json.loads((assets / "config.json").read_text())["model"]
    config = config or get_config("pi05_yam_hybrid")
    if dataclasses.asdict(config.model) != published:
        raise ValueError("Requested model configuration differs from training")
    for key, file in [("pi0", "pi0_pytorch.py"), ("gemma", "gemma_pytorch.py"), ("heads", "domain_heads.py")]:
        if sha(root / "reference" / file) != run["alignment"]["implementation_sha256"][key]:
            raise ValueError("Training source mismatch " + key)
    if sha(Path(pi0_pytorch.__file__)) != run["alignment"]["implementation_sha256"]["pi0"]:
        raise ValueError("Hybrid model differs from trained PI0")
    os.environ["PALIGEMMA_TOKENIZER_PATH"] = str(assets / "deployment/tokenizer.model")
    metadata = {
        "action_version": "camera_anchor_ee_yam1_v3",
        "state_dim": 22,
        "action_dim": 14,
        "horizon": 30,
        "action_horizon": 30,
        "fps": 30,
        "ee_definition": "gripper_mount",
        "ee_mapping_id": axis["mapping_id"],
        "native_T_policy_ee": AXES,
        "dataset_identity": TARGET_ID,
        "target_identity": TARGET_ID,
        "model_sha256": manifest["files"]["model.safetensors"]["sha256"],
        "group": manifest["group"],
        "step": manifest["step"],
        "architecture": manifest["architecture"],
        "checkpoint": str(root),
        "hardware_access": False,
    }
    return config, manifest, metadata


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preprocessing(root, config):
    assets = Path(root) / "assets/yam_hybrid"
    data = config.data.create(config.assets_dirs, config.model)
    stats = normalize.load(assets)
    pipeline = transforms.compose(
        [*data.data_transforms.inputs, transforms.Normalize(stats, use_quantiles=True), *data.model_transforms.inputs]
    )
    rows = []
    for task in ["cup", "tumbler"]:
        with np.load(Path(root) / f"validation/replay-{task}.npz", allow_pickle=False) as replay:
            obs = {
                "state": replay["state"],
                "state_valid": replay["state_valid"],
                "camera_valid": replay["camera_valid"],
                "prompt": str(replay["prompt"]),
                "images": {k: replay["image_" + k] for k in ["main", "left_wrist", "right_wrist"]},
            }
            actual = pipeline(obs)
            np.testing.assert_allclose(actual["state"], replay["normalized_state"], atol=1e-6, rtol=0)
            restored = transforms.Unnormalize(stats, use_quantiles=True)(
                {"state": replay["normalized_state"], "actions": replay["normalized_actions"]}
            )["actions"][:, :14]
            valid = replay["action_mask"][:, :14]
            np.testing.assert_allclose(restored[valid], replay["actions"][valid], atol=1e-6, rtol=0)
            assert len(actual["tokenized_prompt"]) == 200
            for source, target in [
                ("main", "base_0_rgb"),
                ("left_wrist", "left_wrist_0_rgb"),
                ("right_wrist", "right_wrist_0_rgb"),
            ]:
                np.testing.assert_array_equal(actual["image"][target], obs["images"][source])
            masked = pipeline(dict(obs, state_valid=np.zeros(22, bool), camera_valid=np.zeros(3, bool)))
            assert not masked["state"].any()
            assert not any(masked["image_mask"].values())
            rows.append(
                {
                    "task": task,
                    "state_max_error": float(np.abs(actual["state"] - replay["normalized_state"]).max()),
                    "action_max_error": float(np.abs(restored[valid] - replay["actions"][valid]).max()),
                    "tokens": 200,
                    "rgb_masks": True,
                }
            )
    return rows


def create_policy(
    root,
    config,
    manifest,
    metadata,
    device="cuda:0",
    *,
    default_prompt=None,
    sample_kwargs=None,
    repack_transforms=None,
):
    root = Path(root)
    # Use the SHA-checked training Gemma implementation; the PI0 model itself is
    # byte-identical to this hybrid branch. No tracked checkout file is replaced.
    gemma = module_from_file("_yam1_trained_gemma", root / "reference/gemma_pytorch.py")
    previous = pi0_pytorch.PaliGemmaWithExpertModel
    pi0_pytorch.PaliGemmaWithExpertModel = gemma.PaliGemmaWithExpertModel
    try:
        model = pi0_pytorch.PI0Pytorch(config.model)
    finally:
        pi0_pytorch.PaliGemmaWithExpertModel = previous
    if manifest["architecture"] == "domain_io":
        heads = module_from_file("_yam1_domain_heads", root / "reference/domain_heads.py")
        routing = json.loads((root / "assets/yam_hybrid/head-routing.json").read_text())
        heads.install(model, routing["domains"])
        target_index = heads.domain_index(routing, "yam/yam1")
        original = model.sample_actions

        def routed_sample_actions(device, observation, **kwargs):
            ids = torch.full(
                (observation.state.shape[0],), target_index, dtype=torch.long, device=observation.state.device
            )
            heads.set_domains(model, ids)
            return original(device, observation, **kwargs)

        model.sample_actions = routed_sample_actions
        metadata.update(selected_head="yam1", selected_head_index=target_index)
    elif manifest["architecture"] != "shared":
        raise ValueError("Unsupported action architecture")
    load_fp32_weights(model, root / "model.safetensors")
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    data = config.data.create(config.assets_dirs, config.model)
    stats = normalize.load(root / "assets/yam_hybrid")
    repack = repack_transforms or transforms.Group()
    return Policy(
        model,
        transforms=[
            *repack.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data.data_transforms.inputs,
            transforms.Normalize(stats, use_quantiles=True),
            *data.model_transforms.inputs,
        ],
        output_transforms=[
            *data.model_transforms.outputs,
            transforms.Unnormalize(stats, use_quantiles=True),
            *data.data_transforms.outputs,
            *repack.outputs,
        ],
        metadata=metadata,
        sample_kwargs=sample_kwargs,
        is_pytorch=True,
        pytorch_device=device,
    )
