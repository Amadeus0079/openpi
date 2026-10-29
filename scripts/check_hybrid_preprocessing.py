"""Checkpoint-only regression against saved training normalization and masks."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from openpi import transforms
from openpi.models.model import Observation
from openpi.models_pytorch.preprocessing_pytorch import preprocess_observation_pytorch
from openpi.policies.yam_hybrid_policy import CAMERAS
from openpi.shared import image_tools
from openpi.shared import normalize
from openpi.training import config as configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assets = args.checkpoint / "assets/yam_hybrid"
    os.environ["PALIGEMMA_TOKENIZER_PATH"] = str(assets / "deployment/tokenizer.model")
    x = np.load(assets / "deployment/replay.npz", allow_pickle=False)
    config = configs.get_config("pi05_yam_hybrid")
    data = config.data.create(config.assets_dirs, config.model)
    stats = normalize.load(assets)
    observation = {
        "state": x["state"],
        "state_valid": x["state_valid"],
        "camera_valid": x["camera_valid"],
        "prompt": str(x["prompt"]),
        "images": {name: x["image_" + name] for name in CAMERAS},
    }
    pipeline = transforms.compose(
        [
            *data.data_transforms.inputs,
            transforms.Normalize(stats, use_quantiles=True),
            *data.model_transforms.inputs,
        ]
    )
    processed = pipeline(observation)
    np.testing.assert_allclose(processed["state"], x["normalized_state"], atol=1e-6)
    restored = transforms.Unnormalize(stats, use_quantiles=True)(
        {
            "state": x["normalized_state"],
            "actions": x["normalized_actions"],
        }
    )["actions"][:, :14]
    valid = x["action_mask"][:, :14]
    np.testing.assert_allclose(restored[valid], x["actions"][valid], atol=1e-6)
    batched = {
        key: {k: torch.from_numpy(np.array(v))[None] for k, v in value.items()}
        if isinstance(value, dict)
        else torch.from_numpy(np.array(value))[None]
        for key, value in processed.items()
    }
    model_obs = preprocess_observation_pytorch(Observation.from_dict(batched), train=False)
    # Inference and training both use the same HWC RGB letterbox operation;
    # Observation converts uint8 to float32 [-1, 1] and CHW before the model.
    for index, (source, target) in enumerate(CAMERAS.items()):
        expected = image_tools.resize_with_pad(observation["images"][source], 224, 224)
        expected = torch.from_numpy(np.asarray(expected, dtype=np.float32) / 255 * 2 - 1)[None]
        actual = model_obs.images[target]
        if actual.shape[1] == 3:
            actual = actual.permute(0, 2, 3, 1)
        torch.testing.assert_close(actual, expected)
        assert bool(model_obs.image_masks[target].item()) == bool(x["camera_valid"][index])
    masked = dict(observation, state_valid=np.zeros(22, bool), camera_valid=np.zeros(3, bool))
    z = pipeline(masked)
    assert not z["state"].any()
    assert not any(z["image_mask"].values())
    report = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "state_max_error": float(np.abs(processed["state"] - x["normalized_state"]).max()),
        "valid_action_max_error": float(np.abs(restored[valid] - x["actions"][valid]).max()),
        "rgb_letterbox_masks": True,
        "masked_state_after_normalization": True,
        "token_count": len(processed["tokenized_prompt"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
