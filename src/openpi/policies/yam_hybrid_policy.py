"""Camera-anchor EE IO; intentionally has no joint delta/absolute transforms."""

import dataclasses

import numpy as np

from openpi import transforms

CAMERAS = dict(
    zip(("main", "left_wrist", "right_wrist"), ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"), strict=True)
)


@dataclasses.dataclass(frozen=True)
class HybridInputs(transforms.DataTransformFn):
    def __call__(self, data):
        state = np.asarray(data["state"], np.float32)
        valid = np.asarray(data["state_valid"], bool)
        camera_valid = np.asarray(data["camera_valid"], bool)
        if (
            state.shape != (22,)
            or valid.shape != (22,)
            or camera_valid.shape != (3,)
            or not np.isfinite(state[valid]).all()
        ):
            raise ValueError("Expected finite valid 22-D state, 22-D state mask, 3-D camera mask")
        if not isinstance(data.get("prompt"), str) or not data["prompt"].strip():
            raise ValueError("Per-observation task instruction is required")
        images, masks = {}, {}
        for i, (source, target) in enumerate(CAMERAS.items()):
            if camera_valid[i] and source not in data["images"]:
                raise ValueError("Valid camera has no image: " + source)
            image = np.asarray(data["images"].get(source, np.zeros((224, 224, 3), np.uint8)))
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise ValueError("Camera EE API expects HWC uint8 RGB images")
            images[target] = image
            masks[target] = np.bool_(camera_valid[i])
        return {
            "state": np.where(valid, state, 0),
            "state_valid": valid,
            "image": images,
            "image_mask": masks,
            "prompt": data["prompt"],
        }


@dataclasses.dataclass(frozen=True)
class MaskState(transforms.DataTransformFn):
    def __call__(self, data):
        data = dict(data)
        data["state"] = np.where(data.pop("state_valid"), data["state"], 0).astype(np.float32)
        return data


@dataclasses.dataclass(frozen=True)
class HybridOutputs(transforms.DataTransformFn):
    def __call__(self, data):
        return {"actions": np.asarray(data["actions"], np.float32)[..., :14]}
