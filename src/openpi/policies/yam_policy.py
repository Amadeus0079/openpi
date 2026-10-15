"""YAM joint/gripper IO, without ALOHA-specific calibration or sign flips."""

import dataclasses

import numpy as np

from openpi import transforms

CAMERA_MAPPING = {"main": "base_0_rgb", "left": "left_wrist_0_rgb", "right": "right_wrist_0_rgb"}


@dataclasses.dataclass(frozen=True)
class YamInputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        images = {}
        for source, target in CAMERA_MAPPING.items():
            image = np.asarray(data["images"][source])
            if np.issubdtype(image.dtype, np.floating):
                image = np.clip(image * 255, 0, 255).astype(np.uint8)
            if image.ndim != 3:
                raise ValueError(f"{source}: expected a three-dimensional RGB image")
            if image.shape[0] == 3:
                image = image.transpose(1, 2, 0)
            if image.shape[-1] != 3:
                raise ValueError(f"{source}: expected RGB, got {image.shape}")
            images[target] = image
        result = {
            "state": np.asarray(data["state"], dtype=np.float32),
            "image": images,
            "image_mask": dict.fromkeys(images, np.True_),
        }
        for key in ("actions", "prompt"):
            if key in data:
                result[key] = data[key]
        return result


@dataclasses.dataclass(frozen=True)
class YamOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :14]}
