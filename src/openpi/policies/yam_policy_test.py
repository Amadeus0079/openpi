import numpy as np
import pytest

from openpi import transforms
from openpi.policies.yam_policy import CAMERA_MAPPING
from openpi.policies.yam_policy import YamInputs
from openpi.policies.yam_policy import YamOutputs
from openpi.training.config import get_config


def test_joint_delta_round_trip_preserves_grippers():
    state = np.arange(14, dtype=np.float32) / 10
    actions = np.arange(16 * 14, dtype=np.float32).reshape(16, 14) / 50
    original = actions.copy()
    data = {"state": state, "actions": actions}
    mask = transforms.make_bool_mask(6, -1, 6, -1)
    data = transforms.DeltaActions(mask)(data)
    np.testing.assert_array_equal(data["actions"][:, [6, 13]], original[:, [6, 13]])
    np.testing.assert_allclose(data["actions"][:, 0], original[:, 0] - state[0])
    data = transforms.AbsoluteActions(mask)(data)
    data["actions"] = np.pad(data["actions"], ((0, 0), (0, 18)))
    np.testing.assert_allclose(YamOutputs()(data)["actions"], original, atol=1e-6)


@pytest.mark.parametrize("channels_first", [True, False])
def test_camera_mapping_and_rgb_values(channels_first):
    images = {}
    for i, key in enumerate(CAMERA_MAPPING):
        image = np.full((8, 12, 3), 10 + i * 50, np.uint8)
        images[key] = image.transpose(2, 0, 1).astype(np.float32) / 255 if channels_first else image
    sample = YamInputs()({"images": images, "state": np.arange(14), "prompt": "pour water"})
    assert sample["prompt"] == "pour water"
    for i, key in enumerate(CAMERA_MAPPING.values()):
        assert sample["image"][key].shape == (8, 12, 3)
        np.testing.assert_allclose(sample["image"][key], 10 + i * 50, atol=1)
        assert sample["image_mask"][key]


def test_yam_config_is_full_8_by_32_training():
    config = get_config("pi05_yam")
    assert config.fsdp_devices == 8
    assert config.batch_size == 256
    assert config.batch_size // config.fsdp_devices == 32
    assert config.num_train_steps == 5000
    assert config.model.pi05
    assert config.model.discrete_state_input
    assert config.model.max_token_len == 200
    assert "lora" not in config.model.paligemma_variant
    assert "lora" not in config.model.action_expert_variant
