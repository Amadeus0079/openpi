"""Portable policy/asset/wire regressions using a fake model, never real weights."""

import asyncio
import dataclasses
import hashlib
import json

import numpy as np
from openpi_client import msgpack_numpy
import pytest
import torch
import websockets.asyncio.client
import websockets.asyncio.server

from openpi.models import tokenizer
from openpi.models.pi0_config import Pi0Config
from openpi.policies import policy_config
from openpi.policies.hybrid_assets import verify_checkpoint
from openpi.policies.yam_hybrid_policy import HybridInputs
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.shared import normalize
from openpi.training import config as configs


def fixture_checkpoint(root):
    config = configs.get_config("pi05_yam_hybrid")
    assets = root / "assets/yam_hybrid"
    (assets / "deployment").mkdir(parents=True)
    stats = {
        k: normalize.NormStats(mean=np.zeros(d), std=np.ones(d), q01=np.full(d, -2.0), q99=np.full(d, 3.0))
        for k, d in (("state", 22), ("actions", 14))
    }
    normalize.save(assets, stats)
    contract = {
        "dataset_identity": "fixture",
        "action_version": "camera_anchor_ee_v1",
        "state_dim": 22,
        "action_dim": 14,
        "horizon": 30,
        "fps": 30,
        "ee_definition": "gripper_mount",
        "norm_stats_sha256": hashlib.sha256((assets / "norm_stats.json").read_bytes()).hexdigest(),
    }
    (assets / "deployment.json").write_text(json.dumps(contract))
    (assets / "config.json").write_text(json.dumps({"model": dataclasses.asdict(config.model)}))
    (assets / "deployment/tokenizer.model").write_bytes(b"fixture-tokenizer")
    (root / "model.safetensors").write_bytes(b"fixture-not-a-real-model")
    (root / "complete.json").write_text(json.dumps({"step": 10000}))
    (root / "run.json").write_text(json.dumps({"step": 10000, "hybrid": {"deployment": contract}}))
    files = {
        str(p.relative_to(root)): {"size": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in root.rglob("*")
        if p.is_file()
    }
    (root / "inference-manifest.json").write_text(
        json.dumps({"dataset_identity": "fixture", "step": 10000, "files": files})
    )
    return config


def test_policy_masks_padding_and_absolute_ee_output(monkeypatch, tmp_path):
    config = fixture_checkpoint(tmp_path)
    expected = np.tile(np.linspace(-0.3, 0.7, 14), (30, 1)).astype(np.float32)
    normalized = np.pad(2 * (expected + 2) / (5 + 1e-6) - 1, ((0, 0), (0, 18))).astype(np.float32)

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.paligemma_with_expert = type("FakeGemma", (), {"to_bfloat16_for_selected_params": lambda *a: None})()

        def sample_actions(self, device, obs, **kwargs):
            assert device == "cpu"
            assert obs.state.shape == (1, 32)
            assert obs.state[0, 6] == 0
            assert (obs.state[0, 22:] == 0).all()
            assert not obs.image_masks["right_wrist_0_rgb"].item()
            assert obs.images["base_0_rgb"].shape == (1, 3, 224, 224)
            return torch.from_numpy(normalized)[None]

    class FakeTokenizer:
        def __init__(self, max_len):
            self.max_len = max_len

        def tokenize(self, prompt, state):
            assert prompt == "pour water"
            assert state.shape == (32,)
            assert state[6] == 0
            return np.zeros(self.max_len, np.int32), np.ones(self.max_len, bool)

    monkeypatch.setattr(Pi0Config, "load_pytorch", lambda *args: FakeModel())
    monkeypatch.setattr(tokenizer, "PaligemmaTokenizer", FakeTokenizer)
    policy = policy_config.create_trained_policy(config, tmp_path, pytorch_device="cpu")
    valid = np.ones(22, bool)
    valid[6] = False
    result = policy.infer(
        {
            "state": np.ones(22, np.float32),
            "state_valid": valid,
            "camera_valid": np.array([True, True, False]),
            "images": {k: np.zeros((480, 640, 3), np.uint8) for k in ("main", "left_wrist")},
            "prompt": "pour water",
        }
    )
    np.testing.assert_allclose(result["actions"], expected, atol=1e-6)
    assert policy.metadata["dataset_identity"] == "fixture"


def test_asset_tampering_rejected_before_model_load(tmp_path):
    config = fixture_checkpoint(tmp_path)
    (tmp_path / "assets/yam_hybrid/norm_stats.json").write_text("corrupted")
    with pytest.raises(ValueError, match="checksum"):
        verify_checkpoint(tmp_path, config.model)


def test_chw_input_is_not_silently_reinterpreted():
    with pytest.raises(ValueError, match="HWC"):
        HybridInputs()(
            {
                "state": np.zeros(22),
                "state_valid": np.ones(22, bool),
                "camera_valid": [True, False, False],
                "images": {"main": np.zeros((3, 480, 640), np.uint8)},
                "prompt": "pour water",
            }
        )


def test_websocket_metadata_reset_and_numpy_actions():

    class FakePolicy:
        def reset(self):
            self.was_reset = True

        def infer(self, observation):
            assert self.was_reset
            assert observation["images"]["main"].shape == (5, 6, 3)
            return {"actions": np.zeros((30, 14), np.float32)}

    async def check():
        handler = WebsocketPolicyServer(FakePolicy(), metadata={"horizon": 30, "dataset_identity": "fixture"})
        async with websockets.asyncio.server.serve(handler._handler, "127.0.0.1", 0) as server:  # noqa: SLF001
            port = server.sockets[0].getsockname()[1]
            async with websockets.asyncio.client.connect(f"ws://127.0.0.1:{port}") as client:
                metadata = msgpack_numpy.unpackb(await client.recv())
                assert metadata["horizon"] == 30
                await client.send(msgpack_numpy.packb({"type": "reset"}))
                assert msgpack_numpy.unpackb(await client.recv()) == {"status": "ok"}
                await client.send(msgpack_numpy.packb({"images": {"main": np.zeros((5, 6, 3), np.uint8)}}))
                reply = msgpack_numpy.unpackb(await client.recv())
                assert reply["actions"].shape == (30, 14)

    asyncio.run(check())
