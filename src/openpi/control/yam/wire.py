"""OpenPI NumPy/msgpack wire compatibility without importing its NumPy-1 package.

Array encoding follows Physical-Intelligence/openpi openpi_client/msgpack_numpy.py
(Apache-2.0); transport has single-attempt request semantics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def pack_array(value):
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind in "VOc":
        raise ValueError("Unsupported array dtype")
    if isinstance(value, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {b"__npgeneric__": True, b"data": value.item(), b"dtype": value.dtype.str}
    raise TypeError(type(value).__name__)


def unpack_array(value):
    if b"__ndarray__" in value:
        return np.ndarray(buffer=value[b"data"], dtype=value[b"dtype"], shape=value[b"shape"]).copy()
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


class PolicyConnection:
    def __init__(self, url, contract, checkpoint, timeout=30):
        import msgpack
        import websockets.sync.client

        self.msgpack, self.timeout = msgpack, timeout
        self.socket = websockets.sync.client.connect(
            url,
            compression=None,
            max_size=None,
            open_timeout=timeout,
            close_timeout=2,
            ping_interval=None,
            proxy=None,
        )
        self.failed = False
        try:
            self.metadata = self._receive()
            for key in (
                "dataset_identity",
                "action_version",
                "horizon",
                "fps",
                "state_dim",
                "action_dim",
                "ee_definition",
            ):
                if self.metadata.get(key) != contract[key]:
                    raise ValueError(f"Policy metadata mismatch: {key}")
            manifest_path = Path(checkpoint) / "inference-manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                expected = manifest["files"]["model.safetensors"]["sha256"]
                if self.metadata.get("model_sha256") != expected:
                    raise ValueError("Policy checkpoint weight identity mismatch")
            self.reset()
        except BaseException:
            self.close()
            raise

    def _receive(self):
        raw = self.socket.recv(timeout=self.timeout)
        if isinstance(raw, str):
            raise RuntimeError("Policy server error: " + raw[:1000])
        return self.msgpack.unpackb(raw, object_hook=unpack_array)

    def exchange(self, payload):
        if self.failed:
            raise RuntimeError("Policy connection cannot be reused after failure")
        try:
            self.socket.send(self.msgpack.packb(payload, default=pack_array))
            return self._receive()
        except BaseException:
            self.failed = True
            self.close()
            raise

    def reset(self):
        if self.exchange({"type": "reset"}) != {"status": "ok"}:
            raise ValueError("Policy reset not acknowledged")

    def infer(self, observation):
        reply = self.exchange(observation)
        actions = np.asarray(reply["actions"], np.float32)
        if actions.shape != (30, 14) or not np.isfinite(actions).all():
            raise ValueError("Policy returned invalid actions")
        return actions, reply.get("server_timing", {})

    def close(self):
        self.socket.close()
