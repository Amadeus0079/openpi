"""Compute YAM normalization over every frame and episode-clamped action chunk."""

import json

import numpy as np
import pyarrow.parquet as pq
import tyro

from openpi.data.yam_mcap import require_complete
from openpi.data.yam_spec import McapTaskSpec
from openpi.shared import normalize
from openpi.training.config import get_config


def main(config_name: str = "pi05_yam"):
    config = get_config(config_name)
    task = McapTaskSpec()
    report = require_complete(task)
    state_stats, action_stats = normalize.RunningStats(), normalize.RunningStats()
    total = 0
    for episode in report["episodes"]:
        index = episode["episode_index"]
        path = task.dataset_path / f"data/chunk-{index // 1000:03d}/episode_{index:06d}.parquet"
        table = pq.read_table(path, columns=["observation.state", "action"])
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        if state.shape != (episode["frames"], 14) or not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError(f"Invalid state/action arrays in {path}")
        state_stats.update(state)
        for start in range(0, len(state), 256):
            times = np.arange(start, min(start + 256, len(state)))
            future = np.minimum(times[:, None] + np.arange(config.model.action_horizon), len(state) - 1)
            chunks = action[future].copy()
            # Only six joints per arm are deltas; both grippers stay absolute.
            mask = np.array([True] * 6 + [False] + [True] * 6 + [False])
            chunks[..., mask] -= state[times, None, :][..., mask]
            action_stats.update(chunks.reshape(-1, 14))
        total += len(state)
    destination = config.assets_dirs / task.repo_id
    normalize.save(destination, {"state": state_stats.get_statistics(), "actions": action_stats.get_statistics()})
    (destination / "provenance.json").write_text(
        json.dumps(
            {
                "dataset_fingerprint": report["dataset_fingerprint"],
                "frames": total,
                "episodes": len(report["episodes"]),
                "action_horizon": config.model.action_horizon,
                "task_instruction": task.prompt,
            },
            indent=2,
        )
    )
    print(f"Saved normalization for all {total} frames to {destination}")


if __name__ == "__main__":
    tyro.cli(main)
