"""Convert YAM JSON/H.264 MCAP episodes to local LeRobot v2.1."""

from pathlib import Path

from filelock import FileLock
import tyro

from openpi.data.yam_mcap import convert
from openpi.data.yam_spec import McapTaskSpec

DEFAULT_TASK = McapTaskSpec()


def main(task: McapTaskSpec = DEFAULT_TASK, *, force_rebuild: bool = False):
    lock = Path(str(task.dataset_path) + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(lock):
        report = convert(task, force=force_rebuild)
    frames = sum(episode["frames"] for episode in report["episodes"])
    print(f"Complete: {task.dataset_path}; {len(report['episodes'])} episodes, {frames} frames")


if __name__ == "__main__":
    tyro.cli(main)
