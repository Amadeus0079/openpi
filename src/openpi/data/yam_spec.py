"""Shared YAM dataset contract for conversion, normalization and training."""

import dataclasses
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CAMERAS = ("main", "left", "right")
TASK_INSTRUCTION = "Pour the water from the right cup into the left cup."
REPO_ID = "yam/pour_water"
DATASET_ROOT = ROOT / "data" / REPO_ID


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclasses.dataclass(frozen=True)
class McapTaskSpec:
    raw_dir: str = str(ROOT.parents[1] / "datasets/yam")
    dataset_dir: str = str(DATASET_ROOT)
    repo_id: str = REPO_ID
    prompt: str = TASK_INSTRUCTION
    robot_type: str = "yam"
    fps: int = 30
    action_topic: str = "/action/accepted"
    left_state_topic: str = "/left/state"
    right_state_topic: str = "/right/state"
    camera_topics: dict[str, str] = dataclasses.field(
        default_factory=lambda: {key: f"/cameras/{key}/rgb" for key in CAMERAS}
    )
    image_max_age_ms: float = 100.0
    state_max_age_ms: float = 50.0
    action_max_age_ms: float = 50.0

    def __post_init__(self):
        if self.fps <= 0 or not self.prompt.strip():
            raise ValueError("A positive FPS and nonempty task instruction are required")

    @property
    def raw_dir_path(self) -> Path:
        return Path(self.raw_dir).resolve()

    @property
    def dataset_path(self) -> Path:
        return Path(self.dataset_dir).resolve()

    def conversion_config(self) -> dict:
        return {"converter_version": 1, **dataclasses.asdict(self)}

    @property
    def fingerprint(self) -> str:
        return digest(self.conversion_config())
