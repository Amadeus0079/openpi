"""SHA-bound raw support demonstrations for live overlay and return-to-start."""

import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def support_references(checkpoint, recordings):
    clips = json.loads((checkpoint / "assets/yam_hybrid/deployment/target/clip-geometry.json").read_text())["clips"]
    paths = {}
    for file in Path(recordings).glob("*/mcap/success/*.mcap"):
        paths.setdefault(file.stem, []).append(file)
    result = []
    for clip in clips:
        if clip.get("split") != "target_support" or clip.get("machine") != "yam/yam1":
            continue
        matches = paths.get(clip["original_id"], [])
        if len(matches) != 1:
            raise ValueError(f"Expected one local support MCAP: {clip['original_id']}")
        result.append(
            {
                "path": matches[0],
                "original_id": clip["original_id"],
                "task": clip["pairing"],
                "prompt": clip["export"]["provenance"]["task_instruction"],
                "sha256": clip["export"]["provenance"]["source_record"]["input"]["source_sha256"],
            }
        )
    if not result:
        raise ValueError("No SHA-bound target-support demonstrations")
    return sorted(result, key=lambda row: (row["task"], row["original_id"]))


class RecordedStart:
    def __init__(self, row, serials, messages):
        self.row = row
        if sha(row["path"]) != row["sha256"]:
            raise ValueError("Local MCAP differs from training support identity")
        self.images, details, joints = {}, {}, []
        for ui, key in [("left", "left_wrist"), ("main", "main"), ("right", "right_wrist")]:
            stream = messages(row["path"], f"/cameras/{ui}/rgb", decode_images=True)
            try:
                _, stamp, data = next(stream)
                if "_pixels" in data:
                    pixels = data["_pixels"]
                elif data["encoding"] == "png":
                    pixels = np.asarray(Image.open(io.BytesIO(base64.b64decode(data["data"]))).convert("RGB"))
                elif data["encoding"] == "rgb8":
                    pixels = np.frombuffer(base64.b64decode(data["data"]), np.uint8).reshape(data["shape"])
                else:
                    raise ValueError("Unsupported recorded camera encoding")
                if str(data["serial"]) != serials[key] or pixels.shape != (480, 640, 3):
                    raise ValueError("Recorded/current camera identity or orientation differs")
                self.images[ui] = pixels.copy()
                details[ui] = {
                    "serial": data["serial"],
                    "source_time_ns": stamp,
                    "frame_number": data.get("frame_number"),
                }
            finally:
                stream.close()
        for side in ["left", "right"]:
            stream = messages(row["path"], f"/{side}/state")
            try:
                _, _, data = next(stream)
                joints.extend(data["position"])
            finally:
                stream.close()
        self.target = np.asarray(joints, float)
        if self.target.shape != (14,) or not np.isfinite(self.target).all():
            raise ValueError("Invalid original first joint feedback")
        self.info = {k: str(v) if isinstance(v, Path) else v for k, v in row.items()}
        self.info.update(
            reference_kind="raw first RGB / first measured joints", cameras=details, target=self.target.tolist()
        )


def slow_ramp(current, target, bounds):
    current, target, bounds = np.asarray(current, float), np.asarray(target, float), np.asarray(bounds, float)
    if current.shape != (14,) or target.shape != (14,) or not np.isfinite([current, target]).all():
        raise ValueError("Expected finite native joint feedback[14]")
    if np.any(current < bounds[:, 0] - 1e-6) or np.any(current > bounds[:, 1] + 1e-6):
        raise ValueError("Current joints exceed nominal limits")
    if np.any(target < bounds[:, 0] - 1e-6) or np.any(target > bounds[:, 1] + 1e-6):
        raise ValueError("Recorded target exceeds nominal limits")
    step = np.full(14, 0.005)
    step[[6, 13]] = 0.01
    count = max(1, int(np.ceil(np.max(np.abs(target - current) / step))))
    return current + (target - current) * np.arange(1, count + 1)[:, None] / count
