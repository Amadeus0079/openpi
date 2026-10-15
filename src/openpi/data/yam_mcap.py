"""Strict YAM MCAP alignment and transactional LeRobot v2.1 conversion.

Only outer source_time_ns is used for sampling. A camera message is one H.264
access unit; decoder contexts persist for the entire camera stream/episode.
"""

import base64
import dataclasses
from fractions import Fraction
import json
from pathlib import Path
import shutil
import uuid

import numpy as np

from openpi.data.yam_spec import CAMERAS
from openpi.data.yam_spec import McapTaskSpec
from openpi.data.yam_spec import digest

REPORT = "mcap_conversion.json"


@dataclasses.dataclass
class Sample:
    time_ns: int
    data: dict


def vector(value, size: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"Expected {size} finite values, got {result}")
    return result


def read_streams(path: Path, task: McapTaskSpec) -> tuple[dict[str, list[Sample]], dict]:
    from mcap.reader import make_reader  # noqa: PLC0415 -- optional conversion dependency

    topics = {
        "left_state": task.left_state_topic,
        "right_state": task.right_state_topic,
        "action": task.action_topic,
        **task.camera_topics,
    }
    by_topic = {topic: key for key, topic in topics.items()}
    streams = {key: [] for key in topics}
    report = {"file": str(path), "topics": topics, "limited_actions": 0, "camera_gaps": [], "timing": {}}
    offsets = {key: [] for key in topics}
    device_offsets = {key: [] for key in CAMERAS}
    with path.open("rb") as handle:
        for _, channel, message in make_reader(handle).iter_messages(topics=list(by_topic), log_time_order=False):
            key = by_topic[channel.topic]
            if channel.message_encoding != "json":
                raise ValueError(f"{channel.topic}: expected JSON messages")
            envelope = json.loads(message.data)
            time_ns = envelope["source_time_ns"]
            if type(time_ns) is not int or time_ns < 0:
                raise ValueError(f"{channel.topic}: invalid source_time_ns")
            data = envelope["data"]
            if streams[key] and time_ns <= streams[key][-1].time_ns:
                raise ValueError(f"{channel.topic}: non-increasing source_time_ns at {time_ns}")
            if key in ("left_state", "right_state", "action"):
                data["position"] = vector(data["position"], 14 if key == "action" else 7)
            if key == "action":
                report["limited_actions"] += int(bool(data.get("limited", False)))
            if key in CAMERAS:
                if data.get("encoding") != "h264":
                    raise ValueError(f"{channel.topic}: expected h264")
                if streams[key]:
                    previous = streams[key][-1]
                    gap_ms = (time_ns - previous.time_ns) / 1e6
                    frame_gap = data.get("frame_number", 0) - previous.data.get("frame_number", 0)
                    if gap_ms > 1.5 * 1000 / task.fps or frame_gap > 1:
                        report["camera_gaps"].append(
                            {
                                "camera": key,
                                "source_time_ns": time_ns,
                                "gap_ms": gap_ms,
                                "frame_number_delta": frame_gap,
                            }
                        )
                if "device_timestamp_ms" in data:
                    device_offsets[key].append(data["device_timestamp_ms"] - time_ns / 1e6)
            offsets[key].append((envelope.get("record_time_ns", message.log_time) - time_ns) / 1e6)
            streams[key].append(Sample(time_ns, data))
    for key, samples in streams.items():
        if not samples:
            raise ValueError(f"Missing stream {topics[key]}")
        report["timing"][key] = {
            "messages": len(samples),
            "first_source_time_ns": samples[0].time_ns,
            "last_source_time_ns": samples[-1].time_ns,
            "record_minus_source_ms": summary(offsets[key]),
        }
        if key in CAMERAS and device_offsets[key]:
            report["timing"][key]["device_minus_source_ms_diagnostic_only"] = summary(device_offsets[key])
    return streams, report


def summary(values) -> dict:
    return {"min": float(np.min(values)), "max": float(np.max(values)), "mean": float(np.mean(values))}


def align(streams: dict[str, list[Sample]], task: McapTaskSpec) -> tuple[np.ndarray, dict, dict]:
    start = max(samples[0].time_ns for samples in streams.values())
    end = min(samples[-1].time_ns for samples in streams.values())
    if end < start:
        raise ValueError("Required streams have no common coverage")
    # Integer arithmetic avoids loss of nanoseconds at Unix-epoch magnitudes.
    timeline = (
        start + np.arange((end - start) * task.fps // 1_000_000_000 + 1, dtype=np.int64) * 1_000_000_000 // task.fps
    )
    indices, diagnostics = {}, {}
    for key, samples in streams.items():
        times = np.array([sample.time_ns for sample in samples], dtype=np.int64)
        selected = np.searchsorted(times, timeline, side="right") - 1
        age_ms = (timeline - times[selected]) / 1e6
        limit = (
            task.image_max_age_ms
            if key in CAMERAS
            else (task.action_max_age_ms if key == "action" else task.state_max_age_ms)
        )
        bad = np.flatnonzero(age_ms > limit)
        if bad.size:
            i = bad[0]
            raise ValueError(f"{key}: stale sample at {timeline[i]}: {age_ms[i]:.3f} ms > {limit} ms")
        indices[key] = selected
        diagnostics[key] = {
            "sample_age_ms": summary(age_ms),
            "repeated_samples": int(np.count_nonzero(np.diff(selected) == 0)),
        }
    return timeline, indices, diagnostics


def decode_camera(samples: list[Sample]):
    """Yield RGB frames in source order, preserving packet timestamps through decode delay."""
    import av  # noqa: PLC0415 -- keep metadata validation independent of video decoding

    decoder = av.CodecContext.create("h264", "r")
    decoder.options = {"err_detect": "explode"}
    decoder.thread_count = 1
    output_index = 0
    for index in range(len(samples) + 1):
        packet = None
        if index < len(samples):
            packet = av.Packet(base64.b64decode(samples[index].data["data"], validate=True))
            packet.pts = index
            packet.dts = index
            packet.time_base = Fraction(1, 1)
        for frame in decoder.decode(packet):
            if output_index >= len(samples) or frame.pts != output_index or frame.is_corrupt:
                raise ValueError(
                    f"H.264 corrupt, reordered or unmatched frame: expected {output_index}, got {frame.pts}"
                )
            rgb = frame.to_ndarray(format="rgb24")
            if list(rgb.shape) != samples[output_index].data["shape"]:
                raise ValueError(f"H.264 shape mismatch at frame {output_index}")
            output_index += 1
            yield rgb
    if output_index != len(samples):
        raise ValueError(f"H.264 decoded {output_index} frames for {len(samples)} messages")


def episode_frames(streams: dict, indices: dict, prompt: str):
    decoders = {key: iter(decode_camera(streams[key])) for key in CAMERAS}
    positions = dict.fromkeys(CAMERAS, -1)
    current = {}
    for i in range(len(indices["action"])):
        frame = {
            "observation.state": np.concatenate(
                [streams[key][indices[key][i]].data["position"] for key in ("left_state", "right_state")]
            ),
            "action": streams["action"][indices["action"][i]].data["position"].copy(),
            "task": prompt,
        }
        for key in CAMERAS:
            while positions[key] < indices[key][i]:
                current[key] = next(decoders[key])
                positions[key] += 1
            frame[f"observation.images.{key}"] = current[key]
        yield frame
    # Validate even the unsampled tail; no silent decoder failures.
    for decoder in decoders.values():
        for _ in decoder:
            pass


def source_manifest(files: list[Path]) -> list[dict]:
    return [
        {"path": str(path.resolve()), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for path in files
    ]


def output_manifest(root: Path) -> list[dict]:
    return [
        {"path": str(path.relative_to(root)), "size": path.stat().st_size}
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name != REPORT
        and not any(part.startswith(".") for part in path.relative_to(root).parts)
    ]


def require_complete(task: McapTaskSpec, files: list[Path] | None = None) -> dict:
    root = task.dataset_path
    if root.with_name(root.name + ".incomplete").exists():
        raise ValueError(f"Interrupted conversion for {root}; use --force-rebuild")
    try:
        report = json.loads((root / REPORT).read_text())
        if report["status"] != "complete" or report["config_fingerprint"] != task.fingerprint:
            raise ValueError("Incomplete dataset or conversion configuration mismatch")
        files = sorted(task.raw_dir_path.rglob("*.mcap")) if files is None else files
        if not files or report["sources"] != source_manifest(files):
            raise ValueError("MCAP source manifest mismatch")
        if report["outputs"] != output_manifest(root):
            raise ValueError("Dataset files missing or changed")
        info = json.loads((root / "meta/info.json").read_text())
        if info["codebase_version"] != "v2.1" or info["total_episodes"] != len(files):
            raise ValueError("Invalid LeRobot v2.1 metadata")
        return report
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"Not a complete MCAP dataset: {root}: {exc}") from exc


def convert(task: McapTaskSpec, *, force: bool = False, files: list[Path] | None = None) -> dict:
    """Publish only after all episodes and videos succeed. Caller holds the dataset lock.

    Adapted from YushunXiang/OCL-openpi feat/mcap, commit e517a3a2.
    """
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

    files = sorted(task.raw_dir_path.rglob("*.mcap")) if files is None else sorted(files)
    if not files:
        raise ValueError(f"No MCAP files in {task.raw_dir_path}")
    root = task.dataset_path
    staging = root.with_name(root.name + ".incomplete")
    if root.exists() and not force:
        return require_complete(task, files)
    if staging.exists():
        if not force:
            raise ValueError(f"Interrupted conversion: {staging}; use --force-rebuild to retry")
        shutil.rmtree(staging)
    root.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "incomplete",
        "generation": uuid.uuid4().hex,
        "config": task.conversion_config(),
        "config_fingerprint": task.fingerprint,
        "sources": source_manifest(files),
        "episodes": [],
    }
    dataset = None
    path = files[0]
    try:
        for episode, path in enumerate(files):
            streams, diagnostics = read_streams(path, task)
            report["episodes"].append({**diagnostics, "episode_index": episode})
            timeline, indices, alignment = align(streams, task)
            frames = iter(episode_frames(streams, indices, task.prompt))
            first = next(frames)
            if dataset is None:
                features = {
                    key: {"dtype": "float32", "shape": (14,), "names": None} for key in ("observation.state", "action")
                }
                features.update(
                    {
                        f"observation.images.{key}": {
                            "dtype": "video",
                            "shape": first[f"observation.images.{key}"].shape,
                            "names": ["height", "width", "channels"],
                        }
                        for key in CAMERAS
                    }
                )
                dataset = LeRobotDataset.create(
                    repo_id=task.repo_id,
                    root=staging,
                    fps=task.fps,
                    robot_type=task.robot_type,
                    features=features,
                    use_videos=True,
                    image_writer_threads=4,
                    image_writer_processes=0,
                )
            dataset.add_frame(first)
            for frame_index, frame in enumerate(frames, start=1):
                dataset.add_frame(frame)
                # LeRobot's image queue is unbounded. Bound outstanding RGB frames
                # while retaining parallel PNG writes and continuous decoding.
                if frame_index % 32 == 0:
                    dataset.image_writer.wait_until_done()
            print(f"Saving episode {episode}: {path.name}, {len(timeline)} aligned frames", flush=True)
            dataset.save_episode()
            report["episodes"][-1].update(
                frames=len(timeline), start_source_time_ns=int(timeline[0]), alignment=alignment
            )
            (staging / REPORT).write_text(json.dumps(report, indent=2))
        if hasattr(dataset, "stop_image_writer"):
            dataset.stop_image_writer()
        info = json.loads((staging / "meta/info.json").read_text())
        if info["codebase_version"] != "v2.1" or info["total_episodes"] != len(files):
            raise ValueError("Writer did not produce complete LeRobot v2.1 metadata")
        if report["sources"] != source_manifest(files):
            raise ValueError("MCAP sources changed during conversion")
        report["status"] = "complete"
        report["outputs"] = output_manifest(staging)
        report["dataset_fingerprint"] = digest(
            {key: report[key] for key in ("generation", "config_fingerprint", "sources", "outputs")}
        )
        (staging / REPORT).write_text(json.dumps(report, indent=2))
        if root.exists():
            shutil.rmtree(root)
        staging.rename(root)
        return report
    except BaseException as exc:
        staging.mkdir(parents=True, exist_ok=True)
        report.update(status="incomplete", error=f"{path}: {exc}")
        (staging / REPORT).write_text(json.dumps(report, indent=2))
        raise RuntimeError(f"MCAP conversion failed: {path}: {exc}") from exc
    finally:
        if dataset is not None:
            dataset.stop_image_writer()
