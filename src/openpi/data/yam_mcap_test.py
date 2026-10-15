import base64
import dataclasses
from fractions import Fraction
import json

import av
from mcap.writer import Writer
import numpy as np
import pytest

from openpi.data import yam_mcap as mcap
from openpi.data.yam_spec import McapTaskSpec


def get_task(_name):
    return McapTaskSpec()


def write_mcap(path, *, reverse_channels=False, bad_position=False):
    topics = [
        "/left/state",
        "/right/state",
        "/action/requested",
        "/action/accepted",
        "/cameras/main/rgb",
        "/cameras/left/rgb",
        "/cameras/right/rgb",
    ]
    with path.open("wb") as handle:
        writer = Writer(handle)
        writer.start()
        channels = {
            topic: writer.register_channel(topic, "json", 0)
            for topic in (reversed(topics) if reverse_channels else topics)
        }
        for tick in range(4):
            for topic in topics:
                if "cameras" in topic:
                    data = {"encoding": "h264", "shape": [16, 16, 3], "data": "", "frame_number": tick}
                else:
                    size = 14 if "action" in topic else 7
                    value = 99 if "requested" in topic else (2 if "right" in topic else 1)
                    data = {"position": [value] * size, "limited": "accepted" in topic}
                    if bad_position and topic == "/left/state":
                        data["position"] = [1]
                envelope = {
                    "source_time_ns": tick * 33_333_334,
                    "record_time_ns": tick * 33_333_334 + 200,
                    "data": data,
                }
                writer.add_message(channels[topic], tick * 33_333_334 + 100, json.dumps(envelope).encode(), 0)
        writer.finish()


def test_channel_ids_and_accepted_actions(tmp_path):
    task = get_task("yam_0012")
    for reverse in (False, True):
        path = tmp_path / f"{reverse}.mcap"
        write_mcap(path, reverse_channels=reverse)
        streams, report = mcap.read_streams(path, task)
        assert report["limited_actions"] == 4
        np.testing.assert_array_equal(streams["action"][0].data["position"], np.ones(14))
        np.testing.assert_array_equal(streams["left_state"][0].data["position"], np.ones(7))
        np.testing.assert_array_equal(streams["right_state"][0].data["position"], np.full(7, 2))


def test_left_right_frame_order_and_causal_alignment(tmp_path, monkeypatch):
    path = tmp_path / "episode.mcap"
    write_mcap(path)
    task = get_task("yam_0012")
    streams, _ = mcap.read_streams(path, task)
    timeline, indices, _ = mcap.align(streams, task)
    for key, selected in indices.items():
        assert all(streams[key][index].time_ns <= stamp for index, stamp in zip(selected, timeline, strict=True))
    monkeypatch.setattr(mcap, "decode_camera", lambda samples: iter([np.zeros((16, 16, 3), np.uint8)] * len(samples)))
    frames = list(mcap.episode_frames(streams, indices, task.prompt))
    np.testing.assert_array_equal(frames[0]["observation.state"], [1] * 7 + [2] * 7)
    assert frames[0]["task"] == task.prompt
    assert set(frames[0]) == {"observation.state", "action", "task", *[f"observation.images.{k}" for k in mcap.CAMERAS]}


@pytest.mark.parametrize(("key", "limit"), [("main", 100), ("left_state", 50), ("action", 50)])
def test_stale_samples_fail_without_skipping(key, limit):
    task = get_task("yam_0012")
    streams = {
        name: [mcap.Sample(i * 10_000_000, {}) for i in range(31)]
        for name in (*mcap.CAMERAS, "left_state", "right_state", "action")
    }
    streams[key] = [mcap.Sample(0, {}), mcap.Sample(300_000_000, {})]
    with pytest.raises(ValueError, match=f"{key}: stale sample"):
        mcap.align(streams, task)
    streams[key] = [mcap.Sample(i * limit * 1_000_000, {}) for i in range(300 // limit + 1)]
    mcap.align(streams, task)


@pytest.mark.parametrize("value", [[1] * 13, [float("nan")] * 14, [float("inf")] * 14])
def test_reject_invalid_values(value):
    with pytest.raises(ValueError, match="finite values"):
        mcap.vector(value, 14)


def test_bad_dimension_reports_error(tmp_path):
    path = tmp_path / "bad.mcap"
    write_mcap(path, bad_position=True)
    with pytest.raises(ValueError, match="7 finite values"):
        mcap.read_streams(path, get_task("yam_0012"))


def h264_samples():
    encoder = av.CodecContext.create("libx264", "w")
    encoder.width = encoder.height = 16
    encoder.pix_fmt = "yuv420p"
    encoder.time_base = Fraction(1, 30)
    encoder.options = {"preset": "ultrafast", "tune": "zerolatency", "g": "30", "bf": "0"}
    packets = []
    for i in range(5):
        frame = av.VideoFrame.from_ndarray(np.full((16, 16, 3), 20 + i * 30, dtype=np.uint8), format="rgb24")
        frame.pts = i
        packets.extend(encoder.encode(frame))
    packets.extend(encoder.encode(None))
    return [
        mcap.Sample(i * 33_333_334, {"data": base64.b64encode(bytes(packet)).decode(), "shape": [16, 16, 3]})
        for i, packet in enumerate(packets)
    ]


def test_continuous_h264_and_corruption():
    samples = h264_samples()
    frames = list(mcap.decode_camera(samples))
    assert len(frames) == 5
    assert all(frame.shape == (16, 16, 3) for frame in frames)
    assert frames[-1].mean() > frames[0].mean() + 100
    # A P frame cannot be independently decoded without its initial keyframe.
    with pytest.raises((ValueError, av.error.FFmpegError)):
        list(mcap.decode_camera(samples[1:]))
    samples[2].data["data"] = "not base64!"
    with pytest.raises(ValueError, match="base64"):
        list(mcap.decode_camera(samples))


def test_reject_incomplete_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))
    task = dataclasses.replace(get_task("yam_0012"), raw_dir=str(tmp_path), dataset_dir=str(tmp_path / "dataset"))
    task.dataset_path.mkdir(parents=True)
    (task.dataset_path / mcap.REPORT).write_text(json.dumps({"status": "incomplete"}))
    with pytest.raises(ValueError, match="Incomplete"):
        mcap.require_complete(task)


def test_output_manifest_ignores_hidden_sidecars(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "episode.parquet").write_bytes(b"episode")
    (data / ".episode.parquet.backup").write_bytes(b"episode")
    (tmp_path / ".metadata.tmp").write_bytes(b"metadata")
    (tmp_path / mcap.REPORT).write_text("{}")

    assert mcap.output_manifest(tmp_path) == [{"path": "data/episode.parquet", "size": 7}]


def test_failed_conversion_is_not_published(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path / "datasets"))
    task = dataclasses.replace(get_task("yam_0012"), raw_dir=str(tmp_path), dataset_dir=str(tmp_path / "dataset"))
    source = tmp_path / "broken.mcap"
    write_mcap(source, bad_position=True)
    with pytest.raises(RuntimeError, match=r"broken\.mcap"):
        mcap.convert(task)
    assert not task.dataset_path.exists()
    staging = task.dataset_path.with_name(task.dataset_path.name + ".incomplete")
    assert json.loads((staging / mcap.REPORT).read_text())["status"] == "incomplete"
    with pytest.raises(ValueError, match="Interrupted"):
        mcap.require_complete(task)
    with pytest.raises(ValueError, match="Interrupted"):
        mcap.convert(task)
