"""Control boundaries exercised with fake RPCs, never real hardware."""

import hashlib
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from openpi.control.app import Console
from openpi.control.yam import live_worker
from openpi.control.yam.execution import CONTROL_RATES
from openpi.control.yam.execution import CommandWriter
from openpi.control.yam.execution import model_step_period
from openpi.control.yam.execution import resample_commands
from openpi.control.yam.execution import wire_rate
from openpi.control.yam.live_kinematics import LiveKinematics
from openpi.control.yam.mapping import IKFailure
from openpi.control.yam.references import RecordedStart
from openpi.control.yam.references import slow_ramp
from openpi.control.yam.references import support_references


@pytest.mark.parametrize("rate", CONTROL_RATES)
def test_control_rates_keep_every_model_endpoint(rate):
    points = np.arange(1, 4)[:, None] * np.ones((1, 14)) * 0.01
    expanded = resample_commands(np.zeros(14), points, rate)
    ratio = max(30 // rate, rate // 30)
    np.testing.assert_allclose(expanded[ratio - 1 :: ratio], points)
    assert len(expanded) / wire_rate(rate) == pytest.approx(3 * model_step_period(rate))
    assert wire_rate(rate) >= 30
    Console.validate_settings({"horizon": 30, "execute_steps": 15, "max_chunks": 120, "control_hz": rate}, mode="live")


def test_recorded_start_uses_raw_joints_and_checks_source_identity(tmp_path):
    file = tmp_path / "demo.mcap"
    file.write_bytes(b"raw-support")
    row = {"path": file, "sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
    serials = {"main": "main", "left_wrist": "left", "right_wrist": "right"}

    def messages(_file, topic, **_kwargs):
        side = topic.split("/")[1]
        if side == "cameras":
            camera = topic.split("/")[2]
            yield topic, 123, {"serial": camera, "_pixels": np.zeros((480, 640, 3), np.uint8)}
        else:
            yield topic, 123, {"position": list(range(7)) if side == "left" else list(range(7, 14))}

    reference = RecordedStart(row, serials, messages)
    np.testing.assert_array_equal(reference.target, np.arange(14))
    file.write_bytes(b"replaced")
    with pytest.raises(ValueError, match="identity"):
        RecordedStart(row, serials, messages)


def test_only_target_support_is_eligible_for_overlay(tmp_path):
    target = tmp_path / "checkpoint/assets/yam_hybrid/deployment/target"
    target.mkdir(parents=True)
    recordings = tmp_path / "recordings/task/mcap/success"
    recordings.mkdir(parents=True)
    (recordings / "support.mcap").touch()
    clip = {
        "split": "target_support",
        "machine": "yam/yam1",
        "original_id": "support",
        "pairing": "cup",
        "export": {
            "provenance": {"task_instruction": "pour water", "source_record": {"input": {"source_sha256": "abc"}}}
        },
    }
    # Non-support clips deliberately do not have usable paths or payloads.
    (target / "clip-geometry.json").write_text(
        json.dumps({"clips": [clip, {"split": "locked_test"}, {"split": "dev"}]})
    )
    rows = support_references(tmp_path / "checkpoint", tmp_path / "recordings")
    assert [row["original_id"] for row in rows] == ["support"]


def test_return_ramp_is_slow_bounded_and_ends_at_recorded_target():
    target = np.full(14, 0.51)
    points = slow_ramp(np.zeros(14), target, np.tile([-1.0, 1.0], (14, 1)))
    steps = np.diff(np.vstack([np.zeros(14), points]), axis=0)
    assert np.max(np.abs(steps[:, :6])) <= 0.005 + 1e-9
    np.testing.assert_allclose(points[-1], target)
    with pytest.raises(ValueError, match="target"):
        slow_ramp(np.zeros(14), target * 3, np.tile([-1.0, 1.0], (14, 1)))


def test_unknown_ack_is_never_retried_and_finishes_in_hold(tmp_path):
    calls = []
    state = {"state": "safe_hold", "hold_confirmed": True}

    class Safety:
        def __init__(self, **_kwargs):
            pass

        def status(self):
            return state.copy()

        def begin_session(self, _session):
            calls.append("begin")

        def heartbeat(self, *_args):
            return state.copy()

        def activate(self, _session):
            state.update(state="active", hold_confirmed=False)
            return state.copy()

        def safe_hold(self, _reason):
            calls.append("safe_hold")
            state.update(state="safe_hold", hold_confirmed=True)

        def close(self):
            pass

    class Robot:
        def __init__(self, **_kwargs):
            pass

        def get_joint_state(self):
            return np.zeros(14)

        def command_joint_state(self, _point):
            calls.append("command")
            raise TimeoutError("ACK unknown")

        def close(self):
            pass

    writer = CommandWriter(
        SimpleNamespace(network=SimpleNamespace(safety_port=6002, hardware_port=6001)),
        np.zeros(14),
        np.tile([-1, 1], (14, 1)),
        np.ones(14),
        tmp_path / "commands.jsonl",
        factories=(Safety, Robot),
    )
    with pytest.raises(RuntimeError):
        writer.start()
    with pytest.raises(RuntimeError):
        writer.close()
    assert calls == ["begin", "command", "safe_hold"]
    assert writer.hold_status == {"state": "safe_hold", "hold_confirmed": True}
    assert json.loads((tmp_path / "commands.jsonl").read_text())["execution"] == "unknown"


def test_live_shadow_has_no_command_writer_and_uses_fresh_ik_feedback(tmp_path, monkeypatch):
    engine = live_worker.LiveEngine.__new__(live_worker.LiveEngine)
    engine.args = SimpleNamespace(output=tmp_path, owner=Path("/sdk"))
    engine.profile = object()
    engine.checkpoint = tmp_path / "checkpoint"
    engine.guard = lambda: None
    engine.lock = threading.RLock()
    engine.reference = SimpleNamespace(info={"sha256": "raw"}, row={"prompt": "pour water"})
    engine.bundle = SimpleNamespace(
        skew_s=0.001,
        frames={name: SimpleNamespace(rgb=np.zeros((2, 2, 3), np.uint8)) for name in ("left", "main", "right")},
    )
    feedback = iter([np.zeros(14), np.ones(14) * 0.1, np.ones(14) * 0.2])
    engine.joints = lambda: next(feedback)
    seeds, events = [], []
    engine.emit = lambda kind, **data: events.append((kind, data))
    engine.adapter = SimpleNamespace(
        reset=lambda q: None,
        set_tolerances=lambda position, rotation: None,
        bridge=SimpleNamespace(contract={}, snapshot_anchors=dict, validate_policy_metadata=lambda metadata: None),
        observation=lambda q, images, prompt: {"state": np.zeros(22)},
        plan=lambda actions, q, steps, **kwargs: seeds.append(q.copy()) or {"command_targets": np.zeros((steps, 14))},
    )

    class Connection:
        def __init__(self, *_args, **_kwargs):
            self.metadata = {"model_sha256": "model"}

        def infer(self, _observation):
            return np.zeros((30, 14)), {}

        def close(self):
            pass

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Read-only operation constructed a command writer")

    monkeypatch.setattr(live_worker, "PolicyConnection", Connection)
    monkeypatch.setattr(live_worker, "CommandWriter", forbidden)
    monkeypatch.setattr(live_worker, "daemon_status", lambda *_args, **_kwargs: {})
    result = engine.infer(
        {
            "url": "ws://test",
            "model_sha256": "model",
            "settings": {
                "horizon": 30,
                "execute_steps": 15,
                "max_chunks": 120,
                "control_hz": 30,
                "velocity_mode": "urdf",
                "position_tolerance_mm": 5,
                "rotation_tolerance_rad": 0.01,
            },
        },
        motion=False,
    )
    assert result["hardware_motion"] is False
    assert result["status"] == "complete"
    assert len(result["requests"]) == 1
    np.testing.assert_allclose(seeds[0], 0.2)
    assert not list(tmp_path.rglob("commands.jsonl"))


def test_camera_or_browser_expiry_cancels_live_operation():
    engine = live_worker.LiveEngine.__new__(live_worker.LiveEngine)
    engine.cancel, engine.closed = threading.Event(), threading.Event()
    engine.heartbeat, engine.bundle_time = time.monotonic(), time.monotonic() - 2
    with pytest.raises(TimeoutError, match="camera"):
        engine.guard()
    engine.bundle_time, engine.heartbeat = time.monotonic(), time.monotonic() - 4
    with pytest.raises(TimeoutError, match="heartbeat"):
        engine.guard()


def test_readonly_return_plan_uses_native_daemon_limits_and_preserves_target(tmp_path, monkeypatch):
    engine = live_worker.LiveEngine.__new__(live_worker.LiveEngine)
    engine.args = SimpleNamespace(output=tmp_path, owner=Path("/sdk"))
    engine.profile = object()
    engine.guard = lambda: None
    current = np.zeros(14)
    current[8] = -0.00019  # Native follower zero calibration, within daemon limits.
    target = np.ones(14) * 0.02
    engine.joints = current.copy
    engine.reference = SimpleNamespace(target=target, info={"sha256": "raw"})
    engine.adapter = SimpleNamespace(
        left=SimpleNamespace(bounds=np.tile([0, 1], (6, 1))),
        right=SimpleNamespace(bounds=np.tile([0, 1], (6, 1))),
        calibration=SimpleNamespace(model_sha256="model"),
    )
    monkeypatch.setattr(live_worker, "daemon_status", lambda *_args: {"joint_limits": np.tile([-0.15, 1], (14, 1))})
    result = engine.make_plan()
    assert result["hardware_motion"] is False
    np.testing.assert_allclose(engine.plan["points"][-1], target)
    assert len(list((tmp_path / "plans").glob("*.json"))) == 1
    assert not list(tmp_path.rglob("commands.jsonl"))
    assert engine.bounds({"joint_limits": np.tile([-0.15, 1], (14, 1))})[8, 0] == 0


def test_stale_return_plan_is_rejected_before_writer_construction(tmp_path, monkeypatch):
    engine = live_worker.LiveEngine.__new__(live_worker.LiveEngine)
    engine.aligned, engine.skipped = True, False
    engine.args = SimpleNamespace(owner=Path("/sdk"))
    engine.profile = object()
    file = tmp_path / "raw.mcap"
    file.write_bytes(b"demo")
    engine.reference = SimpleNamespace(row={"path": file, "sha256": hashlib.sha256(b"demo").hexdigest()})
    engine.plan = {"public": {"plan_id": "old", "current": [0] * 14}, "created": time.monotonic() - 61}
    engine.joints = lambda: np.zeros(14)
    monkeypatch.setattr(live_worker, "daemon_status", lambda *_args: {})

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Stale plan constructed a command writer")

    monkeypatch.setattr(live_worker, "CommandWriter", forbidden)
    with pytest.raises(ValueError, match="Feedback/plan changed"):
        engine.prepare({"plan_id": "old"})


def test_relaxed_ik_still_optimizes_and_records_explicit_tolerances():
    kin = LiveKinematics.__new__(LiveKinematics)
    kin.set_tolerances(0.005, 0.01)
    with pytest.raises(ValueError, match="approved"):
        kin.set_tolerances(0.006, 0.01)
    iterations = []
    kin.bounds = np.tile([-1, 1], (6, 1))
    kin.configuration = SimpleNamespace(
        q=np.zeros(6), update=lambda q: None, integrate_inplace=lambda velocity, dt: iterations.append(dt)
    )
    kin.task = SimpleNamespace(set_target=lambda target: None)
    kin.mink = SimpleNamespace(
        SE3=SimpleNamespace(from_matrix=lambda target: target), solve_ik=lambda *_args, **_kwargs: np.zeros(6)
    )
    kin.limits = []
    kin.fk = lambda q: np.eye(4)
    target = np.eye(4)
    target[0, 3] = 0.001
    _, report = kin.solve(target, np.zeros(6))
    assert len(iterations) == 3
    assert report["iterations"] == 3
    assert report["position_error_m"] == 0.001
    assert report["position_tolerance_m"] == 0.005
    kin.set_tolerances(0.0001, 0.0001)
    with pytest.raises(IKFailure):
        kin.solve(target, np.zeros(6), max_iters=3)
