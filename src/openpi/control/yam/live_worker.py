"""YAM live camera, overlay, planned return and policy execution worker.

Only explicit prepare/policy requests construct a command writer. Preview,
planning and shadow requests use read-only RPC on the existing daemon.
"""

import argparse
import io
import json
from pathlib import Path
import signal
import sys
import threading
import time
import traceback
from uuid import uuid4

import numpy as np
from PIL import Image

from openpi.control.yam.execution import CommandWriter
from openpi.control.yam.execution import daemon_status
from openpi.control.yam.execution import model_step_period
from openpi.control.yam.live_kinematics import LiveActionAdapter
from openpi.control.yam.references import RecordedStart
from openpi.control.yam.references import sha
from openpi.control.yam.references import slow_ramp
from openpi.control.yam.references import support_references
from openpi.control.yam.wire import PolicyConnection


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False))
    temporary.replace(path)


class LiveEngine:
    def __init__(self, args, emit):
        sys.path.insert(0, str(args.client_root))
        from gello.cameras.realsense_rig import CameraRig
        from gello.cameras.realsense_rig import load_camera_rig_config
        from gello.data_utils.mcap_tools import messages
        from gello.profile import load_profile
        from gello.zmq_core.robot_node import ZMQClientRobot

        self.args, self.emit, self.messages, self.robot_type = args, emit, messages, ZMQClientRobot
        self.profile = load_profile(args.profile)
        if self.profile.network.host != "127.0.0.1":
            raise ValueError("Live control requires the local daemon")
        self.lock, self.operation = threading.RLock(), threading.Lock()
        self.cancel, self.closed = threading.Event(), threading.Event()
        self.heartbeat = time.monotonic()
        self.writer, self.reference, self.plan = None, None, None
        self.aligned, self.skipped = False, False
        self.bundle, self.bundle_time = None, float("-inf")
        self.state = {"ready": False, "phase": "connecting", "hardware_motion": False}
        self.args.output.mkdir(parents=True, exist_ok=True)
        self.configure(args.checkpoint)
        config = load_camera_rig_config(self.profile.cameras.config_path)
        for ui, key in [("left", "left_wrist"), ("main", "main"), ("right", "right_wrist")]:
            spec = config.cameras[ui]
            if spec.serial != self.adapter.calibration.camera_serials[key] or spec.rotation != 0:
                raise ValueError("Camera serial/rotation differs from the bound YAM-1 recording")
            if (spec.width, spec.height) != (640, 480):
                raise ValueError("Camera dimensions differ from the bound calibration")
        self.rig = CameraRig(config, depth_enabled=False)
        self.rig.start(timeout_s=15)
        self.preview = threading.Thread(target=self._preview, name="yam-live-preview", daemon=True)
        self.preview.start()

    def configure(self, checkpoint):
        checkpoint = Path(checkpoint).resolve()
        adapter = LiveActionAdapter(checkpoint, self.args.output / "numerical-models")
        references = support_references(checkpoint, self.args.recordings)
        reference = RecordedStart(references[0], adapter.calibration.camera_serials, self.messages)
        with self.lock:
            self.checkpoint, self.adapter, self.references = checkpoint, adapter, references
            self.reference, self.reference_index = reference, 0
            self.plan, self.aligned, self.skipped = None, False, False
        self._reference_images()
        return self.reference.info

    def select_reference(self, index):
        if type(index) is not int or not 0 <= index < len(self.references):
            raise ValueError("Invalid demonstration index")
        reference = RecordedStart(self.references[index], self.adapter.calibration.camera_serials, self.messages)
        with self.lock:
            self.reference, self.reference_index = reference, index
            self.plan, self.aligned, self.skipped = None, False, False
        self._reference_images()
        return reference.info

    def _reference_images(self):
        for name, image in self.reference.images.items():
            self._image("reference-" + name, image)

    def _image(self, name, pixels):
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format="JPEG", quality=88)
        path = self.args.output / (name + ".jpg")
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(buffer.getvalue())
        temporary.replace(path)

    def robot(self):
        return self.robot_type(host=self.profile.network.host, port=self.profile.network.hardware_port, timeout_ms=500)

    def joints(self):
        robot = self.robot()
        try:
            value = np.asarray(robot.get_joint_state(), float)
        finally:
            robot.close()
        if value.shape != (14,) or not np.isfinite(value).all():
            raise ValueError("Invalid native follower feedback")
        return value

    def _preview(self):
        last_status = 0.0
        try:
            while not self.closed.wait(0.015):
                now = time.monotonic()
                sample = self.rig.sample(now)
                if hasattr(sample, "frames"):
                    with self.lock:
                        self.bundle = sample
                        self.bundle_time = min(frame.host_timestamp for frame in sample.frames.values())
                if now - last_status < 0.12:
                    continue
                last_status = now
                try:
                    q = self.joints()
                    status = daemon_status(self.profile, self.args.owner, require_hold=False)
                    if self.bundle:
                        for name, frame in self.bundle.frames.items():
                            self._image("camera-" + name, frame.rgb)
                    with self.lock:
                        age = now - self.bundle_time
                        value = dict(
                            self.state,
                            updated_monotonic=now,
                            camera_age_s=age if np.isfinite(age) else None,
                            camera_skew_s=self.bundle.skew_s if self.bundle else None,
                            ready=bool(age < 1),
                            joints=q.tolist(),
                            daemon=status,
                            reference=self.reference.info,
                            reference_index=self.reference_index,
                            references=[
                                {k: str(v) if isinstance(v, Path) else v for k, v in x.items()} for x in self.references
                            ],
                            aligned=self.aligned,
                            alignment_skipped=self.skipped,
                            plan=self.plan["public"] if self.plan else None,
                            plan_valid=bool(
                                self.plan
                                and now - self.plan["created"] < 60
                                and np.max(np.abs(q - self.plan["public"]["current"])) <= 0.02
                            ),
                            checkpoint=str(self.checkpoint),
                            model_sha256=self.adapter.calibration.model_sha256,
                        )
                        self.state.pop("error", None)
                except Exception as error:
                    value = dict(
                        self.state,
                        updated_monotonic=now,
                        ready=False,
                        error=str(error),
                        reference=self.reference.info,
                        reference_index=self.reference_index,
                        references=[
                            {k: str(v) if isinstance(v, Path) else v for k, v in x.items()} for x in self.references
                        ],
                        aligned=self.aligned,
                        alignment_skipped=self.skipped,
                        plan=self.plan["public"] if self.plan else None,
                    )
                atomic_json(self.args.output / "status.json", value)
        except BaseException:
            traceback.print_exc()
            self.cancel.set()

    def guard(self):
        if self.cancel.is_set() or self.closed.is_set():
            raise InterruptedError("Operation stopped")
        if time.monotonic() - self.heartbeat > 3:
            raise TimeoutError("Browser/parent heartbeat expired")
        if time.monotonic() - self.bundle_time > 1:
            raise TimeoutError("Live camera observation expired")

    def bounds(self, status, *, nominal=True):
        urdf = np.vstack([self.adapter.left.bounds, [[0.0, 1.0]], self.adapter.right.bounds, [[0.0, 1.0]]])
        reported = np.asarray(status["joint_limits"], float)
        if reported.shape != (14, 2) or not np.isfinite(reported).all():
            raise ValueError("Invalid daemon joint limits")
        # Recorded return targets are native follower joints, checked against
        # the calibrated daemon limits. URDF bounds apply to policy IK.
        bounds = (
            np.c_[np.maximum(urdf[:, 0], reported[:, 0]), np.minimum(urdf[:, 1], reported[:, 1])]
            if nominal
            else reported
        )
        if np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("Daemon and checkpoint joint limits disagree")
        return bounds

    def make_plan(self):
        self.guard()
        status = daemon_status(self.profile, self.args.owner)
        current = self.joints()
        points = slow_ramp(current, self.reference.target, self.bounds(status, nominal=False))
        public = {
            "plan_id": uuid4().hex,
            "execution_duration_s": len(points) / 30,
            "target": self.reference.target.tolist(),
            "current": current.tolist(),
            "source": self.reference.info,
            "hardware_motion": False,
            "model_sha256": self.adapter.calibration.model_sha256,
        }
        self.plan = {"public": public, "points": points, "created": time.monotonic()}
        directory = self.args.output / "plans"
        directory.mkdir(exist_ok=True)
        atomic_json(directory / (public["plan_id"] + ".json"), dict(public, command_targets=points.tolist()))
        return public

    def _wait(self, writer, seconds, *, target=None):
        deadline, bad = time.monotonic() + seconds, 0
        arms = [i for i in range(14) if i not in (6, 13)]
        while True:
            self.guard()
            writer.touch()
            measured = self.joints()
            with writer.lock:
                accepted = writer.target.copy()
            error = float(np.max(np.abs(measured[arms] - accepted[arms])))
            bad = bad + 1 if error > 0.25 else 0
            if bad >= 5:
                raise RuntimeError("Tracking error exceeds 0.25rad for five checks")
            if writer.finished.is_set():
                writer.check()
                if target is None or np.max(np.abs(measured[arms] - target[arms])) <= self.args.arrival_tolerance:
                    return
            if time.monotonic() > deadline:
                raise TimeoutError("Motion/endpoint convergence timed out")
            time.sleep(0.03)

    def watch(self, done):
        while not done.wait(0.05):
            try:
                self.guard()
            except BaseException:
                self.cancel.set()
                writer = self.writer
                if writer is not None:
                    writer.stop.set()
                return

    def prepare(self, payload):
        if not self.aligned or self.skipped or not self.plan:
            raise ValueError("Confirm the overlay scene and run read-only start planning first")
        if payload.get("plan_id") != self.plan["public"]["plan_id"]:
            raise ValueError("Return plan changed")
        if sha(self.reference.row["path"]) != self.reference.row["sha256"]:
            raise ValueError("Recorded source changed")
        status = daemon_status(self.profile, self.args.owner)
        current = self.joints()
        if (
            np.max(np.abs(current - self.plan["public"]["current"])) > 0.02
            or time.monotonic() - self.plan["created"] > 60
        ):
            raise ValueError("Feedback/plan changed; repeat start planning")
        points = slow_ramp(current, self.reference.target, self.bounds(status, nominal=False))
        output = self.args.output / "runs" / ("prepare-" + uuid4().hex)
        output.mkdir(parents=True)
        velocities = np.full(14, 0.15)
        velocities[[6, 13]] = np.inf
        writer = CommandWriter(
            self.profile,
            current,
            self.bounds(status, nominal=False),
            velocities,
            output / "commands.jsonl",
            event_callback=self.emit,
        )
        self.writer = writer
        normal = False
        try:
            self.guard()
            writer.start()
            with writer.lock:
                start = writer.target.copy()
            points = slow_ramp(start, self.reference.target, self.bounds(status, nominal=False))
            writer.submit(points, "recorded-start")
            self._wait(writer, len(points) / 30 + 8, target=self.reference.target)
            normal = True
            return {"status": "complete", "output": str(output), "target": self.reference.target.tolist()}
        finally:
            try:
                writer.close(normal=normal, reason="recorded-start operation ended")
                atomic_json(
                    output / "report.json",
                    {
                        "hardware_motion": True,
                        "normal": normal,
                        "source": self.reference.info,
                        "hold_status": writer.hold_status,
                    },
                )
            finally:
                self.writer = None
                self.plan = None

    def infer(self, payload, *, motion):
        settings = payload["settings"]
        self.adapter.set_tolerances(settings["position_tolerance_mm"] / 1000, settings["rotation_tolerance_rad"])
        if motion and not (self.aligned or self.skipped):
            raise ValueError("Confirm scene overlay or explicitly choose current-pose start")
        self.guard()
        status = daemon_status(self.profile, self.args.owner)
        q = self.joints()
        self.adapter.reset(q)
        output = self.args.output / "runs" / (("policy-" if motion else "shadow-") + uuid4().hex)
        output.mkdir(parents=True)
        connection = PolicyConnection(payload["url"], self.adapter.bridge.contract, self.checkpoint, timeout=45)
        writer, normal = None, False
        report = {
            "hardware_motion": motion,
            "checkpoint": str(self.checkpoint),
            "metadata": connection.metadata,
            "settings": settings,
            "starting_joints": q.tolist(),
            "anchors": self.adapter.bridge.snapshot_anchors(),
            "source": self.reference.info,
            "requests": [],
        }
        try:
            self.adapter.bridge.validate_policy_metadata(connection.metadata)
            if connection.metadata.get("model_sha256") != payload["model_sha256"]:
                raise ValueError("Selected model changed")
            if motion:
                from openpi.control.yam.reference.legacy_limits import collection_speed_limits

                velocity = (
                    self.adapter.left.velocities
                    if settings["velocity_mode"] == "urdf"
                    else np.full(6, collection_speed_limits(self.args.owner)["arm_velocity_limit_rad_s"])
                )
                velocities = np.r_[velocity, np.inf, velocity, np.inf]
                writer = CommandWriter(
                    self.profile,
                    q,
                    self.bounds(status),
                    velocities,
                    output / "commands.jsonl",
                    control_hz=settings["control_hz"],
                    event_callback=self.emit,
                )
                self.writer = writer
                writer.start()
            self.emit("start", hardware_motion=motion, settings=settings)
            chunks = settings["max_chunks"] if motion else 1
            for index in range(chunks):
                self.guard()
                q = self.joints()
                with self.lock:
                    bundle = self.bundle
                    images = {
                        key: bundle.frames[ui].rgb.copy()
                        for ui, key in [("left", "left_wrist"), ("main", "main"), ("right", "right_wrist")]
                    }
                observation = self.adapter.observation(q, images, self.reference.row["prompt"])
                self.emit("decision", index=index)
                started = time.monotonic()
                actions, timing = connection.infer(observation)
                self.guard()
                self.emit(
                    "prediction",
                    index=index,
                    actions=actions.tolist(),
                    horizon=settings["horizon"],
                    execute_steps=settings["execute_steps"],
                    inference_s=time.monotonic() - started,
                )
                np.savez_compressed(
                    output / f"prediction-{index:04d}.npz",
                    actions=actions,
                    state=observation["state"],
                    joint_feedback=q,
                    **{"image_" + k: v for k, v in images.items()},
                )
                seed = self.joints()
                plan = self.adapter.plan(
                    actions,
                    seed,
                    settings["execute_steps"],
                    velocity_mode=settings["velocity_mode"],
                    yam_owner=self.args.owner,
                )
                points = plan["command_targets"]
                self.emit(
                    "ik", index=index, points=points.tolist(), step_period_s=model_step_period(settings["control_hz"])
                )
                report["requests"].append(
                    {
                        "index": index,
                        "finite": True,
                        "shape": list(actions.shape),
                        "ik_passed": True,
                        "camera_skew_s": bundle.skew_s,
                        "server_timing": timing,
                        "ik_diagnostics": plan.get("ik"),
                    }
                )
                if writer:
                    self.guard()
                    writer.submit(points, f"policy-{index:04d}")
                    self.emit(
                        "submit",
                        index=index,
                        points=points.tolist(),
                        control_hz=settings["control_hz"],
                        step_period_s=model_step_period(settings["control_hz"]),
                    )
                    self._wait(writer, len(points) * model_step_period(settings["control_hz"]) + 8)
                self.emit("chunk_complete", index=index)
                atomic_json(output / "report.json", report)
            normal = True
            report["status"] = "complete"
            return dict(report, output=str(output))
        except BaseException as error:
            report.update(status="failed", error=str(error), diagnostics=getattr(error, "diagnostics", None))
            raise
        finally:
            connection.close()
            try:
                if writer:
                    writer.close(normal=normal, reason=report.get("error", "policy completed"))
                    report["final_hold"] = writer.hold_status
            finally:
                self.writer = None
                self.emit("end", output=str(output))
                atomic_json(output / "report.json", report)

    def request(self, action, payload):
        if action in {"prepare", "shadow", "policy"}:
            self.state.update(
                phase={"prepare": "preparing", "shadow": "shadow", "policy": "policy"}[action],
                hardware_motion=action in {"prepare", "policy"},
            )
        if action == "configure":
            return self.configure(payload["checkpoint"])
        if action == "reference":
            return self.select_reference(payload["index"])
        if action == "align":
            self.aligned = bool(payload["aligned"])
            return self.aligned
        if action == "skip_alignment":
            self.skipped, self.aligned = bool(payload["skip"]), False
            self.plan = None
            return self.skipped
        if action == "plan":
            return self.make_plan()
        if action == "prepare":
            return self.prepare(payload)
        if action in {"shadow", "policy"}:
            return self.infer(payload, motion=action == "policy")
        raise ValueError("Unknown live operation")

    def close(self):
        self.closed.set()
        self.cancel.set()
        if self.writer:
            self.writer.stop.set()
        if hasattr(self, "preview"):
            self.preview.join(timeout=3)
        self.rig.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["checkpoint", "profile", "client-root", "recordings", "output", "owner"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--arrival-tolerance", type=float, default=0.02)
    args = parser.parse_args()
    wire, wire_lock = sys.stdout, threading.Lock()
    sys.stdout = sys.stderr  # SDK diagnostics must not enter the JSON IPC channel.

    def send(message):
        with wire_lock:
            wire.write(json.dumps(message, allow_nan=False) + "\n")
            wire.flush()

    def emit(kind, **data):
        data.setdefault("event_monotonic", time.monotonic())
        send({"event": kind, "data": data})

    engine = LiveEngine(args, emit)
    jobs = []

    def stop(*_):
        engine.cancel.set()
        if engine.writer:
            engine.writer.stop.set()
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)

    def run(request):
        request_id = request.get("id")
        if not engine.operation.acquire(blocking=False):
            if request_id:
                send({"id": request_id, "error": "Another live operation is running"})
            return
        done = threading.Event()
        engine.cancel.clear()
        watch = threading.Thread(target=engine.watch, args=(done,), daemon=True)
        if request["action"] in {"prepare", "policy", "shadow"}:
            watch.start()
        try:
            result = engine.request(request["action"], request.get("payload", {}))
            if request_id:
                send({"id": request_id, "result": result})
        except BaseException as error:
            traceback.print_exc()
            if request_id:
                send({"id": request_id, "error": f"{type(error).__name__}: {error}"})
        finally:
            done.set()
            engine.state.update(phase="preview", hardware_motion=False)
            engine.operation.release()

    try:
        for line in sys.stdin:
            request = json.loads(line)
            if request["action"] == "heartbeat":
                engine.heartbeat = time.monotonic()
                continue
            if request["action"] == "stop":
                engine.cancel.set()
                if engine.writer:
                    engine.writer.stop.set()
                continue
            job = threading.Thread(target=run, args=(request,), daemon=True)
            job.start()
            jobs.append(job)
            jobs = [job for job in jobs if job.is_alive()]
    except KeyboardInterrupt:
        pass
    finally:
        engine.cancel.set()
        if engine.writer:
            engine.writer.stop.set()
        for job in jobs:
            job.join(timeout=5)
        engine.close()


if __name__ == "__main__":
    main()
