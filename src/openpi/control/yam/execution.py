"""One timed command writer attaching to an existing protocol-6 daemon."""

from __future__ import annotations

from collections import deque
import json
import threading
import time
from uuid import uuid4

import numpy as np


def sdk_types():
    from gello.safety.rpc import SafetyClient
    from gello.zmq_core.robot_node import ZMQClientRobot

    return SafetyClient, ZMQClientRobot


def daemon_status(profile, expected_owner, *, require_hold=True):
    safety_type, _ = sdk_types()
    client = safety_type(port=profile.network.safety_port, timeout_ms=500)
    try:
        status = client.status()
    finally:
        client.close()
    expected = {name: arm.can_channel for name, arm in profile.arms.items()}
    if (
        status.get("protocol_version") != 6
        or status.get("channels") != expected
        or status.get("hardware_port") != profile.network.hardware_port
        or status.get("repo_root") != str(expected_owner.resolve())
    ):
        raise ValueError("Daemon owner/protocol/arm configuration mismatch")
    if require_hold and (status.get("state") != "safe_hold" or status.get("hold_confirmed") is not True):
        raise RuntimeError("An existing confirmed SAFE_HOLD daemon is required")
    if status.get("hardware_health", {}).get("ok") is not True:
        raise RuntimeError("Daemon hardware health is not confirmed")
    return status


CONTROL_RATES = (1, 2, 5, 15, 30, 60, 90, 120)


def wire_rate(control_hz):
    if type(control_hz) is not int or control_hz not in CONTROL_RATES:
        raise ValueError("Unsupported target update frequency")
    return max(30, control_hz)


def command_repeats(control_hz):
    wire_rate(control_hz)
    return 30 // control_hz if control_hz < 30 else control_hz // 30


def model_step_period(control_hz):
    wire_rate(control_hz)
    return 1 / min(30, control_hz)


def resample_commands(initial, points, control_hz):
    """Low rates hold every model target; high rates interpolate at fixed duration.

    The wire always stays at >=30 Hz, preserving the daemon's timed deadline.
    Low-rate holds deliberately stretch model time, never discard model targets.
    """
    wire_rate(control_hz)
    points = np.asarray(points, float)
    initial = np.asarray(initial, float)
    if (
        initial.shape != (14,)
        or points.ndim != 2
        or points.shape[1] != 14
        or not len(points)
        or not np.isfinite(points).all()
        or not np.isfinite(initial).all()
    ):
        raise ValueError("Expected finite joints and a nonempty trajectory")
    ratio = command_repeats(control_hz)
    if control_hz < 30:
        return np.repeat(points, ratio, axis=0)
    previous = np.vstack([initial, points[:-1]])
    fractions = np.arange(1, ratio + 1) / ratio
    return (previous[:, None] + (points - previous)[:, None] * fractions[None, :, None]).reshape(-1, 14)


class CommandWriter:
    """Own all command, lease and motion state changes on one thread.

    Inference and IK run on the caller. Waiting never suspends the >=30 Hz timed
    command stream; each waiting tick repeats the last *accepted* target.
    """

    def __init__(
        self,
        profile,
        initial,
        limits,
        velocities,
        log_path,
        *,
        producer_timeout=45,
        control_hz=30,
        event_callback=None,
        factories=None,
    ):
        self.event_callback, self.factories = event_callback, factories
        self.wire_hz = wire_rate(control_hz)
        self.control_hz = control_hz
        self.profile = profile
        self.target = np.asarray(initial, float).copy()
        self.limits = np.asarray(limits, float)
        self.velocities = np.asarray(velocities, float)
        self.producer_timeout = producer_timeout
        self.producer_deadline = time.monotonic() + producer_timeout
        self.log_path = log_path
        self.queue = deque()
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.finished = threading.Event()
        self.error = None
        self.hold_status = None
        self.stop_reason = "normal completion"
        self.thread = threading.Thread(target=self._run, name="pi-command-writer", daemon=True)
        self.normal = False

    def start(self):
        self.thread.start()
        if not self.ready.wait(5):
            self.stop.set()
            raise TimeoutError("Command writer did not activate")
        self.check()

    def check(self):
        if self.error:
            raise RuntimeError("Command writer failed") from self.error
        if self.stop.is_set():
            raise RuntimeError("Command writer stopped")

    def touch(self):
        self.check()
        self.producer_deadline = time.monotonic() + self.producer_timeout

    def submit(self, points, kind):
        points = np.asarray(points, float)
        if points.ndim != 2 or points.shape[1] != 14 or not len(points) or not np.isfinite(points).all():
            raise ValueError("Expected a nonempty finite joint trajectory")
        if np.any(points < self.limits[:, 0] - 1e-6) or np.any(points > self.limits[:, 1] + 1e-6):
            raise ValueError("Trajectory exceeds joint limits")
        with self.lock:
            self.check()
            if self.queue:
                raise RuntimeError("A trajectory is still pending")
            points = resample_commands(self.target, points, self.control_hz)
            previous = np.vstack([self.target, points[:-1]])
            arm = [i for i in range(14) if i % 7 != 6]
            if np.any(np.abs(points[:, arm] - previous[:, arm]) * self.wire_hz > self.velocities[arm] + 1e-3):
                raise ValueError("Trajectory exceeds nominal joint speed")
            self.finished.clear()
            self.queue.extend((p.copy(), kind, i) for i, p in enumerate(points))
        self.touch()

    def wait(self, seconds):
        deadline = time.monotonic() + seconds
        while not self.finished.wait(0.05):
            self.touch()
            if time.monotonic() > deadline:
                raise TimeoutError("Trajectory execution timed out")
        self.check()

    def _run(self):
        profile = self.profile
        safety_type, robot_type = self.factories or sdk_types()
        safety = safety_type(port=profile.network.safety_port, timeout_ms=500)
        robot = robot_type(port=profile.network.hardware_port, timeout_ms=500)
        robot.command_period_s = 1 / self.wire_hz
        session = "pi-client-" + uuid4().hex
        begun = False
        try:
            safety.status()
            safety.begin_session(session)
            begun = True
            for name in ("parent", "collector"):
                safety.heartbeat(name, session)
            status = safety.activate(session)
            if status.get("state") != "active":
                raise RuntimeError("Daemon did not activate")
            # First repeat measured feedback after a full control period. Do not
            # admit a homing point while the daemon's first timed dt is tiny.
            with self.lock:
                self.target = np.asarray(robot.get_joint_state(), float)
            tick, last_heartbeat = time.monotonic() + 1 / self.wire_hz, 0.0
            with self.log_path.open("x") as log:
                while not self.stop.is_set():
                    self.stop.wait(max(0, tick - time.monotonic()))
                    if self.stop.is_set():
                        break
                    now = time.monotonic()
                    if now > self.producer_deadline:
                        raise TimeoutError("Inference/IK producer deadline expired")
                    if now - last_heartbeat >= 0.1:
                        for name in ("parent", "collector"):
                            status = safety.heartbeat(name, session)
                            if status.get("state") != "active":
                                raise RuntimeError("Daemon lost active session")
                        last_heartbeat = now
                    with self.lock:
                        point, kind, index = self.queue.popleft() if self.queue else (self.target.copy(), "hold", None)
                    # A timeout is never resent. The finally block queries status
                    # via a fresh RPC before requesting the terminal hold.
                    try:
                        reply = robot.command_joint_state(point)
                    except BaseException:
                        log.write(
                            json.dumps(
                                {
                                    "time_ns": time.time_ns(),
                                    "kind": kind,
                                    "requested": point.tolist(),
                                    "execution": "unknown",
                                }
                            )
                            + "\n"
                        )
                        log.flush()
                        raise
                    applied = np.asarray(reply["applied_joint_state"], float)
                    measured = np.asarray(robot.get_joint_state(), float)
                    if applied.shape != (14,) or measured.shape != (14,) or not np.isfinite([applied, measured]).all():
                        raise ValueError("Invalid daemon acknowledgement/feedback")
                    log.write(
                        json.dumps(
                            {
                                "time_ns": time.time_ns(),
                                "kind": kind,
                                "index": index,
                                "requested": point.tolist(),
                                "accepted": applied.tolist(),
                                "measured": measured.tolist(),
                                "accepted_time_ns": reply.get("accepted_time_ns"),
                                "target_update_hz": self.control_hz,
                                "wire_hz": self.wire_hz,
                                "limited": reply.get("limited", False),
                            }
                        )
                        + "\n"
                    )
                    log.flush()
                    if self.event_callback:
                        self.event_callback(
                            "ack",
                            accepted=applied.tolist(),
                            measured=measured.tolist(),
                            time_ns=time.time_ns(),
                            event_monotonic=time.monotonic(),
                        )
                    if reply.get("limited") or not np.allclose(applied, point, atol=1e-7, rtol=0):
                        raise RuntimeError("Daemon changed/rejected the requested target")
                    with self.lock:
                        self.target = applied.copy()
                        if not self.queue:
                            self.finished.set()
                    self.ready.set()
                    tick += 1 / self.wire_hz
                    if tick <= time.monotonic():
                        tick = time.monotonic() + 1 / self.wire_hz
        except BaseException as error:
            self.error = error
            self.stop_reason = f"{type(error).__name__}: {error}"
        finally:
            self.ready.set()
            try:
                if begun:
                    observed = safety.status()
                    if self.normal and self.error is None and observed.get("supports_finish_motion"):
                        try:
                            safety._call("finish_motion", session_id=session)  # noqa: SLF001 - protocol-6 SDK RPC
                        except Exception:
                            safety.safe_hold("pi client normal completion: endpoint not settled")
                    else:
                        safety.safe_hold("pi client ended: " + self.stop_reason)
                    self.hold_status = safety.status()
                    if self.hold_status.get("state") != "safe_hold" or not self.hold_status.get("hold_confirmed"):
                        raise RuntimeError("Final SAFE_HOLD was not confirmed")
            except BaseException as error:
                self.error = self.error or error
            robot.close()
            safety.close()
            self.stop.set()

    def close(self, *, normal=False, reason="client shutdown"):
        self.normal, self.stop_reason = normal, reason
        self.stop.set()
        if self.thread.ident is None:
            return
        self.thread.join(5)
        if self.thread.is_alive():
            raise RuntimeError("Command writer did not stop")
        if self.error:
            raise RuntimeError("Client ended with a control failure") from self.error
