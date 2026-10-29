"""Six-joint, checkpoint-defined FK and continuous constrained IK."""

from __future__ import annotations

from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np


class IKFailure(ValueError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


class CheckpointKinematics:
    def __init__(self, urdf: Path):
        import mink
        import mujoco

        self.mj, self.mink = mujoco, mink
        root = ET.parse(urdf).getroot()
        self.bounds, self.velocities = [], []
        for i in range(1, 7):
            joint = root.find(f"joint[@name='joint{i}']")
            if joint is None or joint.get("type") != "revolute":
                raise ValueError(f"Missing expected joint{i}")
            limit = joint.find("limit")
            self.bounds.append([float(limit.get("lower")), float(limit.get("upper"))])
            self.velocities.append(float(limit.get("velocity")))
        self.bounds = np.asarray(self.bounds)
        self.velocities = np.asarray(self.velocities)
        for joint in root.findall("joint"):
            if joint.get("type") == "prismatic":
                joint.set("type", "fixed")
                for tag in ("axis", "limit"):
                    child = joint.find(tag)
                    if child is not None:
                        joint.remove(child)
        spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
        spec.body("gripper").add_site(name="policy_ee_mount", pos=[0, 0, 0], quat=[1, 0, 0, 0])
        self.model = spec.compile()
        if self.model.nq != 6 or self.model.nv != 6:
            raise ValueError("Policy IK requires exactly six arm DOFs")
        names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(6)]
        if names != [f"joint{i}" for i in range(1, 7)]:
            raise ValueError("Unexpected MuJoCo joint order")
        self.data = mujoco.MjData(self.model)
        self.site = self.model.site("policy_ee_mount").id
        self.configuration = mink.Configuration(self.model)
        self.task = mink.FrameTask("policy_ee_mount", "site", position_cost=1.0, orientation_cost=1.0, lm_damping=1.0)
        self.limits = [mink.ConfigurationLimit(self.model)]

    def fk(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("Expected six finite arm joints")
        self.data.qpos[:] = q
        self.mj.mj_forward(self.model, self.data)
        result = np.eye(4)
        result[:3, 3] = self.data.site_xpos[self.site]
        result[:3, :3] = self.data.site_xmat[self.site].reshape(3, 3)
        return result

    def solve(self, target, seed, *, max_iters=200):
        from scipy.spatial.transform import Rotation

        target, seed = np.asarray(target, float), np.asarray(seed, float)
        if target.shape != (4, 4) or not np.isfinite(target).all():
            raise ValueError("Invalid IK target")
        self.configuration.update(np.clip(seed, self.bounds[:, 0], self.bounds[:, 1]))
        self.task.set_target(self.mink.SE3.from_matrix(target))
        started = time.monotonic()
        for iteration in range(max_iters + 1):
            q = self.configuration.q.copy()
            actual = self.fk(q)
            pe = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
            re = float(Rotation.from_matrix(actual[:3, :3] @ target[:3, :3].T).magnitude())
            if pe <= 1e-4 and re <= 1e-4:
                if np.any(q < self.bounds[:, 0] - 1e-6) or np.any(q > self.bounds[:, 1] + 1e-6):
                    raise ValueError("IK exceeded joint limits")
                return q, {
                    "iterations": iteration,
                    "position_error_m": pe,
                    "rotation_error_rad": re,
                    "seconds": time.monotonic() - started,
                }
            if iteration == max_iters:
                break
            v = self.mink.solve_ik(self.configuration, [self.task], 0.01, "quadprog", damping=1e-4, limits=self.limits)
            self.configuration.integrate_inplace(v, 0.01)
        raise IKFailure(
            f"IK did not converge: position={pe:.6g} m rotation={re:.6g} rad",
            {
                "position_error_m": pe,
                "rotation_error_rad": re,
                "seconds": time.monotonic() - started,
                "iterations": max_iters,
                "last_q": q.tolist(),
                "seed": seed.tolist(),
                "joint_limits": self.bounds.tolist(),
            },
        )

    def sequence(self, targets, current, *, check_speed=True, velocity_limits=None):
        current = np.asarray(current, float).copy()
        velocity_limits = self.velocities if velocity_limits is None else np.asarray(velocity_limits, float)
        if velocity_limits.shape != (6,) or not np.isfinite(velocity_limits).all() or np.any(velocity_limits <= 0):
            raise ValueError("Expected six positive execution velocity limits")
        rows, diagnostics = [], []
        for index, target in enumerate(targets):
            try:
                q, report = self.solve(target, current)
            except IKFailure as error:
                raise IKFailure(f"IK target {index}: {error}", {**error.diagnostics, "target_index": index}) from error
            speed = np.abs(q - current) * 30
            if check_speed and np.any(speed > velocity_limits + 1e-3):
                raise IKFailure(
                    f"IK target {index} exceeds execution joint velocity: {speed.tolist()}",
                    {
                        "target_index": index,
                        "joint_speed_rad_s": speed.tolist(),
                        "joint_velocity_limits": velocity_limits.tolist(),
                        "q": q.tolist(),
                        "seed": current.tolist(),
                    },
                )
            report["max_speed_rad_s"] = float(speed.max())
            rows.append(q)
            diagnostics.append(report)
            current = q
        return np.asarray(rows), diagnostics


def joint_prefix(left_kin, right_kin, targets, current, steps=15, *, arm_velocity_limit=None):
    current = np.asarray(current, float)
    if current.shape != (14,) or not 1 <= steps <= 30:
        raise ValueError("Invalid joint feedback or execution prefix")
    results = {}
    for side, kin, seed in (("left", left_kin, current[:6]), ("right", right_kin, current[7:13])):
        try:
            limits = None if arm_velocity_limit is None else np.full(6, arm_velocity_limit)
            results[side] = kin.sequence(targets[side + "_base_T_ee"][:steps], seed, velocity_limits=limits)
        except IKFailure as error:
            raise IKFailure(f"{side} arm: {error}", {**error.diagnostics, "arm": side}) from error
    left, ld = results["left"]
    right, rd = results["right"]
    points = np.concatenate(
        [
            left,
            targets["left_gripper"][:steps, None],
            right,
            targets["right_gripper"][:steps, None],
        ],
        axis=1,
    )
    return points, {"left": ld, "right": rd}
