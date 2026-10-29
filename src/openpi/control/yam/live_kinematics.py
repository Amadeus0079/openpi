"""Explicit live-run IK residual settings; original geometry and limits retained."""

import time

import numpy as np
from scipy.spatial.transform import Rotation

from openpi.control.yam.mapping import Calibration
from openpi.control.yam.mapping import IKFailure
from openpi.control.yam.mapping import Yam1ActionAdapter
from openpi.control.yam.mapping import Yam1Bridge
from openpi.control.yam.mapping import Yam1Kinematics

POSITION_TOLERANCE_M = 0.005
ROTATION_TOLERANCE_RAD = 0.01


class LiveKinematics(Yam1Kinematics):
    position_tolerance_m = POSITION_TOLERANCE_M
    rotation_tolerance_rad = ROTATION_TOLERANCE_RAD

    def set_tolerances(self, position_m, rotation_rad):
        if not np.isfinite([position_m, rotation_rad]).all() or not (
            0.0001 <= position_m <= 0.005 and 0.0001 <= rotation_rad <= 0.01
        ):
            raise ValueError("Live IK tolerances exceed the approved range")
        self.position_tolerance_m, self.rotation_tolerance_rad = position_m, rotation_rad

    def solve(self, target, seed, *, max_iters=200):
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
            if iteration >= 3 and pe <= self.position_tolerance_m and re <= self.rotation_tolerance_rad:
                if np.any(q < self.bounds[:, 0] - 1e-6) or np.any(q > self.bounds[:, 1] + 1e-6):
                    raise ValueError("IK exceeded joint limits")
                return q, {
                    "iterations": iteration,
                    "position_error_m": pe,
                    "rotation_error_rad": re,
                    "seconds": time.monotonic() - started,
                    "position_tolerance_m": self.position_tolerance_m,
                    "rotation_tolerance_rad": self.rotation_tolerance_rad,
                    "minimum_iterations": 3,
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
                "position_tolerance_m": self.position_tolerance_m,
                "rotation_tolerance_rad": self.rotation_tolerance_rad,
                "minimum_iterations": 3,
            },
        )


class LiveActionAdapter(Yam1ActionAdapter):
    def __init__(self, checkpoint, work):
        from pathlib import Path

        self.calibration = Calibration.load(checkpoint)
        self.bridge = Yam1Bridge(self.calibration)
        self.left = LiveKinematics(self.calibration, Path(work) / "left")
        self.right = LiveKinematics(self.calibration, Path(work) / "right")

    def set_tolerances(self, position_m, rotation_rad):
        for arm in (self.left, self.right):
            arm.set_tolerances(position_m, rotation_rad)
