"""Camera-anchor EE bridge. All transforms map child coordinates into parent.

Only geometry and offline application live here; this module never commands hardware.
Requires numpy and scipy (the OpenPI environment already provides both).
"""

import numpy as np
from scipy.spatial.transform import Rotation


def pose(value):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError("Expected finite 4x4 rigid transform")
    if (
        not np.allclose(value[3], [0, 0, 0, 1], atol=1e-7)
        or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-6)
        or np.linalg.det(value[:3, :3]) < 0
    ):
        raise ValueError("Invalid rigid transform")
    return value.copy()


def delta(current, anchor):
    return np.r_[
        current[:3, 3] - anchor[:3, 3],
        Rotation.from_matrix(current[:3, :3] @ anchor[:3, :3].T).as_rotvec(),
    ]


EE_AXES = "gripper_x_forward_y_left_z_up_v1"
YAM_NATIVE_T_POLICY_EE = np.array(
    [[0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
)


def policy_mount_transform(contract):
    axes = contract.get("ee_axes")
    if axes is None:
        if "native_mount_T_policy_ee" in contract:
            raise ValueError("Tool transform requires explicit ee_axes")
        return np.eye(4)  # Immutable legacy checkpoints used native URDF mount axes.
    if axes != EE_AXES:
        raise ValueError("Unsupported EE axis convention")
    matrix = pose(contract["native_mount_T_policy_ee"])
    if not np.allclose(matrix, YAM_NATIVE_T_POLICY_EE, atol=1e-12, rtol=0):
        raise ValueError("YAM native mount / policy EE axis transform mismatch")
    return matrix


class CameraEEBridge:
    def __init__(self, contract):
        self.contract = dict(contract)
        if (
            contract["action_version"],
            contract["state_dim"],
            contract["action_dim"],
            contract["horizon"],
            contract["fps"],
            contract["ee_definition"],
        ) != ("camera_anchor_ee_v1", 22, 14, 30, 30, "gripper_mount"):
            raise ValueError("Unsupported camera EE deployment contract")
        self.native_mount_T_policy_ee = policy_mount_transform(contract)
        self.policy_ee_T_native_mount = np.linalg.inv(self.native_mount_T_policy_ee)
        self.camera0_T_left_base = None

    def reset(self, left_base_T_ee, right_base_T_ee, camera_T_left_base, left_base_T_right_base):
        """Reset from native URDF mount FK; policy anchors use the declared tool axes."""
        self.camera0_T_left_base = pose(camera_T_left_base)
        self.left_base_T_right_base = pose(left_base_T_right_base)
        self.anchors = (
            self.camera0_T_left_base @ pose(left_base_T_ee) @ self.native_mount_T_policy_ee,
            self.camera0_T_left_base
            @ self.left_base_T_right_base
            @ pose(right_base_T_ee)
            @ self.native_mount_T_policy_ee,
        )

    def reset_policy_anchors(
        self, left_base_T_policy_ee, right_base_T_policy_ee, camera_T_left_base, left_base_T_right_base
    ):
        """Recorded export anchors already use policy axes; do not rotate them twice."""
        self.reset(
            pose(left_base_T_policy_ee) @ self.policy_ee_T_native_mount,
            pose(right_base_T_policy_ee) @ self.policy_ee_T_native_mount,
            camera_T_left_base,
            left_base_T_right_base,
        )

    def _ready(self):
        if self.camera0_T_left_base is None:
            raise RuntimeError("Call episode reset before observation or decoding")

    def observation(
        self,
        left_base_T_ee,
        right_base_T_ee,
        grippers,
        images,
        prompt,
        *,
        camera_T_left_base=None,
        camera_valid=None,
        ee_valid=(True, True),
    ):
        self._ready()
        transforms = (
            self.camera0_T_left_base @ pose(left_base_T_ee) @ self.native_mount_T_policy_ee,
            self.camera0_T_left_base
            @ self.left_base_T_right_base
            @ pose(right_base_T_ee)
            @ self.native_mount_T_policy_ee,
        )
        state, valid = np.zeros(22, np.float32), np.ones(22, bool)
        for i, (current, anchor) in enumerate(zip(transforms, self.anchors, strict=True)):
            state[i * 7 : i * 7 + 6] = delta(current, anchor)
            g = grippers[i]
            present = g is not None and np.isfinite(g)
            if present and not 0 <= g <= 1:
                raise ValueError("Gripper must be normalized into [0,1]")
            state[i * 7 + 6] = float(g) if present else 0
            state[20 + i] = present
            valid[i * 7 : i * 7 + 7] = bool(ee_valid[i])
            valid[i * 7 + 6] &= present
        relative_camera = (
            np.eye(4)
            if camera_T_left_base is None
            else self.camera0_T_left_base @ np.linalg.inv(pose(camera_T_left_base))
        )
        state[14:20] = delta(relative_camera, np.eye(4))
        return {
            "state": state,
            "state_valid": valid,
            "images": images,
            "camera_valid": np.asarray(
                camera_valid if camera_valid is not None else [k in images for k in self.contract["camera_names"]],
                bool,
            ),
            "prompt": prompt,
        }

    def decode(self, actions, *, mount_T_tcp=None):
        self._ready()
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (self.contract["horizon"], 14) or not np.isfinite(actions).all():
            raise ValueError("Expected finite (30,14) unnormalized EE actions")
        output = {"relative_time_s": np.asarray(self.contract["action_times_s"], dtype=np.float64)}
        for i, name in enumerate(("left", "right")):
            x = actions[:, i * 7 : i * 7 + 7]
            camera_T_target = np.broadcast_to(np.eye(4), (len(x), 4, 4)).copy()
            camera_T_target[:, :3, 3] = self.anchors[i][:3, 3] + x[:, :3]
            camera_T_target[:, :3, :3] = Rotation.from_rotvec(x[:, 3:6]).as_matrix() @ self.anchors[i][:3, :3]
            camera_T_base = self.camera0_T_left_base @ (self.left_base_T_right_base if i else np.eye(4))
            # IK and TCP transforms consume the original URDF mount frame.
            target = np.linalg.inv(camera_T_base) @ camera_T_target @ self.policy_ee_T_native_mount
            if mount_T_tcp is not None:
                target = target @ pose(mount_T_tcp[name])
            output[name + "_base_T_ee"] = target
            output[name + "_gripper"] = np.clip(x[:, 6], 0, 1)
        return output
