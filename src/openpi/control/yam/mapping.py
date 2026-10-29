"""YAM-1 v3 actions -> native mounts -> joint plans; no hardware execution."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from openpi.control.yam.reference.legacy_geometry import CameraEEBridge
from openpi.control.yam.reference.legacy_geometry import pose
from openpi.control.yam.reference.legacy_kinematics import CheckpointKinematics
from openpi.control.yam.reference.legacy_kinematics import IKFailure
from openpi.control.yam.reference.legacy_kinematics import joint_prefix
from openpi.control.yam.reference.legacy_limits import collection_speed_limits

__all__ = ["Calibration", "IKFailure", "Yam1ActionAdapter", "Yam1Bridge", "Yam1Kinematics"]

TARGET_ID = "e5ee832e9d5566f8d7a06a17db9e03090422b2136cc8030bcc0d0757e69f45dc"
MAPPING_ID = "yam_pro_mount_x_forward_z_upper_20261028_v3"
NATIVE_T_POLICY = np.array([[0.0, 0.0, -1.0, 0.0], [0.0, -1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
JOINT_ORDER = [
    *[f"left_joint{i}" for i in range(1, 7)],
    "left_gripper",
    *[f"right_joint{i}" for i in range(1, 7)],
    "right_gripper",
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class Calibration:
    checkpoint: Path
    urdf: Path
    left_base_T_camera: np.ndarray
    right_base_T_camera: np.ndarray
    left_base_T_right_base: np.ndarray
    native_T_policy: np.ndarray
    camera_serials: dict
    hashes: dict
    model_sha256: str
    group: str
    step: int
    architecture: str

    @classmethod
    def load(cls, checkpoint):
        checkpoint = Path(checkpoint).resolve()
        manifest = json.loads((checkpoint / "inference-manifest.json").read_text())
        if manifest.get("schema") != "yam1-hybrid-inference-v1" or manifest["target_identity"] != TARGET_ID:
            raise ValueError("Require the new YAM-1 checkpoint")
        if not isinstance(manifest.get("group"), str) or type(manifest.get("step")) is not int or manifest["step"] <= 0:
            raise ValueError("Require a published training step")
        target = checkpoint / "assets/yam_hybrid/deployment/target"
        geometry = target / "geometry"
        for name, record in manifest["files"].items():
            if not name.startswith("assets/yam_hybrid/deployment/target/"):
                continue
            file = checkpoint / name
            if (
                not file.resolve().is_relative_to(checkpoint)
                or file.stat().st_size != record["size"]
                or sha(file) != record["sha256"]
            ):
                raise ValueError("Target asset identity mismatch: " + name)
        binding = json.loads((target / "target-calibration.json").read_text())
        if binding["machine"] != "yam/yam1" or binding["ee_mapping_id"] != MAPPING_ID:
            raise ValueError("Wrong YAM machine or EE mapping")
        if set(binding["files"]) != {"cameras.json", "published-main-camera.json", "robot.urdf", "axis-contract.json"}:
            raise ValueError("Unexpected calibration asset set")
        for name, digest in binding["files"].items():
            if sha(geometry / name) != digest:
                raise ValueError("Calibration binding mismatch: " + name)
        axes = json.loads((geometry / "axis-contract.json").read_text())
        if axes["status"] != "user_approved" or axes["mapping_id"] != MAPPING_ID:
            raise ValueError("Unapproved axes")
        matrix = pose(axes["native_T_policy_ee"])
        if not np.array_equal(matrix, NATIVE_T_POLICY) or not np.array_equal(binding["native_T_policy_ee"], matrix):
            raise ValueError("Different native mount / policy EE axes")
        contract = json.loads((target / "deployment.json").read_text())
        if (
            contract["fps"],
            contract["horizon"],
            contract["state_dim"],
            contract["action_dim"],
            contract["ee_origin"],
        ) != (30, 30, 22, 14, "gripper_mount"):
            raise ValueError("Unsupported YAM-1 action interface")
        if contract["dataset_identity"] != TARGET_ID or contract["native_T_policy_ee"] != matrix.tolist():
            raise ValueError("Training target and deployment geometry differ")
        cameras = json.loads((geometry / "cameras.json").read_text())
        published = json.loads((geometry / "published-main-camera.json").read_text())["default"]
        extra = cameras["extrinsics"]
        left = pose(extra["left_base_T_main_camera"])
        right = pose(extra["right_base_T_main_camera"])
        relative = pose(extra["left_base_T_right_base"])
        np.testing.assert_allclose(relative @ right, left, atol=1e-9, rtol=0)
        np.testing.assert_allclose(published["reference_T_camera"], left, atol=1e-12, rtol=0)
        if (
            published["matrix_convention"] != "reference_T_camera"
            or published["reference_frame"] != "left_base_reference"
        ):
            raise ValueError("Unexpected camera transform direction")
        serials = {k: v["serial"] for k, v in cameras["cameras"].items()}
        if serials != binding["camera_serials"]:
            raise ValueError("Camera serial mapping differs")
        return cls(
            checkpoint,
            geometry / "robot.urdf",
            left,
            right,
            relative,
            matrix,
            serials,
            binding["files"],
            manifest["files"]["model.safetensors"]["sha256"],
            manifest["group"],
            manifest["step"],
            manifest["architecture"],
        )

    def bridge_contract(self):
        return {
            "action_version": "camera_anchor_ee_yam1_v3",
            "state_dim": 22,
            "action_dim": 14,
            "horizon": 30,
            "fps": 30,
            "ee_definition": "gripper_mount",
            "ee_axes": "gripper_x_forward_y_left_z_up_v1",
            "ee_mapping_id": MAPPING_ID,
            "native_mount_T_policy_ee": self.native_T_policy.tolist(),
            "camera_names": ["main", "left_wrist", "right_wrist"],
            "action_times_s": (np.arange(1, 31) / 30).tolist(),
            "dataset_identity": TARGET_ID,
        }


class Yam1Bridge(CameraEEBridge):
    def __init__(self, calibration):
        # Reuse generic reset/encode/decode, replacing the legacy constructor's
        # hardcoded axis contract with the SHA-bound YAM-1 calibration.
        self.calibration = calibration
        self.contract = calibration.bridge_contract()
        self.native_mount_T_policy_ee = calibration.native_T_policy.copy()
        self.policy_ee_T_native_mount = np.linalg.inv(self.native_mount_T_policy_ee)
        self.camera0_T_left_base = None

    def reset_native(self, left_base_T_mount, right_base_T_mount):
        CameraEEBridge.reset(
            self,
            left_base_T_mount,
            right_base_T_mount,
            np.linalg.inv(self.calibration.left_base_T_camera),
            self.calibration.left_base_T_right_base,
        )

    def reset_recorded_clip(self, clip):
        if clip.get("machine") != "yam/yam1" or clip.get("split") != "target_support":
            raise ValueError("Offline fixture must be a YAM-1 support clip")
        export = clip["export"]
        if not np.array_equal(export["provenance"]["native_T_canonical_ee"], self.native_mount_T_policy_ee):
            raise ValueError("Recorded anchors use different tool axes")
        camera = pose(export["camera0_T_body"])
        np.testing.assert_allclose(camera, np.linalg.inv(self.calibration.left_base_T_camera), atol=1e-9, rtol=0)
        self.camera0_T_left_base = camera
        self.left_base_T_right_base = self.calibration.left_base_T_right_base.copy()
        self.anchors = tuple(pose(export[key]) for key in ["anchor_left", "anchor_right"])

    def validate_policy_metadata(self, metadata):
        for key, value in [
            ("action_version", "camera_anchor_ee_yam1_v3"),
            ("target_identity", TARGET_ID),
            ("ee_mapping_id", MAPPING_ID),
            ("state_dim", 22),
            ("action_dim", 14),
            ("horizon", 30),
            ("fps", 30),
        ]:
            if metadata.get(key) != value:
                raise ValueError("Model/robot interface mismatch: " + key)
        for key, value in [
            ("group", self.calibration.group),
            ("step", self.calibration.step),
            ("model_sha256", self.calibration.model_sha256),
        ]:
            if metadata.get(key) != value:
                raise ValueError("Selected model identity mismatch: " + key)
        if self.calibration.architecture == "domain_io" and (
            metadata.get("selected_head"),
            metadata.get("selected_head_index"),
        ) != ("yam1", 8):
            raise ValueError("M3 must explicitly use its YAM-1 head")
        if metadata.get("native_T_policy_ee") != self.native_mount_T_policy_ee.tolist():
            raise ValueError("Model/robot tool axes differ")

    def snapshot_anchors(self):
        self._ready()
        return {
            "schema": "yam1-episode-anchors-v1",
            "target_identity": TARGET_ID,
            "ee_mapping_id": MAPPING_ID,
            "camera0_T_left_base": self.camera0_T_left_base.tolist(),
            "left_base_T_right_base": self.left_base_T_right_base.tolist(),
            "left_camera0_T_policy_ee": self.anchors[0].tolist(),
            "right_camera0_T_policy_ee": self.anchors[1].tolist(),
        }

    def restore_anchors(self, value):
        if (value.get("schema"), value.get("target_identity"), value.get("ee_mapping_id")) != (
            "yam1-episode-anchors-v1",
            TARGET_ID,
            MAPPING_ID,
        ):
            raise ValueError("Wrong episode anchor identity")
        np.testing.assert_allclose(
            value["camera0_T_left_base"], np.linalg.inv(self.calibration.left_base_T_camera), atol=1e-9, rtol=0
        )
        np.testing.assert_allclose(
            value["left_base_T_right_base"], self.calibration.left_base_T_right_base, atol=1e-9, rtol=0
        )
        self.camera0_T_left_base = pose(value["camera0_T_left_base"])
        self.left_base_T_right_base = pose(value["left_base_T_right_base"])
        self.anchors = tuple(pose(value[k]) for k in ["left_camera0_T_policy_ee", "right_camera0_T_policy_ee"])

    def decode(self, actions):
        result = super().decode(actions)
        for side in ["left", "right"]:
            result[side + "_base_T_native_mount"] = result[side + "_base_T_ee"]
            result[side + "_base_T_policy_ee"] = result[side + "_base_T_ee"] @ self.native_mount_T_policy_ee
        return result


class Yam1Kinematics(CheckpointKinematics):
    def __init__(self, calibration, work):
        work = Path(work)
        work.mkdir(parents=True, exist_ok=True)
        root = ET.parse(calibration.urdf).getroot()
        # Numeric FK/IK preserves joint origins and axes, but does not perform
        # collision checking. Visual meshes are unnecessary and are not loaded.
        for link in root.findall("link"):
            for tag in ["visual", "collision"]:
                for child in list(link.findall(tag)):
                    link.remove(child)
        numerical = work / "yam-pro-numerical.urdf"
        numerical.write_text(ET.tostring(root, encoding="unicode"))
        super().__init__(numerical)
        self.source_urdf_sha256 = sha(calibration.urdf)
        self.numerical_urdf = numerical


class Yam1ActionAdapter:
    def __init__(self, checkpoint, work):
        self.calibration = Calibration.load(checkpoint)
        self.bridge = Yam1Bridge(self.calibration)
        self.left = Yam1Kinematics(self.calibration, Path(work) / "left")
        self.right = Yam1Kinematics(self.calibration, Path(work) / "right")

    def reset(self, joint_feedback):
        q = np.asarray(joint_feedback, float)
        if q.shape != (14,) or not np.isfinite(q).all():
            raise ValueError("Expected L6+Lg+R6+Rg feedback")
        self.bridge.reset_native(self.left.fk(q[:6]), self.right.fk(q[7:13]))

    def observation(self, joint_feedback, images, prompt, **kwargs):
        q = np.asarray(joint_feedback, float)
        if q.shape != (14,) or not np.isfinite(q).all():
            raise ValueError("Invalid joint feedback")
        return self.bridge.observation(
            self.left.fk(q[:6]), self.right.fk(q[7:13]), [q[6], q[13]], images, prompt, **kwargs
        )

    def plan(self, actions, joint_feedback, steps=15, *, velocity_mode="urdf", yam_owner=None):
        if velocity_mode not in ["urdf", "collection"]:
            raise ValueError("Unknown velocity limit mode")
        velocity = None
        source = {"rule": "checkpoint URDF velocity attributes", "values_rad_s": self.left.velocities.tolist()}
        if velocity_mode == "collection":
            if yam_owner is None:
                raise ValueError("Require YAM-1 source to derive its collection limit")
            source = collection_speed_limits(Path(yam_owner))
            velocity = source["arm_velocity_limit_rad_s"]
        targets = self.bridge.decode(actions)
        points, diagnostics = joint_prefix(
            self.left, self.right, targets, joint_feedback, steps=steps, arm_velocity_limit=velocity
        )
        return {
            "command_targets": points,
            "relative_time_s": np.arange(1, steps + 1) / 30,
            "joint_order": JOINT_ORDER,
            "ee_targets": targets,
            "ik": diagnostics,
            "velocity_source": source,
            "units": {"joint_angles": "rad", "translation": "m", "grippers": "normalized_0_closed_1_open"},
            "hardware_access": False,
            "hardware_motion": False,
            "executable_command_sent": False,
        }
